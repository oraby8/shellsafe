"""Rule-based risk rating for a shell command. Deterministic on purpose: the model never decides what is safe."""
import re
import shlex
from dataclasses import dataclass, field
from enum import IntEnum


class Risk(IntEnum):
    SAFE = 0
    CAUTION = 1
    DANGEROUS = 2


@dataclass
class Verdict:
    risk: Risk = Risk.SAFE
    reasons: list[str] = field(default_factory=list)

    def flag(self, risk: Risk, reason: str):
        self.risk = max(self.risk, risk)
        if reason not in self.reasons:
            self.reasons.append(reason)


# Paths whose recursive removal / permission change wrecks the system or the user's data.
CRITICAL_PATHS = re.compile(r"^(/|/\*|~|~/|~/\*|\$HOME/?|\*|\.|\./|\.\./?|/(bin|boot|dev|etc|home|lib|lib64|opt|proc|root|sbin|srv|sys|usr|var)/?\*?)$")
BLOCK_DEVICE = re.compile(r"/dev/(sd[a-z]|nvme\d|hd[a-z]|vd[a-z]|xvd[a-z]|mmcblk\d|disk\d)")

WHOLE_COMMAND_RULES = [
    (re.compile(r":\s*\(\s*\)\s*\{.*:\s*\|\s*:.*&.*\}"), Risk.DANGEROUS, "fork bomb: exhausts processes until the machine hangs"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(sudo\s+)?(ba|z|da|k)?sh\b"), Risk.DANGEROUS, "pipes a downloaded script straight into a shell without reviewing it"),
    (re.compile(r">\s*" + BLOCK_DEVICE.pattern), Risk.DANGEROUS, "writes directly to a disk device, destroying its data"),
    (re.compile(r"(^|[^>&0-9])>\s*/etc/"), Risk.DANGEROUS, "overwrites a system configuration file"),
]

# Commands that are risky whatever their arguments.
COMMAND_RISK = {
    "mkfs": (Risk.DANGEROUS, "formats a filesystem, erasing everything on it"),
    "wipefs": (Risk.DANGEROUS, "erases filesystem signatures from a device"),
    "shred": (Risk.DANGEROUS, "irreversibly overwrites file contents"),
    "fdisk": (Risk.DANGEROUS, "edits the disk partition table"),
    "parted": (Risk.DANGEROUS, "edits the disk partition table"),
    "sfdisk": (Risk.DANGEROUS, "edits the disk partition table"),
    "shutdown": (Risk.CAUTION, "shuts down or reboots the machine"),
    "reboot": (Risk.CAUTION, "reboots the machine"),
    "poweroff": (Risk.CAUTION, "powers off the machine"),
    "halt": (Risk.CAUTION, "halts the machine"),
    "killall": (Risk.CAUTION, "kills every process with that name"),
    "pkill": (Risk.CAUTION, "kills processes matching a pattern"),
    "truncate": (Risk.CAUTION, "changes file size and can discard data"),
}


def _segments(command: str) -> list[list[str]]:
    """Split a command line into simple commands (argv lists) at pipes and list operators."""
    argvs = []
    for seg in re.split(r"\|\||&&|[|;&\n]|\$\(|`", command):
        seg = seg.strip().rstrip(")")
        if not seg:
            continue
        try:
            argv = shlex.split(seg, comments=True)
        except ValueError:
            argv = seg.split()
        while argv and re.match(r"^\w+=", argv[0]):
            argv = argv[1:]
        if argv:
            argvs.append(argv)
    return argvs


def _flags(argv: list[str]) -> set[str]:
    """Short flags expanded ('-rf' -> {'r','f'}) plus long flags ('--force')."""
    out = set()
    for a in argv[1:]:
        if a.startswith("--"):
            out.add(a)
        elif a.startswith("-") and len(a) > 1:
            out.update(a[1:])
    return out


def _operands(argv: list[str]) -> list[str]:
    return [a for a in argv[1:] if not a.startswith("-")]


def _check_argv(argv: list[str], v: Verdict, elevated: bool):
    name = argv[0].rsplit("/", 1)[-1]
    flags, operands = _flags(argv), _operands(argv)

    if name in ("sudo", "doas"):
        v.flag(Risk.CAUTION, "runs with root privileges")
        rest = [a for a in argv[1:] if not a.startswith("-")]
        if rest:
            _check_argv(rest, v, elevated=True)
        return
    if name in ("xargs", "nohup", "time", "nice", "env", "watch") and len(argv) > 1:
        rest = argv[1:]
        while rest and rest[0].startswith("-"):
            rest = rest[1:]
        if rest:
            _check_argv(rest, v, elevated)
        return

    base = name.split(".")[0]  # mkfs.ext4 -> mkfs
    if base in COMMAND_RISK:
        v.flag(*COMMAND_RISK[base])

    if name == "rm":
        recursive = bool(flags & {"r", "R", "--recursive"})
        if recursive and any(CRITICAL_PATHS.match(o) for o in operands):
            v.flag(Risk.DANGEROUS, "recursively deletes a system, home or whole-directory path")
        elif recursive or "f" in flags or "--force" in flags:
            v.flag(Risk.CAUTION, "deletes files without asking; there is no undo")
        else:
            v.flag(Risk.CAUTION, "deletes files; there is no undo")
        if "--no-preserve-root" in flags:
            v.flag(Risk.DANGEROUS, "disables the protection against deleting /")
    elif name == "dd":
        if any(a.startswith("of=") and BLOCK_DEVICE.search(a) for a in argv):
            v.flag(Risk.DANGEROUS, "dd writes raw data onto a disk device")
        elif any(a.startswith("of=") for a in argv):
            v.flag(Risk.CAUTION, "dd overwrites the output file")
    elif name in ("chmod", "chown", "chgrp"):
        recursive = bool(flags & {"R", "--recursive"})
        if recursive and any(CRITICAL_PATHS.match(o) for o in operands):
            v.flag(Risk.DANGEROUS, f"recursive {name} on a system or home path can break the system")
        elif recursive:
            v.flag(Risk.CAUTION, f"recursive {name} changes every file below the path")
        if name == "chmod" and any(re.fullmatch(r"0?777|a\+rwx", o) for o in operands):
            v.flag(Risk.CAUTION, "makes files writable by every user")
    elif name == "find":
        if "-delete" in argv or any(a == "-exec" and i + 1 < len(argv) and argv[i + 1] in ("rm", "shred")
                                    for i, a in enumerate(argv)):
            v.flag(Risk.CAUTION, "find deletes every matching file")
    elif name == "mv" and operands and operands[-1] == "/dev/null":
        v.flag(Risk.DANGEROUS, "moving to /dev/null destroys the file")
    elif name == "crontab" and "r" in flags:
        v.flag(Risk.DANGEROUS, "removes all scheduled cron jobs without confirmation")
    elif name == "kill" and "-1" in argv:
        v.flag(Risk.DANGEROUS, "kills every process you are allowed to signal")
    elif name == "git":
        sub = operands[0] if operands else ""
        if sub == "push" and (flags & {"f", "--force", "--force-with-lease"}):
            v.flag(Risk.CAUTION, "force-push rewrites remote history")
        elif sub == "reset" and "--hard" in flags:
            v.flag(Risk.CAUTION, "discards uncommitted changes")
        elif sub == "clean" and "f" in flags:
            v.flag(Risk.CAUTION, "deletes untracked files")
    elif name == "sed" and ("i" in flags or "--in-place" in flags):
        v.flag(Risk.CAUTION, "edits the file in place")
    elif name in ("iptables", "nft") and flags & {"F", "--flush"}:
        v.flag(Risk.CAUTION, "flushes firewall rules")
    elif name == "docker" and operands[:2] in (["system", "prune"], ["volume", "prune"]):
        v.flag(Risk.CAUTION, "permanently removes Docker data")

    if elevated and v.risk == Risk.CAUTION and name in ("rm", "chmod", "chown", "dd", "find"):
        v.flag(Risk.DANGEROUS, "destructive operation run as root")


def assess(command: str) -> Verdict:
    v = Verdict()
    for pattern, risk, reason in WHOLE_COMMAND_RULES:
        if pattern.search(command):
            v.flag(risk, reason)
    # Plain `> file` truncates; `>>`, `2>`, `&>`, `>&` and process substitution `>(...)` are not flagged.
    for m in re.finditer(r"(^|[^>&0-9])>(?!>|&|\()\s*([^\s|&;]+)", command):
        if m.group(2) != "/dev/null":
            v.flag(Risk.CAUTION, "> overwrites the target file if it exists")
    for argv in _segments(command):
        _check_argv(argv, v, elevated=False)
    return v
