# Verification sandbox: common CLI tools + a realistic fixture filesystem.
FROM ubuntu:24.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      bash coreutils findutils grep sed gawk diffutils procps psmisc util-linux bsdmainutils bsdextrautils \
      tree jq curl wget rsync zip unzip p7zip-full tar gzip bzip2 xz-utils zstd git file bc less nano vim-tiny \
      net-tools iproute2 iputils-ping dnsutils lsof sqlite3 python3 openssl imagemagick ffmpeg cron at acl attr \
      moreutils parallel tmux screen htop sysstat pv netcat-openbsd socat whois traceroute ripgrep fd-find \
      libimage-exiftool-perl poppler-utils ncal time tzdata locales gnupg md5deep dos2unix sharutils \
      man-db manpages && \
    rm -rf /var/lib/apt/lists/* && \
    ln -s /usr/bin/fdfind /usr/local/bin/fd && \
    useradd -m -u 1001 -s /bin/bash user
COPY make_fixture.sh /tmp/make_fixture.sh
RUN bash /tmp/make_fixture.sh /fixture-pristine && rm /tmp/make_fixture.sh
COPY worker.py /worker.py
ENTRYPOINT ["python3", "/worker.py"]
