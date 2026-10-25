#!/usr/bin/env bash
# Build the pristine fixture trees that every verified command starts from.
set -euo pipefail
R=$1
H=$R/home/user
mkdir -p $H/{Documents,Downloads,Pictures,projects/webapp/src,projects/webapp/.git-placeholder,logs,backup,"my folder",.config/app,tmp}
cd $H
printf 'line one\nline two\nerror: disk full\nwarning: low memory\nline five\n' > notes.txt
printf 'hello world\nHello World\nfoo bar baz\nTODO: fix this\n' > Documents/readme.md
for i in $(seq 1 50); do echo "2026-10-$(printf %02d $((i%28+1))) 12:$((i%60)):00 INFO request id=$i status=$((200 + (i%3)*100)) path=/api/v$((i%2+1))/items"; done > logs/access.log
for i in $(seq 1 20); do echo "$(date -d "2026-10-01 + $i hours" '+%b %d %H:%M:%S') host sshd[$((1000+i))]: Failed password for root from 10.0.0.$i port 22"; done > logs/auth.log
printf 'name,age,city,salary\nalice,30,Paris,5000\nbob,25,Berlin,4200\ncarol,35,Paris,6100\ndave,28,Madrid,3900\n' > Documents/people.csv
printf '{"users":[{"name":"alice","active":true,"roles":["admin"]},{"name":"bob","active":false,"roles":[]}],"version":"1.2.3"}\n' > .config/app/config.json
printf 'PORT=8080\nDEBUG=false\nDB_HOST=localhost\n' > projects/webapp/.env
printf '#include <stdio.h>\nint main(){printf("hi\\n");return 0;}\n' > projects/webapp/src/main.c
printf 'import os\nprint(os.getcwd())\n# TODO: refactor\n' > projects/webapp/src/app.py
printf '<?php echo "hello"; ?>\n' > projects/webapp/src/index.php
printf '<html><body><h1>Title</h1><a href="https://example.com">x</a></body></html>\n' > projects/webapp/index.html
printf 'def add(a, b):\n    return a + b\n' > projects/webapp/src/util.py
printf 'space name\n' > "my folder/file with spaces.txt"
printf 'secret\n' > .hidden_file
dd if=/dev/urandom of=Downloads/big.bin bs=1M count=3 status=none
dd if=/dev/zero of=Downloads/small.bin bs=1K count=4 status=none
convert -size 64x48 xc:red Pictures/red.png
convert -size 640x480 xc:blue Pictures/photo.jpg
tar czf backup/docs.tar.gz -C $H Documents
(cd $H && zip -qr backup/projects.zip projects)
gzip -c notes.txt > backup/notes.txt.gz
cp notes.txt "backup/notes copy.txt"
touch -d '2026-01-15' backup/docs.tar.gz Downloads/small.bin
touch -d '2025-06-01' "backup/notes copy.txt"
(cd projects/webapp && git init -q && git -c user.email=u@x -c user.name=u add -A && git -c user.email=u@x -c user.name=u commit -qm init && echo "change" >> src/app.py)
ln -s notes.txt link_to_notes
# Absolute-path trees used by many public NL2Bash-style datasets.
mkdir -p $R/testbed/dir1/subdir1 $R/testbed/dir2 $R/system/folder1 $R/system/folder2 $R/workspace/src
cp notes.txt $R/testbed/textfile.txt; cp projects/webapp/src/main.c $R/testbed/hello.c
cp projects/webapp/src/index.php $R/testbed/hello.php; cp projects/webapp/index.html $R/testbed/FooBar.html
printf 'a\nb\nc\n' > $R/testbed/dir1/subdir1/list.txt; cp Documents/people.csv $R/testbed/dir2/data.csv
printf 'log1\n' > $R/system/folder1/a.log; printf 'doc\n' > $R/system/folder1/b.txt; cp logs/access.log $R/system/folder2/
cp projects/webapp/src/*.py $R/workspace/src/
chown -R 1001:1001 $R
