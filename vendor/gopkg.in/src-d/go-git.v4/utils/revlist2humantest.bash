#!/bin/bash

# you can run this over a whole repo with:
#
# for file in `find . -type f | sed 's/^\.\///' | egrep -v '^\.git\/.*$'` ; do revlist2humantest.bash $file ; done > /tmp/output
#
# be careful with  files with spaces, though

set -e

repo=`git remote show origin | grep Fetch | cut -d' ' -f5`
branch=`git branch | egrep '^\* .*' | cut -d' ' -f2`
if [ "$#" -eq 1 ] ; then
    commit=`git log | head -1 | cut -d' ' -f2`
    path=$1
elif [ "$#" -eq 2 ] ; then
    commit=$1
    path=$2
else
    echo "bad number of parameters" > /dev/stderr
    echo > /dev/stderr
    echo "    try with: [commit] path" > /dev/stderr
    exit
fi

hashes=`git rev-list --remove-empty --reverse $commit -- $path`

# some remotes have the .git, other don't,
# repoDot makes sure all have
repoDot="${repo%.git}.git"

echo -e "\t&humanTest{\"${repoDot}\", \"${branch}\", \"${commit}\", \"${path}\", []string{"
for i in $hashes ; do
        echo -e "\t\t\"${i}\","
done
echo -e "\t}},"
