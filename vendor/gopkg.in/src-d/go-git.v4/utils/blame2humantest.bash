#!/bin/bash

set -e

repo=`git remote show origin | grep Fetch | cut -d' ' -f5`
branch="master"
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

blames=`git blame --root $path | cut -d' ' -f1`
declare -a blame
i=0
for shortBlame in $blames ; do
    blame[$i]=`git show $shortBlame | head -1 | cut -d' ' -f2`
    i=`expr $i + 1`
done

# some remotes have the .git, other don't,
# repoDot makes sure all have
repoDot="${repo%.git}.git"

echo -e "\t{\"${repoDot}\", \"${branch}\", \"${commit}\", \"${path}\", concat(&[]string{},"
prev=""
count=1
for i in ${blame[@]} ; do
    if [ "${prev}" == "" ] ; then
        prev=$i
    elif [ "$prev" == "$i" ] ; then
        count=`expr $count + 1`
    else
        echo -e "\t\trepeat(\"${prev}\", $count),"
        count=1
        prev=$i
    fi
done
echo -e "\t\trepeat(\"${prev}\", $count),"
echo -e "\t)},"
