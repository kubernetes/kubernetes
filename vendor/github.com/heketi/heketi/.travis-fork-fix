#!/bin/sh

# Go imports point to github.com/pblcache/pblcache. When
# a fork is tested by Travis-ci, imports may point to code
# from pblcache repo and not from the fork.  This program
# will fix the issue.

REPO="github.com/heketi/heketi"
REPODIR="../../heketi"

if ! git remote -v | grep origin | grep ${REPO} ; then
    mkdir -p ${REPODIR}
    ln -s $PWD ${REPODIR}/heketi
fi
