#!/bin/sh
# This is a hack to import from a branch and call it a versioned snapshot.

if [ $# != 4 ]; then
    echo "usage: $0 <version> <outdir> <report_tag> <branch>"
    echo
    echo "ARGS:"
    echo "  version: the major.minor version to report (matches the k8s.io URL)"
    echo "  outdir: where to write the output of this tool"
    echo "  report_tag: what string to report was imported (usually a git tag)"
    echo "  branch: what git branch to pull docs from"
    echo
    echo "example: $0 v1.0 _v1.0 v1.0.6 release-1.0"
    exit 1
fi

VERSION=$1
OUTDIR=$2
REPORT_TAG=$3
BRANCH=$4

set -e

SED=sed
if which gsed &>/dev/null; then
  SED=gsed
fi
if ! ($SED --version 2>&1 | grep -q GNU); then
  echo "!!! GNU sed is required.  If on OS X, use 'brew install gnu-sed'."
  exit 1
fi
tmpdir=docs.$RANDOM

echo fetching upstream
git fetch upstream
go build ./_tools/release_docs
./release_docs --branch ${BRANCH} --output-dir $tmpdir --version $VERSION >/dev/null
rm ./release_docs

echo removing old
git rm -rf ${OUTDIR}/docs/ ${OUTDIR}/examples/ > /dev/null
rm -rf ${OUTDIR}/docs/ ${OUTDIR}/examples/

echo adding new
mv $tmpdir/docs ${OUTDIR}
mv $tmpdir/examples ${OUTDIR}
git add ${OUTDIR}/docs/ ${OUTDIR}/examples/ > /dev/null
git add _includes/*{definitions,operations}.html
rmdir $tmpdir

echo stripping
for dir in docs examples; do
    find ${OUTDIR}/${dir} -type f -name \*.md | while read X; do
        $SED -ri \
            -e '/<!-- BEGIN STRIP_FOR_RELEASE.*/,/<!-- END STRIP_FOR_RELEASE.*/d' \
            -e "s|releases.k8s.io/HEAD|releases.k8s.io/${REPORT_TAG}|g" \
            ${X}
    done
    git stage ${OUTDIR}/${dir}
done

git status
