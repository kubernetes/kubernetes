#!/bin/sh
# This is a hack to import from a branch and call it a versioned snapshot.

OUTDIR=v1.0
REPORT_TAG=v1.0.1
BRANCH=release-1.0

set -e

TMPDIR=docs.$RANDOM

echo fetching upstream
git fetch upstream
go build ./_tools/release_docs
./release_docs --branch ${BRANCH} --output-dir $TMPDIR >/dev/null
rm ./release_docs

echo removing old
git rm -rf ${OUTDIR}/docs/ ${OUTDIR}/examples/ > /dev/null
rm -rf ${OUTDIR}/docs/ ${OUTDIR}/examples/

echo adding new
mv $TMPDIR/docs ${OUTDIR}
mv $TMPDIR/examples ${OUTDIR}
git add ${OUTDIR}/docs/ ${OUTDIR}/examples/ > /dev/null
rmdir $TMPDIR

echo stripping
for dir in docs examples; do
    find ${OUTDIR}/${dir} -type f -name \*.md | while read X; do
        sed -i \
            -e '/<!-- BEGIN STRIP_FOR_RELEASE.*/,/<!-- END STRIP_FOR_RELEASE.*/d' \
            -e "s|releases.k8s.io/HEAD|releases.k8s.io/${REPORT_TAG}|" \
            ${X}
    done
done

git status
