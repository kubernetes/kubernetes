#!/bin/bash -eu
#
# $1 = version string (e.g. 0.8.0)

VERSION="${1:?version must be set}"
if [ "${VERSION:0:1}" == "v" ]; then
	echo "version tag shouldn't start with v" >> /dev/stderr
	exit 255
fi
ORIGIN="${ORIGIN:=upstream}"
VERSIONTAG="v${VERSION}"

TAGBR="v${VERSION}-tag"

replace_version() {
	sed -i -e "s/const Version.*/const Version = \"$1\"/" ../version.go
	git commit -m "version: bump to v$1" ../version.go
}

# make sure we're up to date
git pull --ff-only ${ORIGIN} master

# tag it
replace_version ${VERSION}
git tag -a -m "${VERSIONTAG}" "${VERSIONTAG}"

# bump ver to +git and push to origin
replace_version "${VERSION}+git"
git push "${ORIGIN}" master

# push the tag
git push "${ORIGIN}" "${VERSIONTAG}"

echo
echo "============================================================"
echo "Tagged $VERSIONTAG in $ORIGIN"
echo "Now run \"build-release.sh $VERSION\""
echo
