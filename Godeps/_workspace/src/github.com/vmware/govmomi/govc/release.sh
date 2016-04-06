#!/bin/bash -e

if ! which github-release > /dev/null; then
  echo 'Please install github-release...'
  echo ''
  echo '  $ go get github.com/aktau/github-release'
  echo ''
  exit 1
fi

if [ -z "${GITHUB_TOKEN}" ]; then
  echo 'Please set GITHUB_TOKEN...'
  exit 1
fi

export GITHUB_USER="${GITHUB_USER:-vmware}"
export GITHUB_REPO="${GITHUB_REPO:-govmomi}"

name="$(git describe)"

case "$1" in
  release)
    tag="${name}"
    ;;
  prerelease)
    tag="prerelease-${name}"
    ;;
  *)
    echo "Usage: $0 [release|prerelease]"
    exit 1
    ;;
esac

echo "Building govc..."
rm -f govc_*
./build.sh
gzip -f govc_*

echo "Pushing tag ${tag}..."
git tag -f "${tag}"
git push origin "refs/tags/${tag}"

# Generate description
description=$(
if [[ "${tag}" == "prerelease-"* ]]; then
  echo '**This is a PRERELEASE version.**'
fi

echo '
The binaries below are provided without warranty, following the [Apache license](LICENSE).
'

echo '
Instructions:
* Download the file relevant to your operating system
* Decompress (i.e. `gzip -d govc_linux_amd64.gz`)
* Set the executable bit (i.e. `chmod +x govc_linux_amd64`)
* Move the file to a directory in your `$PATH` (i.e. `mv govc_linux_amd64 /usr/local/bin`)
'

echo '```'
echo '$ sha1sum govc_*.gz'
sha1sum govc_*.gz
echo '```'
)

echo "Creating release..."
github-release release --tag "${tag}" --name "${name}" --description "${description}" --draft --pre-release

# Upload build artifacts
for f in govc_*.gz; do
  echo "Uploading $f..."
  github-release upload --tag "${tag}" --name "${f}" --file "${f}"
done

echo "Remember to publish the draft release!"
