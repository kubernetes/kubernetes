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
release=(github-release release --draft --name "${name}")

case "$1" in
  release)
    tag="${name}"
    ;;
  prerelease)
    tag="prerelease-${name}"
    release+=(--pre-release)
    ;;
  dryrun)
    ;;
  *)
    echo "Usage: $0 [release|prerelease]"
    exit 1
    ;;
esac

echo "Building govc..."
rm -f ./govc_*
./build.sh

for name in govc_* ; do
  if [ "${name: -4}" == ".exe" ] ; then
    zip "${name}.zip" "$name" &
  else
    gzip -f "$name" &
  fi
done

wait

if [ -n "$tag" ] ; then
  echo "Pushing tag ${tag}..."
  git tag -f "${tag}"
  git push origin "refs/tags/${tag}"
fi

# Generate description
description=$(
if [[ "${tag}" == "prerelease-"* ]]; then
  echo '**This is a PRERELEASE version.**'
fi

echo "
Documentation:

* [CHANGELOG](https://github.com/vmware/govmomi/blob/$tag/govc/CHANGELOG.md)

* [README](https://github.com/vmware/govmomi/blob/$tag/govc/README.md)

* [USAGE](https://github.com/vmware/govmomi/blob/$tag/govc/USAGE.md)

The binaries below are provided without warranty, following the [Apache license](LICENSE).
"

echo '
Instructions:
* Download the file relevant to your operating system
* Decompress (i.e. `gzip -d govc_linux_amd64.gz`)
* Set the executable bit (i.e. `chmod +x govc_linux_amd64`)
* Move the file to a directory in your `$PATH` (i.e. `mv govc_linux_amd64 /usr/local/bin/govc`)
'

echo '```'
echo '$ sha1sum govc_*.gz'
sha1sum govc_*.gz
echo '$ sha1sum govc_*.zip'
sha1sum govc_*.zip
echo '```'
)

release+=(--tag "${tag}" --description "${description}")

if [ -n "$tag" ] ; then
  echo "Creating release..."
  "${release[@]}"
else
  echo "${release[@]}"
fi

# Upload build artifacts
for f in govc_*.{gz,zip}; do
  echo "Uploading $f..."
  if [ -n "$tag" ] ; then
    github-release upload --tag "${tag}" --name "${f}" --file "${f}"
  fi
done

echo "Remember to publish the draft release!"
