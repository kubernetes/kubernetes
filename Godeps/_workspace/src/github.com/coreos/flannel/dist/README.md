## Doing a release

To do a release, e.g. version 0.5.0, do the following steps.
This assumes that the remote that's hosting the project (i.e. https://github.com/coreos/flannel) is named "upstream".

```
VER=0.5.0
cd ./dist`

# Make two commits: v0.5.0 and v0.5.0+git; create a tag v0.5.0; push commits and tags to $ORIGIN
ORIGIN=upstream ./bump-release.sh $VER

# Build docker, ACI images and tarball
./build-release.sh $VER

# Publish to quay.io (credentials required)
./publish.sh $VER
```
