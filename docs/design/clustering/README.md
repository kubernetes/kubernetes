<!-- BEGIN MUNGE: UNVERSIONED_WARNING -->


<!-- END MUNGE: UNVERSIONED_WARNING -->
This directory contains diagrams for the clustering design doc.

This depends on the `seqdiag` [utility](http://blockdiag.com/en/seqdiag/index.html).
Assuming you have a non-borked python install, this should be installable with:

```sh
pip install seqdiag
```

Just call `make` to regenerate the diagrams.

## Building with Docker

If you are on a Mac or your pip install is messed up, you can easily build with
docker:

```sh
make docker
```

The first run will be slow but things should be fast after that.

To clean up the docker containers that are created (and other cruft that is left
around) you can run `make docker-clean`.

If you are using boot2docker and get warnings about clock skew (or if things
aren't building for some reason) then you can fix that up with
`make fix-clock-skew`.

## Automatically rebuild on file changes

If you have the fswatch utility installed, you can have it monitor the file
system and automatically rebuild when files have changed. Just do a
`make watch`.




<!-- BEGIN MUNGE: IS_VERSIONED -->
<!-- TAG IS_VERSIONED -->
<!-- END MUNGE: IS_VERSIONED -->


<!-- BEGIN MUNGE: GENERATED_ANALYTICS -->
[![Analytics](https://kubernetes-site.appspot.com/UA-36037335-10/GitHub/docs/design/clustering/README.md?pixel)]()
<!-- END MUNGE: GENERATED_ANALYTICS -->
