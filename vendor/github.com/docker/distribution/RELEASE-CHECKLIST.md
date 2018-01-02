## Registry Release Checklist

10. Compile release notes detailing features and since the last release.

  Update the `CHANGELOG.md` file and create a PR to master with the updates.
Once that PR has been approved by maintainers the change may be cherry-picked
to the release branch (new release branches may be forked from this commit).

20. Update the version file: `https://github.com/docker/distribution/blob/master/version/version.go`

30. Update the `MAINTAINERS` (if necessary), `AUTHORS` and `.mailmap` files.

```
make AUTHORS
```

40. Create a signed tag.

   Distribution uses semantic versioning.  Tags are of the format
`vx.y.z[-rcn]`. You will need PGP installed and a PGP key which has been added
to your Github account. The comment for the tag should include the release
notes, use previous tags as a guide for formatting consistently. Run
`git tag -s vx.y.z[-rcn]` to create tag and `git -v vx.y.z[-rcn]` to verify tag,
check comment and correct commit hash.

50. Push the signed tag

60. Create a new [release](https://github.com/docker/distribution/releases).  In the case of a release candidate, tick the `pre-release` checkbox. 

70. Update the registry binary in [distribution library image repo](https://github.com/docker/distribution-library-image) by running the update script and  opening a pull request.

80. Update the official image.  Add the new version in the [official images repo](https://github.com/docker-library/official-images) by appending a new version to the `registry/registry` file with the git hash pointed to by the signed tag.  Update the major version to point to the latest version and the minor version to point to new patch release if necessary.
e.g. to release `2.3.1`

   `2.3.1 (new)`

   `2.3.0 -> 2.3.0` can be removed

   `2 -> 2.3.1`

   `2.3 -> 2.3.1`

90. Build a new distribution/registry image on [Docker hub](https://hub.docker.com/u/distribution/dashboard) by adding a new automated build with the new tag and re-building the images.

