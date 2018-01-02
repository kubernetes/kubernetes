# Release process

## Resulting artifacts
Creating a new release produces the following artifacts:

- Binaries (stored in the `release-<TAG>` directory) :
  - `cni-<PLATFORM>-<VERSION>.tgz` binaries
  - `cni-<VERSION>.tgz` binary (copy of amd64 platform binary)
  - `sha1`, `sha256` and `sha512` files for the above files.

## Preparing for a release
1. Releases are performed by maintainers and should usually be discussed and planned at a maintainer meeting.
  - Choose the version number. It should be prefixed with `v`, e.g. `v1.2.3`
  - Take a quick scan through the PRs and issues to make sure there isn't anything crucial that _must_ be in the next release.
  - Create a draft of the release note
  - Discuss the level of testing that's needed and create a test plan if sensible
  - Check what version of `go` is used in the build container, updating it if there's a new stable release.

## Creating the release artifacts
1. Make sure you are on the master branch and don't have any local uncommitted changes.
1. Create a signed tag for the release `git tag -s $VERSION` (Ensure that GPG keys are created and added to GitHub)
1. Run the release script from the root of the repository
  - `scripts/release.sh`
  - The script requires Docker and ensures that a consistent environment is used.
  - The artifacts will now be present in the `release-<TAG>` directory.
1. Test these binaries according to the test plan.

## Publishing the release
1. Push the tag to git `git push origin <TAG>`
1. Create a release on Github, using the tag which was just pushed.
1. Attach all the artifacts from the release directory.
1. Add the release note to the release.
1. Announce the release on at least the CNI mailing, IRC and Slack.

