# rkt release guide

## Release cycle

This section describes the typical release cycle of rkt:

1. A GitHub [milestone][milestones] sets the target date for a future rkt release. Releases occur approximately every two to three weeks.
2. Issues grouped into the next release milestone are worked on in order of priority.
3. Changes are submitted for review in the form of a GitHub Pull Request (PR). Each PR undergoes review and must pass continuous integration (CI) tests before being accepted and merged into the main line of rkt source code.
4. The day before each release is a short code freeze during which no new code or dependencies may be merged. Instead, this period focuses on polishing the release, with tasks concerning:
  * Documentation
  * Usability tests
  * Issues triaging
  * Roadmap planning and scheduling the next release milestone
  * Organizational and backlog review
  * Build, distribution, and install testing by release manager

## Release process

This section shows how to perform a release of rkt.
Only parts of the procedure are automated; this is somewhat intentional (manual steps for sanity checking) but it can probably be further scripted, please help.
The following example assumes we're going from version 1.1.0 (`v1.1.0`) to 1.2.0 (`v1.2.0`).

Let's get started:

- Start at the relevant milestone on GitHub (e.g. https://github.com/coreos/rkt/milestones/v1.2.0): ensure all referenced issues are closed (or moved elsewhere, if they're not done). Close the milestone.
- Update the [roadmap][roadmap] to remove the release you're performing, if necessary
- Ensure that `stage1/aci/aci-manifest.in` is the same version of appc/spec vendored with rkt. Otherwise, update it.
- Branch from the latest master, make sure your git status is clean
- Ensure the build is clean!
  - `git clean -ffdx && ./autogen.sh && ./configure --enable-tpm=no --enable-functional-tests && make && make check` should work
  - Integration tests on CI should be green
- Update the [release notes][changelog].
  Try to capture most of the salient changes since the last release, but don't go into unnecessary detail (better to link/reference the documentation wherever possible).
  `scripts/changelog.sh` will help generating an initial list of changes. Correct/fix entries if necessary, and group them by category.

The rkt version is [hardcoded in the repository][configure_ac], so the first thing to do is bump it:

- Run `scripts/bump-release v1.2.0`.
  This should generate two commits: a bump to the actual release (e.g. v1.2.0, including CHANGELOG updates), and then a bump to the release+git (e.g. v1.2.0+git).
  The actual release version should only exist in a single commit!
- Sanity check what the script did with `git diff HEAD^^` or similar.
  As well as changing the actual version, it also attempts to fix a bunch of references in the documentation etc.
- If the script didn't work, yell at the author and/or fix it.
  It can almost certainly be improved.
- File a PR and get a review from another [maintainer][maintainers].
  This is useful to a) sanity check the diff, and b) be very explicit/public that a release is happening
- Ensure the CI on the release PR is green!
- Merge the PR

Check out the release commit and build it!

- `git checkout HEAD^` should work. You want to be at the commit where the version is without "+git". Sanity check configure.ac (2nd line).
- Build rkt inside rkt (so make sure you have rkt in your $PATH):
  - `export BUILDDIR=$PWD/release-build && mkdir -p $BUILDDIR && sudo BUILDDIR=$BUILDDIR ./scripts/build-rir.sh`
- Sanity check the binary:
  - Check `release-build/target/bin/rkt version`
  - Check `ldd release-build/target/bin/rkt`: it can contain linux-vdso.so, libpthread.so, libc.so, libdl.so and ld-linux-x86-64.so but nothing else.
  - Check `ldd release-build/target/tools/init`: same as above.
- Build convenience packages:
  - `sudo BUILDDIR=$BUILDDIR ./scripts/build-rir.sh --exec=./scripts/pkg/build-pkgs.sh -- 1.2.0` (add correct version)

Sign a tagged release and push it to GitHub:

- Grab the release key (see details below) and add a signed tag: `GIT_COMMITTER_NAME="CoreOS Application Signing Key" GIT_COMMITTER_EMAIL="security@coreos.com" git tag -u $RKTSUBKEYID'!' -s v1.2.0 -m "rkt v1.2.0"`
- Push the tag to GitHub: `git push --tags`

Now we switch to the GitHub web UI to conduct the release:

- Start a [new release][gh-new-release] on Github
- Tag "v1.2.0", release title "v1.2.0"
- Copy-paste the release notes you added earlier in [CHANGELOG.md][changelog]
- You can also add a little more detail and polish to the release notes here if you wish, as it is more targeted towards users (vs the changelog being more for developers); use your best judgement and see previous releases on GH for examples.
- Attach the release.
  This is a simple tarball:

```
export RKTVER="1.2.0"
export NAME="rkt-v$RKTVER"
mkdir $NAME
cp release-build/target/bin/rkt release-build/target/bin/stage1-{coreos,kvm,fly}.aci $NAME/
cp -r dist/* $NAME/
sudo chown -R root:root $NAME/
tar czvf $NAME.tar.gz --numeric-owner $NAME/
```

- Attach packages, as well as each stage1 file individually so they can be fetched by the ACI discovery mechanism:

```
cp release-build/target/bin/*.deb .
cp release-build/target/bin/*.rpm .
cp release-build/target/bin/stage1-coreos.aci stage1-coreos-$RKTVER-linux-amd64.aci
cp release-build/target/bin/stage1-kvm.aci stage1-kvm-$RKTVER-linux-amd64.aci
cp release-build/target/bin/stage1-fly.aci stage1-fly-$RKTVER-linux-amd64.aci
```

- Sign all release artifacts.

rkt project key must be used to sign the generated binaries and images.`$RKTSUBKEYID` is the key ID of rkt project Yubikey. Connect the key and run `gpg2 --card-status` to get the ID.
The public key for GPG signing can be found at [CoreOS Application Signing Key][coreos-key] and is assumed as trusted.

The following commands are used for public release signing:

```
for i in $NAME.tar.gz stage1-*.aci *.deb *.rpm; do gpg2 -u $RKTSUBKEYID'!' --armor --output ${i}.asc --detach-sign ${i}; done
for i in $NAME.tar.gz stage1-*.aci *.deb *.rpm; do gpg2 --verify ${i}.asc ${i}; done
```

- Once signed and uploaded, double-check that all artifacts and signatures are on github. There should be 8 files in attachments (1x tar.gz, 3x ACI, 4x armored signatures).

- Publish the release!

- Clean your git tree: `sudo git clean -ffdx`.

Now it's announcement time: send an email to rkt-dev@googlegroups.com describing the release.
Generally this is higher level overview outlining some of the major features, not a copy-paste of the release notes.
Use your discretion and see [previous release emails][rkt-dev-list] for examples.
Make sure to include a list of authors that contributed since the previous release - something like the following might be handy:

```
git log v1.1.0..v1.2.0 --pretty=format:"%an" | sort | uniq | tr '\n' ',' | sed -e 's#,#, #g' -e 's#, $#\n#'
```


[changelog]: https://github.com/coreos/rkt/blob/master/CHANGELOG.md
[configure_ac]: https://github.com/coreos/rkt/blob/master/configure.ac#L2
[coreos-key]: https://coreos.com/security/app-signing-key
[gh-new-release]: https://github.com/coreos/rkt/releases/new
[milestones]: https://github.com/coreos/rkt/milestones
[maintainers]: https://github.com/coreos/rkt/blob/master/MAINTAINERS
[rkt-dev-list]: https://groups.google.com/forum/#!forum/rkt-dev
[roadmap]: https://github.com/coreos/rkt/blob/master/ROADMAP.md
