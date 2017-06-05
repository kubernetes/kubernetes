# etcd release guide

The guide talks about how to release a new version of etcd.

The procedure includes some manual steps for sanity checking but it can probably be further scripted. Please keep this document up-to-date if you want to make changes to the release process. 

## Prepare Release

Set desired version as environment variable for following steps. Here is an example to release 2.1.3:

```
export VERSION=v2.1.3
export PREV_VERSION=v2.1.2
```

All releases version numbers follow the format of [semantic versioning 2.0.0](http://semver.org/).

### Major, Minor Version Release, or its Pre-release

- Ensure the relevant milestone on GitHub is complete. All referenced issues should be closed, or moved elsewhere.
- Remove this release from [roadmap](https://github.com/coreos/etcd/blob/master/ROADMAP.md), if necessary.
- Ensure the latest upgrade documentation is available.
- Bump [hardcoded MinClusterVerion in the repository](https://github.com/coreos/etcd/blob/master/version/version.go#L29), if necessary.
- Add feature capability maps for the new version, if necessary.

### Patch Version Release

- Discuss about commits that are backported to the patch release. The commits should not include merge commits.
- Cherry-pick these commits starting from the oldest one into stable branch.

## Write Release Note


- Write introduction for the new release. For example, what major bug we fix, what new features we introduce or what performance improvement we make.
- Write changelog for the last release. ChangeLog should be straightforward and easy to understand for the end-user.
- Put `[GH XXXX]` at the head of change line to reference Pull Request that introduces the change. Moreover, add a link on it to jump to the Pull Request.

## Tag Version

- Bump [hardcoded Version in the repository](https://github.com/coreos/etcd/blob/master/version/version.go#L30) to the latest version `${VERSION}`.
- Ensure all tests on CI system are passed.
- Manually check etcd is buildable in Linux, Darwin and Windows.
- Manually check upgrade etcd cluster of previous minor version works well.
- Manually check new features work well.
- Add a signed tag through `git tag -s ${VERSION}`.
- Sanity check tag correctness through `git show tags/$VERSION`.
- Push the tag to GitHub through `git push origin tags/$VERSION`. This assumes `origin` corresponds to "https://github.com/coreos/etcd".

## Build Release Binaries and Images

- Ensure `actool` is available, or installing it through `go get github.com/appc/spec/actool`.
- Ensure `docker` is available.

Run release script in root directory:

```
./scripts/release.sh ${VERSION}
```

It generates all release binaries and images under directory ./release.

## Sign Binaries and Images

Choose appropriate private key to sign the generated binaries and images.

The following commands are used for public release sign:

```
cd release
# personal GPG is okay for now
for i in etcd-*{.zip,.tar.gz}; do gpg --sign ${i}; done
# use `CoreOS ACI Builder <release@coreos.com>` secret key
gpg -u 88182190 -a --output etcd-${VERSION}-linux-amd64.aci.asc --detach-sig etcd-${VERSION}-linux-amd64.aci
```

## Publish Release Page in GitHub

- Set release title as the version name.
- Follow the format of previous release pages.
- Attach the generated binaries, aci image and signatures.
- Select whether it is a pre-release.
- Publish the release!

## Publish Docker Image in Quay.io

- Push docker image:

```
docker login quay.io
docker push quay.io/coreos/etcd:${VERSION}
```

- Add `latest` tag to the new image on [quay.io](https://quay.io/repository/coreos/etcd?tag=latest&tab=tags) if this is a stable release.

## Announce to etcd-dev Googlegroup

- Follow the format of [previous release emails](https://groups.google.com/forum/#!forum/etcd-dev).
- Make sure to include a list of authors that contributed since the previous release - something like the following might be handy:

```
git log ...${PREV_VERSION} --pretty=format:"%an" | sort | uniq | tr '\n' ',' | sed -e 's#,#, #g' -e 's#, $##'
```

- Send email to etcd-dev@googlegroups.com

## Post Release

- Create new stable branch through `git push origin ${VERSION_MAJOR}.${VERSION_MINOR}` if this is a major stable release. This assumes `origin` corresponds to "https://github.com/coreos/etcd".
- Bump [hardcoded Version in the repository](https://github.com/coreos/etcd/blob/master/version/version.go#L30) to the version `${VERSION}+git`.
