# How to create a new `govmomi` Release on Github

On every new tag matching `v*` pushed to the repository a Github Action Release
Workflow is executed. 

The Github Actions release [workflow](.github/workflows/govmomi-release.yaml)
uses [`goreleaser`](http://goreleaser.com/) ([configuration
file](.goreleaser.yml)) and automatically creates/pushes:

- Release artifacts for `govc` and `vcsim` to the
  [release](https://github.com/vmware/govmomi/releases) page, including
  `LICENSE.txt`, `README` and `CHANGELOG`
- Docker images for `vmware/govc` and `vmware/vcsim` to Docker Hub
- Source code

⚠️ **Note:** These steps can only be performed by maintainers or administrators
of this project.

## Verify `master` branch is up to date with the remote

```console
$ git checkout master
$ git fetch -avp
$ git diff master origin/master

# if your local and remote branches diverge run
$ git pull origin/master
```

⚠️ **Note:** These steps assume `origin` to point to the remote
`https://github.com/vmware/govmomi`, respectively
`git@github.com:vmware/govmomi`.


## Set `RELEASE_VERSION` variable

This variable is used and referenced in the subsequent commands. Set it to the
**upcoming** release version, adhering to the [semantic
versioning](https://semver.org/) scheme:

```console
$ export RELEASE_VERSION=v0.25.0
```

## Create the Git Tag

```console
$ git tag -a ${RELEASE_VERSION} -m "Release ${RELEASE_VERSION}"
```

## Push the new Tag

```console
# Will trigger Github Actions Release Workflow
$ git push origin refs/tags/${RELEASE_VERSION}
```

## Verify Github Action Release Workflow

After pushing a new release tag, the status of the
workflow can be inspected
[here](https://github.com/vmware/govmomi/actions/workflows/govmomi-release.yaml).

![Release](static/release-workflow.png "Successful Release Run")

After a successful release, a pull request is automatically created by the
Github Actions bot to update the [CHANGELOG](CHANGELOG.md). This `CHANGELOG.md`
is also generated with `git-chglog` but uses a slightly different template
(`.chglog/CHANGELOG.tpl.md`) for rendering (issue/PR refs are excluded).
