# How to create a `govmomi` Release on Github

> **Note** 
>
> The steps outlined in this document can only be performed by maintainers or
> administrators of this project.

The release automation is based on Github
[Actions](https://github.com/features/actions) and has been improved over time
to simplify the experience for creating `govmomi` releases.

The Github Actions release [workflow](.github/workflows/govmomi-release.yaml)
uses [`goreleaser`](http://goreleaser.com/) and automatically creates/pushes:

- Release artifacts for `govc` and `vcsim` to the
  [release](https://github.com/vmware/govmomi/releases) page, including
  `LICENSE.txt`, `README` and `CHANGELOG`
- Docker images for `vmware/govc` and `vmware/vcsim` to Docker Hub
- Source code

Starting with release tag `v0.29.0`, releases are not tagged on the `master`
branch anymore but a dedicated release branch, for example `release-0.29`. This
process has already been followed for patch releases and back-ports.

> **Warning** 
>
> If you create a release after the `v0.29.0` tag, start
> [here](#creating-a-release-after-v0290). To create a release with an older
> tag, e.g. cherrypick or back-port, continue
> [here](#creating-a-release-before-v0290).

## Creating a release after Version `v0.29.0`

The release process from `v0.29.0` has been further simplified and is done
through the Github UI. The only pre-requirement is creating a release branch,
which can be done through the Github UI or `git` CLI.

This guide describes the CLI process.

### Verify `master` branch is up to date with the remote

```console
git checkout master
git fetch -avp
git diff master origin/master

# if your local and remote branches diverge run
git pull origin/master
```

> **Warning** 
>
> These steps assume `origin` to point to the remote
> `https://github.com/vmware/govmomi`, respectively
> `git@github.com:vmware/govmomi`.

### Create a release branch

For new releases, create a release branch from the most recent commit in
`master`, e.g. `release-0.30`.

```console
export RELEASE_BRANCH=release-0.30
git checkout -b ${RELEASE_BRANCH}
```

For maintenance/patch releases on **existing** release branches **after** tag
`v0.29.0` simply checkout the existing release branch and add commits to the
existing release branch.

### Verify `make docs` and `CONTRIBUTORS` are up to date

> **Warning**
> 
> Run the following commands and commit any changes to the release branch before
> proceeding with the release.

```console
make doc
./scripts/contributors.sh
if [ -z "$(git status --porcelain)" ]; then 
  echo "working directory clean: proceed with release"
else 
  echo "working directory dirty: please commit changes"
fi

# perform git add && git commit ... in case there were changes
```

### Push the release branch

> **Warning**
>
> Do not create a tag as this will be done by the release automation.

The final step is pushing the new/updated release branch. 

```console
git push origin ${RELEASE_BRANCH}
```

### Create a release in the Github UI

Open the `govmomi` Github [repository](https://github.com/vmware/govmomi) and
navigate to `Actions -> Workflows -> Release`.

Click `Run Workflow` which opens a dropdown list.

Select the new/updated branch, e.g. `release-0.30`, i.e. **not** the `master`
branch.

Specify a semantic `tag` to associate with the release, e.g. `v0.30.0`. 

> **Warning**
>
> This tag **must not** exist or the release will fail during the validation
> phase.

By default, a dry-run is performed to rule out most (but not all) errors during
a release. If you do not want to perform a dry-run, e.g. to finally create a
release, deselect the `Verify release workflow ...` checkbox.

Click `Run Workflow` to kick off the workflow.

After successful completion and if the newly created `tag` is the **latest**
(semantic version sorted) tag in the repository, a PR is automatically opened
against the `master` branch to update the `CHANGELOG`. Please review and merge
accordingly.

## Creating a release before Version `v0.29.0`

The release process before `v0.29.0` differs since it's based on manually
creating and pushing tags. Here, on every new tag matching `v*` pushed to the
repository a Github Action Release Workflow is executed. 

### Verify `master` branch is up to date with the remote

```console
git checkout master
git fetch -avp
git diff master origin/master

# if your local and remote branches diverge run
git pull origin/master
```

> **Warning** 
>
> These steps assume `origin` to point to the remote
> `https://github.com/vmware/govmomi`, respectively
> `git@github.com:vmware/govmomi`.

### Create a release branch

Pick a reference (commit, branch or tag) **before** the `v0.29.0` tag and create
a release branch from there.

The following example creates a cherrypick release (`v0.28.1`) based on the
`v0.28.0` tag.

```console
export RELEASE_BRANCH=release-0.28
git checkout -b ${RELEASE_BRANCH} v0.28.0
```

Optionally, incorporate (cherry-pick) commits into the branch. 

> **Warning** 
>
> Make sure that these commits/ranges do not contain commits after the `v0.29.0`
> tag which include release automation changes, i.e. files in `.github/workflows/`!

### Verify `make docs` and `CONTRIBUTORS` are up to date

> **Warning**
> 
> Run the following commands and commit any changes to the release branch before
> proceeding with the release.

```console
make doc
./scripts/contributors.sh
if [ -z "$(git status --porcelain)" ]; then 
  echo "working directory clean: proceed with release"
else 
  echo "working directory dirty: please commit changes"
fi

# perform git add && git commit ... in case there were changes
```

### Set `RELEASE_VERSION` variable

This variable is used and referenced in the subsequent commands. Set it to the
**upcoming** release version, adhering to the [semantic
versioning](https://semver.org/) scheme:

```console
export RELEASE_VERSION=v0.28.1
```

### Create the Git Tag

```console
git tag -a ${RELEASE_VERSION} -m "Release ${RELEASE_VERSION}"
```

### Push the new Tag

```console
# Will trigger Github Actions Release Workflow
git push --atomic origin ${RELEASE_BRANCH} refs/tags/${RELEASE_VERSION}
```

### Verify Github Action Release Workflow

After pushing a new release tag, the status of the workflow can be inspected
[here](https://github.com/vmware/govmomi/actions/workflows/govmomi-release.yaml).

![Release](static/release-workflow.png "Successful Release Run")

After a successful release, a pull request is automatically created by the
Github Actions bot to update the [CHANGELOG](CHANGELOG.md). This `CHANGELOG.md`
is also generated with `git-chglog` but uses a slightly different template
(`.chglog/CHANGELOG.tpl.md`) for rendering (issue/PR refs are excluded).
