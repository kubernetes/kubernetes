# Contributing to `govmomi`

## Getting started

First, fork the repository on GitHub to your personal account.

Change `$USER` in the examples below to your Github username if they are not the
same.

```bash
git clone https://github.com/vmware/govmomi.git && cd govmomi

# prevent accidentally pushing to vmware/govmomi
git config push.default nothing
git remote rename origin vmware

# add your fork
git remote add $USER git@github.com:$USER/govmomi.git

git fetch -av
```

## Contribution Flow

This is a rough outline of what a contributor's workflow looks like:

- Create an issue describing the feature/fix
- Create a topic branch from where you want to base your work.
- Make commits of logical units.
- Make sure your commit messages are in the proper format (see below).
- Push your changes to a topic branch in your fork of the repository.
- Submit a pull request to `vmware/govmomi`.

See [below](#format-of-the-commit-message) for details on commit best practices
and **supported prefixes**, e.g. `govc: <message>`.

> **Note:** If you are new to Git(hub) check out [Git rebase, squash...oh
> my!](https://www.mgasch.com/2021/05/git-basics/) for more details on how to
> successfully contribute to an open source project.

### Example 1 - Fix a Bug in `govmomi`

```bash
git checkout -b issue-<number> vmware/master
git add <files>
git commit -m "fix: ..." -m "Closes: #<issue-number>"
git push $USER issue-<number>
```

### Example 2 - Add a new (non-breaking) API to `govmomi`

```bash
git checkout -b issue-<number> vmware/master
git add <files>
git commit -m "Add API ..." -m "Closes: #<issue-number>"
git push $USER issue-<number>
```

### Example 3 - Add a Feature to `govc`

```bash
git checkout -b issue-<number> vmware/master
git add <files>
git commit -m "govc: Add feature ..." -m "Closes: #<issue-number>"
git push $USER issue-<number>
```
**Note**:  
To register the new `govc` command package, add a blank `_` import to `govmomi/govc/main.go`.

### Example 4 - Fix a Bug in `vcsim`

```bash
git checkout -b issue-<number> vmware/master
git add <files>
git commit -m "vcsim: Fix ..." -m "Closes: #<issue-number>"
git push $USER issue-<number>
```

### Example 5 - Document Breaking (API) Changes

Breaking changes, e.g. to the `govmomi` APIs, are highlighted in the `CHANGELOG`
and release notes when the keyword `BREAKING:` is used in the commit message
body. 

The text after `BREAKING:` is used in the corresponding highlighted section.
Thus these details should be stated at the body of the commit message.
Multi-line strings are supported.

```bash
git checkout -b issue-<number> vmware/master
git add <files>
cat << EOF | git commit -F -
Add ctx to funcXYZ

This commit introduces context.Context to function XYZ
Closes: #1234

BREAKING: Add ctx to funcXYZ()
EOF

git push $USER issue-<number>
```

### Stay in sync with Upstream

When your branch gets out of sync with the vmware/master branch, use the
following to update (rebase):

```bash
git checkout issue-<number>
git fetch -a
git rebase vmware/master
git push --force-with-lease $USER issue-<number>
```

### Updating Pull Requests

If your PR fails to pass CI or needs changes based on code review, it's ok to
add more commits stating the changes made, e.g. "Address review comments". This
is to assist the reviewer(s) to easily detect and review the recent changes.

In case of small PRs, it's ok to squash and force-push (see further below)
directly instead.

```bash
# incorporate review feedback
git add .

# create a fixup commit which will be merged into your (original) <commit>
git commit --fixup <commit>
git push $USER issue-<number>
```

Be sure to add a comment to the PR indicating your new changes are ready to
review, as Github does not generate a notification when you git push.

Once the review is complete, squash and push your final commit(s):

```bash
# squash all commits into one
# --autosquash will automatically detect and merge fixup commits
git rebase -i --autosquash vmware/master
git push --force-with-lease $USER issue-<number>
```

### Code Style

The coding style suggested by the Go community is used in `govmomi`. See the
[style doc](https://github.com/golang/go/wiki/CodeReviewComments) for details.

Try to limit column width to 120 characters for both code and markdown documents
such as this one.

### Format of the Commit Message

We follow the conventions described in [How to Write a Git Commit
Message](http://chris.beams.io/posts/git-commit/).

Be sure to include any related GitHub issue references in the commit message,
e.g. `Closes: #<number>`.

The [`CHANGELOG.md`](./CHANGELOG.md) and release page uses **commit message
prefixes** for grouping and highlighting. A commit message that
starts with `[prefix:] ` will place this commit under the respective
section in the `CHANGELOG`. 

The following example creates a commit referencing the `issue: 1234` and puts
the commit message in the `govc` `CHANGELOG` section:

```bash
git commit -s -m "govc: Add CLI command X" -m "Closes: #1234"
```

Currently the following prefixes are used:

- `api:` - Use for API-related changes
- `govc:` - Use for changes to `govc` CLI
- `vcsim:` - Use for changes to vCenter Simulator
- `chore:` - Use for repository related activities
- `fix:` - Use for bug fixes
- `docs:` - Use for changes to the documentation
- `examples:` - Use for changes to examples

If your contribution falls into multiple categories, e.g. `api` and `vcsim` it
is recommended to break up your commits using distinct prefixes.

### Running CI Checks and Tests
You can run both `make check` and `make test` from the top level of the
repository. 

While `make check` will catch formatting and import errors, it will not apply
any fixes. The developer is expected to do that.

## Reporting Bugs and Creating Issues

When opening a new issue, try to roughly follow the commit message format
conventions above.
