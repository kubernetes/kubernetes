# Release Checklist
## A maintainer's guide to releasing Docker

So you're in charge of a Docker release? Cool. Here's what to do.

If your experience deviates from this document, please document the changes
to keep it up-to-date.

It is important to note that this document assumes that the git remote in your
repository that corresponds to "https://github.com/docker/docker" is named
"origin".  If yours is not (for example, if you've chosen to name it "upstream"
or something similar instead), be sure to adjust the listed snippets for your
local environment accordingly.  If you are not sure what your upstream remote is
named, use a command like `git remote -v` to find out.

If you don't have an upstream remote, you can add one easily using something
like:

```bash
export GITHUBUSER="YOUR_GITHUB_USER"
git remote add origin https://github.com/docker/docker.git
git remote add $GITHUBUSER git@github.com:$GITHUBUSER/docker.git
```

### 1. Pull from master and create a release branch

Note: Even for major releases, all of X, Y and Z in vX.Y.Z must be specified (e.g. v1.0.0).

```bash
export VERSION=vX.Y.Z
git fetch origin
git branch -D release || true
git checkout --track origin/release
git checkout -b bump_$VERSION
```

If it's a regular release, we usually merge master.
```bash
git merge origin/master
```

Otherwise, if it is a hotfix release, we cherry-pick only the commits we want.
```bash
# get the commits ids we want to cherry-pick
git log
# cherry-pick the commits starting from the oldest one, without including merge commits
git cherry-pick <commit-id>
git cherry-pick <commit-id>
...
```

### 2. Bump the API version on master

We don't want to stop contributions to master just because we are releasing. At
the same time, now that the release branch exists, we don't want API changes to
go to the now frozen API version.

Create a new entry in `docs/reference/api/` by copying the latest and
bumping the version number (in both the file's name and content), and submit
this in a PR against master.

### 3. Update CHANGELOG.md

You can run this command for reference with git 2.0:

```bash
git fetch --tags
LAST_VERSION=$(git tag -l --sort=-version:refname "v*" | grep -E 'v[0-9\.]+$' | head -1)
git log --stat $LAST_VERSION..bump_$VERSION
```

If you don't have git 2.0 but have a sort command that supports `-V`:
```bash
git fetch --tags
LAST_VERSION=$(git tag -l | grep -E 'v[0-9\.]+$' | sort -rV | head -1)
git log --stat $LAST_VERSION..bump_$VERSION
```

If releasing a major version (X or Y increased in vX.Y.Z), simply listing notable user-facing features is sufficient.
```markdown
#### Notable features since <last major version>
* New docker command to do something useful
* Remote API change (deprecating old version)
* Performance improvements in some usecases
* ...
```

For minor releases (only Z increases in vX.Y.Z), provide a list of user-facing changes.
Each change should be listed under a category heading formatted as `#### CATEGORY`.

`CATEGORY` should describe which part of the project is affected.
  Valid categories are:
  * Builder
  * Documentation
  * Hack
  * Packaging
  * Remote API
  * Runtime
  * Other (please use this category sparingly)

Each change should be formatted as `BULLET DESCRIPTION`, given:

* BULLET: either `-`, `+` or `*`, to indicate a bugfix, new feature or
  upgrade, respectively.

* DESCRIPTION: a concise description of the change that is relevant to the
  end-user, using the present tense. Changes should be described in terms
  of how they affect the user, for example "Add new feature X which allows Y",
  "Fix bug which caused X", "Increase performance of Y".

EXAMPLES:

```markdown
## 0.3.6 (1995-12-25)

#### Builder

+ 'docker build -t FOO .' applies the tag FOO to the newly built image

#### Remote API

- Fix a bug in the optional unix socket transport

#### Runtime

* Improve detection of kernel version
```

If you need a list of contributors between the last major release and the
current bump branch, use something like:
```bash
git log --format='%aN <%aE>' v0.7.0...bump_v0.8.0 | sort -uf
```
Obviously, you'll need to adjust version numbers as necessary.  If you just need
a count, add a simple `| wc -l`.

### 4. Change the contents of the VERSION file

Before the big thing, you'll want to make successive release candidates and get
people to test. The release candidate number `N` should be part of the version:

```bash
export RC_VERSION=${VERSION}-rcN
echo ${RC_VERSION#v} > VERSION
```

### 5. Test the docs

Make sure that your tree includes documentation for any modified or
new features, syntax or semantic changes.

To test locally:

```bash
make docs
```

To make a shared test at https://beta-docs.docker.io:

(You will need the `awsconfig` file added to the `docs/` dir)

```bash
make AWS_S3_BUCKET=beta-docs.docker.io BUILD_ROOT=yes docs-release
```

### 6. Commit and create a pull request to the "release" branch

```bash
git add VERSION CHANGELOG.md
git commit -m "Bump version to $VERSION"
git push $GITHUBUSER bump_$VERSION
echo "https://github.com/$GITHUBUSER/docker/compare/docker:release...$GITHUBUSER:bump_$VERSION?expand=1"
```

That last command will give you the proper link to visit to ensure that you
open the PR against the "release" branch instead of accidentally against
"master" (like so many brave souls before you already have).

### 7. Publish release candidate binaries

To run this you will need access to the release credentials. Get them from the
Core maintainers.

Replace "..." with the respective credentials:

```bash
docker build -t docker .
docker run \
       -e AWS_S3_BUCKET=test.docker.com \
       -e AWS_ACCESS_KEY="..." \
       -e AWS_SECRET_KEY="..." \
       -e GPG_PASSPHRASE="..." \
       -i -t --privileged \
       docker \
       hack/release.sh
```

It will run the test suite, build the binaries and packages, and upload to the
specified bucket, so this is a good time to verify that you're running against
**test**.docker.com.

After the binaries and packages are uploaded to test.docker.com, make sure
they get tested in both Ubuntu and Debian for any obvious installation
issues or runtime issues.

If everything looks good, it's time to create a git tag for this candidate:

```bash
git tag -a $RC_VERSION -m $RC_VERSION bump_$VERSION
git push origin $RC_VERSION
```

Announcing on multiple medias is the best way to get some help testing! An easy
way to get some useful links for sharing:

```bash
echo "Ubuntu/Debian: https://test.docker.com/ubuntu or curl -sSL https://test.docker.com/ | sh"
echo "Linux 64bit binary: https://test.docker.com/builds/Linux/x86_64/docker-${VERSION#v}"
echo "Darwin/OSX 64bit client binary: https://test.docker.com/builds/Darwin/x86_64/docker-${VERSION#v}"
echo "Darwin/OSX 32bit client binary: https://test.docker.com/builds/Darwin/i386/docker-${VERSION#v}"
echo "Linux 64bit tgz: https://test.docker.com/builds/Linux/x86_64/docker-${VERSION#v}.tgz"
```

We recommend announcing the release candidate on:

- IRC on #docker, #docker-dev, #docker-maintainers
- In a comment on the pull request to notify subscribed people on GitHub
- The [docker-dev](https://groups.google.com/forum/#!forum/docker-dev) group
- The [docker-maintainers](https://groups.google.com/a/dockerproject.org/forum/#!forum/maintainers) group
- Any social media that can bring some attention to the release candidate

### 8. Iterate on successive release candidates

Spend several days along with the community explicitly investing time and
resources to try and break Docker in every possible way, documenting any
findings pertinent to the release.  This time should be spent testing and
finding ways in which the release might have caused various features or upgrade
environments to have issues, not coding.  During this time, the release is in
code freeze, and any additional code changes will be pushed out to the next
release.

It should include various levels of breaking Docker, beyond just using Docker
by the book.

Any issues found may still remain issues for this release, but they should be
documented and give appropriate warnings.

During this phase, the `bump_$VERSION` branch will keep evolving as you will
produce new release candidates. The frequency of new candidates is up to the
release manager: use your best judgement taking into account the severity of
reported issues, testers availability, and time to scheduled release date.

Each time you'll want to produce a new release candidate, you will start by
adding commits to the branch, usually by cherry-picking from master:

```bash
git cherry-pick -x -m0 <commit_id>
```

You want your "bump commit" (the one that updates the CHANGELOG and VERSION
files) to remain on top, so you'll have to `git rebase -i` to bring it back up.

Now that your bump commit is back on top, you will need to update the CHANGELOG
file (if appropriate for this particular release candidate), and update the
VERSION file to increment the RC number:

```bash
export RC_VERSION=$VERSION-rcN
echo $RC_VERSION > VERSION
```

You can now amend your last commit and update the bump branch:

```bash
git commit --amend
git push -f $GITHUBUSER bump_$VERSION
```

Repeat step 6 to tag the code, publish new binaries, announce availability, and
get help testing.

### 9. Finalize the bump branch

When you're happy with the quality of a release candidate, you can move on and
create the real thing.

You will first have to amend the "bump commit" to drop the release candidate
suffix in the VERSION file:

```bash
echo $VERSION > VERSION
git add VERSION
git commit --amend
```

You will then repeat step 6 to publish the binaries to test

### 10. Get 2 other maintainers to validate the pull request

### 11. Publish final binaries

Once they're tested and reasonably believed to be working, run against
get.docker.com:

```bash
docker run \
       -e AWS_S3_BUCKET=get.docker.com \
       -e AWS_ACCESS_KEY="..." \
       -e AWS_SECRET_KEY="..." \
       -e GPG_PASSPHRASE="..." \
       -i -t --privileged \
       docker \
       hack/release.sh
```

### 12. Apply tag

It's very important that we don't make the tag until after the official
release is uploaded to get.docker.com!

```bash
git tag -a $VERSION -m $VERSION bump_$VERSION
git push origin $VERSION
```

### 13. Go to github to merge the `bump_$VERSION` branch into release

Don't forget to push that pretty blue button to delete the leftover
branch afterwards!

### 14. Update the docs branch

If this is a MAJOR.MINOR.0 release, you need to make an branch for the previous release's
documentation:

```bash
git checkout -b docs-$PREVIOUS_MAJOR_MINOR
git fetch
git reset --hard origin/docs
git push -f origin docs-$PREVIOUS_MAJOR_MINOR
```

You will need the `awsconfig` file added to the `docs/` directory to contain the
s3 credentials for the bucket you are deploying to.

```bash
git checkout -b docs release || git checkout docs
git fetch
git reset --hard origin/release
git push -f origin docs
make AWS_S3_BUCKET=docs.docker.com BUILD_ROOT=yes DISTRIBUTION_ID=C2K6......FL2F docs-release
```

The docs will appear on https://docs.docker.com/ (though there may be cached
versions, so its worth checking http://docs.docker.com.s3-website-us-east-1.amazonaws.com/).
For more information about documentation releases, see `docs/README.md`.

Note that the new docs will not appear live on the site until the cache (a complex,
distributed CDN system) is flushed. The `make docs-release` command will do this
_if_ the `DISTRIBUTION_ID` is set correctly - this will take at least 15 minutes to run
and you can check its progress with the CDN Cloudfront Chrome addin.

### 15. Create a new pull request to merge your bump commit back into master

```bash
git checkout master
git fetch
git reset --hard origin/master
git cherry-pick $VERSION
git push $GITHUBUSER merge_release_$VERSION
echo "https://github.com/$GITHUBUSER/docker/compare/docker:master...$GITHUBUSER:merge_release_$VERSION?expand=1"
```

Again, get two maintainers to validate, then merge, then push that pretty
blue button to delete your branch.

### 16. Update the VERSION files

Now that version X.Y.Z is out, time to start working on the next! Update the
content of the `VERSION` file to be the next minor (incrementing Y) and add the
`-dev` suffix. For example, after 1.5.0 release, the `VERSION` file gets
updated to `1.6.0-dev` (as in "1.6.0 in the making").

### 17. Rejoice and Evangelize!

Congratulations! You're done.

Go forth and announce the glad tidings of the new release in `#docker`,
`#docker-dev`, on the [dev mailing list](https://groups.google.com/forum/#!forum/docker-dev),
the [announce mailing list](https://groups.google.com/forum/#!forum/docker-announce),
and on Twitter!
