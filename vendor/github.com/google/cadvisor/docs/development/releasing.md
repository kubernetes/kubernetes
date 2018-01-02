# cAdvisor Release Instructions

Google internal-only version: [cAdvisor Release Instructions](http://go/cadvisor-release-instructions)

## 1. Send Release PR

Example: https://github.com/google/cadvisor/pull/1281

Add release notes to [CHANGELOG.md](../../CHANGELOG.md)

- Tip: Use a github PR search to find changes since the last release
  `is:pr is:merged merged:>2016-04-21`

## 2. Create the release tag

### 2.a Create the release branch (only for major/minor releases)

Skip this step for patch releases.

```
# Sync to HEAD, or the commit to branch at
$ git fetch upstream && git checkout upstream/master
# Create the branch
$ git branch release-v0.XX
# Push it to upstream
$ git push git@github.com:google/cadvisor.git release-v0.XX
```

### 2.b Tag the release

For a release of minor version XX, patch version YY:

```
# Checkout the release branch
$ git fetch upstream && git checkout upstream/release-v0.XX
# Tag the release commit. If you aren't signing, ommit the -s
$ git tag -s -a v0.XX.YY
# Push it to upstream
$ git push git@github.com:google/cadvisor.git v0.XX.YY
```

## 3. Build release binary

Command: `make release`

- Make sure your git client is synced to the release cut point
- Try to build it from the release branch, since we include that in the binary version
- Verify the ldflags output, in particular check the Version, BuildUser, and GoVersion are expected

Once the build is complete, check the VERSION and note the sha256 hash.

## 4. Push the Docker images

Docker Hub:
```
$ docker login
Username: ****
Password: ****
$ docker push google/cadvisor:$VERSION
$ docker logout # Good practice with shared account
```

Google Container Registry:

```
$ gcloud auth login <account>
...
Go to the following link in your browser:

    https://accounts.google.com/o/oauth2/auth?<redacted>

Enter verification code: ****
$ gcloud docker push gcr.io/google_containers/cadvisor:$VERSION
$ gcloud auth revoke # Log out of shared account
```

## 5. Cut the release

Go to https://github.com/google/cadvisor/releases and click "Draft a new release"

- "Tag version" and "Release title" should be preceded by 'v' and then the version. Select the tag pushed in step 2.b
- Copy an old release as a template (e.g. github.com/google/cadvisor/releases/tag/v0.23.1)
- Body should start with release notes (from CHANGELOG.md)
- Next is the Docker image: `google/cadvisor:$VERSION`
- Next are the binary hashes (from step 3)
- Upload the binary build in step 3
- If this is an alpha or beta release, mark the release as a "pre-release"
- Click publish when done

## 6. Finalize the release

Once you are satisfied with the release quality (consider waiting a week for bug reports to come in), it is time to promote the release to *latest*

1. Edit the github release a final time, and uncheck the "Pre-release" checkbox
2. Tag the docker & gcr.io releases with the latest version
```
$ docker pull google/cadvisor:$VERSION
$ docker tag -f google/cadvisor:$VERSION google/cadvisor:latest
$ docker tag -f google/cadvisor:$VERSION gcr.io/google_containers/cadvisor:latest
```
3. Repeat steps 4.a and 4.b to push the image tagged with latest
