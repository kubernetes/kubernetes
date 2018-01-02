# Apt & Yum Repository Maintenance
## A maintainer's guide to managing Docker's package repos

### How to clean up old experimental debs and rpms

We release debs and rpms for experimental nightly, so these can build up.
To remove old experimental debs and rpms, and _ONLY_ keep the latest, follow the
steps below.

1. Checkout docker master

2. Run clean scripts

```bash
docker build --rm --force-rm -t docker-dev:master .
docker run --rm -it --privileged \
    -v /path/to/your/repos/dir:/volumes/repos \
    -v $HOME/.gnupg:/root/.gnupg \
    -e GPG_PASSPHRASE \
    -e DOCKER_RELEASE_DIR=/volumes/repos \
    docker-dev:master hack/make.sh clean-apt-repo clean-yum-repo generate-index-listing sign-repos
```

3. Upload the changed repos to `s3` (if you host on s3)

4. Purge the cache, PURGE the cache, PURGE THE CACHE!

### How to get out of a sticky situation

Sh\*t happens. We know. Below are steps to get out of any "hash-sum mismatch" or
"gpg sig error" or the likes error that might happen to the apt repo.

**NOTE:** These are apt repo specific, have had no experience with anything similar
happening to the yum repo in the past so you can rest easy.

For each step listed below, move on to the next if the previous didn't work.
Otherwise CELEBRATE!

1. Purge the cache.

2. Did you remember to sign the debs after releasing?

Re-sign the repo with your gpg key:

```bash
docker build --rm --force-rm -t docker-dev:master .
docker run --rm -it --privileged \
    -v /path/to/your/repos/dir:/volumes/repos \
    -v $HOME/.gnupg:/root/.gnupg \
    -e GPG_PASSPHRASE \
    -e DOCKER_RELEASE_DIR=/volumes/repos \
    docker-dev:master hack/make.sh sign-repos
```

Upload the changed repo to `s3` (if that is where you host)

PURGE THE CACHE.

3. Run Jess' magical, save all, only in case of extreme emergencies, "you are
going to have to break this glass to get it" script.

```bash
docker build --rm --force-rm -t docker-dev:master .
docker run --rm -it --privileged \
    -v /path/to/your/repos/dir:/volumes/repos \
    -v $HOME/.gnupg:/root/.gnupg \
    -e GPG_PASSPHRASE \
    -e DOCKER_RELEASE_DIR=/volumes/repos \
    docker-dev:master hack/make.sh update-apt-repo generate-index-listing sign-repos
```

4. Upload the changed repo to `s3` (if that is where you host)

PURGE THE CACHE.
