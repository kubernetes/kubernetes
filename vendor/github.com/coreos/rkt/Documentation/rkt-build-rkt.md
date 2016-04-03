# coreos.com/rkt/builder

This container contains all build-time dependencies in order to build rkt.
It currently can be built in one of two flavors: _Debian Sid_ or _Fedora 22_.

All commands assume you are running them in your local git checkout of rkt.

## Building coreos.com/rkt/builder using acbuild

Requirements:
- Go or `docker2aci`
- rkt

The file `scripts/acbuild-rkt-builder.sh` contains a simple bash script which will download a base docker image which is then converted into an ACI.
Once the ACI has been downloaded, the manifest is patched.
Next acbuild is run which creates an ACI that has all the required build dependencies for compiling `rkt`.

If you want to change the base OS you can set it by using one of the following sets of variables:

```
export IMG=fedora
export IMGVERSION=22
```

Or (default):

```
export IMG=debian
export IMGVERSION=sid
```

Running the build script:
```
./scripts/acbuild-rkt-builder.sh
```

Once that is finished there should be a `rkt-builder.aci` file in the current directory.

## Building rkt in rkt

Now that `rkt-builder.aci` has been built you have a container which will compile `rkt`.

Put it into the rkt CAS:
```
rkt fetch --insecure-options=image ./rkt-builder.aci
```

Configure the path to your git checkout of `rkt` and the build output directory respectively:

```
export SRC_DIR=
export BUILDDIR=
mkdir -p $BUILDDIR
```

Start the container which will compile rkt:
```
./scripts/build-rir.sh
```

You should see rkt building in your rkt container, and once it's finished, the output should be in `$BUILD_DIR` on your host.

# Building rkt in rkt one liners (sort of)

If you don't want to bother with acbuild and want a simple one liner that uses rkt to build rkt,  you can install all the dependencies and build rkt from source in one line using bash in a container.

Set `SRC_DIR` to the absolute path to your git checkout of `rkt`:

```
export SRC_DIR=
```

Now pick a base OS you want to use, and run the appropriate command.
The build output will be in `${SRC_DIR}/build-rkt-${RKT_VERSION}+git`.

## Debian Sid
```
rkt run \
    --volume src-dir,kind=host,source=$SRC_DIR \
    --mount volume=src-dir,target=/opt/rkt \
    --interactive \
    --insecure-options=image \
    docker://debian:sid \
    --exec /bin/bash \
    -- -c 'apt-get update && apt-get install -y --no-install-recommends ca-certificates gcc libc6-dev make automake wget git golang-go coreutils cpio squashfs-tools realpath autoconf file xz-utils patch bc locales libacl1-dev && update-ca-certificates && cd /opt/rkt && ./autogen.sh && ./configure && make'
```

## Fedora 22
```
rkt run \
    --volume src-dir,kind=host,source=$SRC_DIR \
    --mount volume=src-dir,target=/opt/rkt \
    --interactive \
    --insecure-options=image \
    docker://fedora:22 \
    --exec /bin/bash \
    -- -c 'dnf install -y make gcc glibc-devel glibc-static cpio squashfs-tools gpg autoconf make automake golang file git wget tar xz patch bc hostname findutils openssl libacl-devel && cd /opt/rkt && ./autogen.sh && ./configure && make'
```
