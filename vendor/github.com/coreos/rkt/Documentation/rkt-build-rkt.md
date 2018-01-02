# coreos.com/rkt/builder

This container contains all build-time dependencies in order to build rkt.
It currently can be built in: _Debian Sid_.

All commands assume you are running them in your local git checkout of rkt.

## Building rkt in rkt

Configure the path to your git checkout of `rkt` and the build output directory respectively:

```
export SRC_DIR=
export BUILDDIR=
mkdir -p $BUILDDIR
```

Start the container which will run the [rkt builder](https://github.com/coreos/rkt-builder), and compile rkt:
```
./scripts/build-rir.sh
```

You should see rkt building in your rkt container, and once it's finished, the output should be in `$BUILD_DIR` on your host.
