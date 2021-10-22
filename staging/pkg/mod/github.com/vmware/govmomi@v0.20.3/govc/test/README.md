# Functional tests for govc

The govc tests use [bats](https://github.com/sstephenson/bats/)

## Download test images

Some tests depend on ttylinux images, these can be downloaded by running:

```
./images/update.sh
```

These images are uploaded to the `$GOVC_TEST_URL` as needed by tests and can be
removed with the following command:

```
./clean.sh
```

## GOVC_TEST_URL

Some of the govc tests need an ESX instance to run against.  Any ESX box can be used by exporting the following variable:

```
export GOVC_TEST_URL=user:pass@hostname
```

## Running tests

Tests can be run using the top-level Makefile:

```
make govc-test
```

Or the following command:

```
bats .
```

Or individually, for example:

```
./cli.bats
```

Note that the test helper prepends `$GOPATH/bin` to `PATH` as the tests depend on both the *govc* and *vcsim* binaries.

## Platform specific notes

### Darwin (MacOSX)

Install gxargs, greadlink and gmktemp on Darwin

```
brew install coreutils
brew install findutils
```
