# Functional tests for govc

## Bats

Install [Bats](https://github.com/sstephenson/bats/)

## coreutils

Install gxargs, greadlink and gmktemp on Darwin

```
brew install coreutils
brew install findutils
```

## Download test images

Some tests depend on ttylinux images, these can be downloaded by running:

```
./images/update.sh
```

These images are uploaded to the esxbox as needed by tests and can be
removed with the following command:

```
./clean.sh
```

## GOVC_TEST_URL

The govc tests need an ESX instance to run against.  The default
`GOVC_TEST_URL` is that of the vagrant box in the *esxbox* directory:

```
(cd esxbox && vagrant up)
```

Any other ESX box can be used by exporting the following variable:

```
export GOVC_TEST_URL=user:pass@hostname
```

## vCenter Simulator

Some tests require vCenter and depend on the Vagrant box in the
*vcsim* directory.  These tests are skipped if the vcsim box is not
running.  To enable these tests:

```
(cd vcsim && vagrant up)
```

## Running tests

The test helper prepends ../govc to `PATH`.

The tests can be run from any directory, as *govc* is found related to
`PATH` and *images* are found relative to `$BATS_TEST_DIRNAME`.

The entire suite can be run with the following command:

```
bats .
```

Or individually, for example:

```
./cli.bats
```
