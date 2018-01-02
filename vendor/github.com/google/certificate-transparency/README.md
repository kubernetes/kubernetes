certificate-transparency
========================

#Auditing for TLS certificates#

[![Build Status](https://travis-ci.org/google/certificate-transparency.svg?branch=master)](https://travis-ci.org/google/certificate-transparency)


## Build With GClient ##

This is now the recommended method for all supported platforms. It gives you 
a reproducible build and avoids the need to build some dependencies manually.

Known to work on FreeBSD 10, OS X (10.10) [tested with XCode + brew installation
of deps listed below], and Ubuntu 14.04. Tested on Fedora 22 but may require
manual override of compiler options as documented below. Tested on CentOS 7
with similar caveats.

### Install Dependencies ###

Depending on which platform you have the exact packages required will vary.
The following tools must be available for the GClient build to succeed:

 - autoconf/automake etc.
 - clang++ (>=3.4)
 - cmake (>=v3.1.2)
 - git
 - GNU make
 - libtool
 - shtool
 - Tcl
 - pkgconf
 - python27
 - [depot_tools](https://www.chromium.org/developers/how-tos/install-depot-tools)

### Building with gclient ###

```bash
export CXX=clang++ CC=clang
mkdir ct  # or whatever directory you prefer
cd ct
gclient config --name="certificate-transparency" https://github.com/google/certificate-transparency.git
gclient sync
# substitute gmake or gnumake below if that's what your platform calls it:
make -C certificate-transparency check
```

If you're trying to clone from a branch on the CT repo then you'll need to
substitute the following command for the `gclient config` command above,
replacing `branch` as appropriate

```bash
gclient config --name="certificate-transparency" https://github.com/google/certificate-transparency.git@branch
```

### Platform Specific Notes ###

#### Fedora / CentOS ####

When you issue the `gclient sync` command you may need to set compiler options
in order to build successfully. If the build fails to work try using:

```bash
CXXFLAGS="-O2 -Wno-error=unused-variable" gclient sync
```

If this gives an error about an unused typedef in a `glog` header file try this:

```bash
CXXFLAGS="-O2 -Wno-error=unused-variable -Wno-error=unused-local-typedefs" gclient sync
```

When changing `CXXFLAGS` it's safer to remove the existing build directories
in case not all dependencies are properly accounted for and rebuilt. If 
problems persist check that the Makefile in `certificate-transparency` 
contains the options that were passed in `CXXFLAGS`.

If there are still problems using GClient then an older style build can be
attempted. The process should be similar to the one documented for Ubuntu
below or in the [Fedora README](README.Fedora) depending on platform.

## Deprecated: Quickstart on Ubuntu ##

This should no longer be needed as the instructions above should work. But in
case of difficulties the dependencies can be built manually. The following 
steps will checkout the code and build it on a clean Ubuntu 14.04 LTS 
installation.  It has also been tested on an Ubuntu 15.04 installation.

First, install packaged dependencies:

    sudo apt-get update -qq
    sudo apt-get install -qq unzip cmake g++ libevent-dev golang-go autoconf pkg-config \
        libjson-c-dev libgflags-dev libgoogle-glog-dev libprotobuf-dev libleveldb-dev \
        libssl-dev libgoogle-perftools-dev protobuf-compiler libsqlite3-dev ant openjdk-7-jdk \
        libprotobuf-java python-gflags python-protobuf python-ecdsa python-mock \
        python-httplib2 git libldns-dev

Next, we need `libevhtp` version `1.2.10` which is not packaged in Ubuntu yet, so we build from source:

    wget https://github.com/ellzey/libevhtp/archive/1.2.10.zip
    unzip 1.2.10.zip
    cd libevhtp-1.2.10/
    cmake -DEVHTP_DISABLE_REGEX:STRING=ON -DCMAKE_C_FLAGS:STRING=-fPIC .
    make
    cd ..

And let's get our own Google Test / Google Mock as these vary in incompatible ways between packaged releases:

    wget https://googlemock.googlecode.com/files/gmock-1.7.0.zip
    unzip gmock-1.7.0.zip

Now, clone the CT repo:

    git clone https://github.com/google/certificate-transparency.git
    cd certificate-transparency/

One-time setup for Go:

    export GOPATH=$PWD/go
    mkdir -p $GOPATH/src/github.com/google
    ln -s $PWD $GOPATH/src/github.com/google
    go get -v -d ./...

Build CT server C++ code:

    ./autogen.sh
    ./configure GTEST_DIR=../gmock-1.7.0/gtest GMOCK_DIR=../gmock-1.7.0 \
        CPPFLAGS="-I../libevhtp-1.2.10 -I../libevhtp-1.2.10/evthr \
        -I../libevhtp-1.2.10/htparse" LDFLAGS=-L../libevhtp-1.2.10
    make check

Build and test Java code:

    ant build test

Build and test Python code:

    make -C python test

Best and test Go code:

    go test -v ./go/...


## Deprecated: Older Build Method ##

 - [OpenSSL](https://www.openssl.org/source/), at least 1.0.0q,
   preferably 1.0.1l or 1.0.2 (and up)

The checking of SCTs included in the
[RFC 6962](http://tools.ietf.org/html/rfc6962) TLS extension is only
included in OpenSSL 1.0.2. As of this writing, this version is not yet
released, so this means hand building the `OpenSSL_1_0_2-stable`
branch from the
[OpenSSL git repository](https://www.openssl.org/source/repos.html).

 - [googlemock](https://code.google.com/p/googlemock/) (tested with 1.7.0)

Gmock provides a bundled version of gtest, which will also be used.

Unpack googlemock, but do not build it. Upstream recommends to build a
new copy from source for each package to be tested. We follow this
advice in our `Makefile`, which builds gmock/gtest automatically.

Some systems make the googlemock source available as a package; on
Debian, this is in the google-mock package, which puts it in
`/usr/src/gmock`. Our `Makefile` looks in that location by default,
but if your googlemock sources are in a different location, set the
`GMOCK_DIR` environment variable to point at them.

If you are on FreeBSD, you may need to apply the patch in gtest.patch
to the gtest subdirectory of gmock.

 - [protobuf](https://github.com/google/protobuf) (tested with 2.5.0)
 - [gflags](https://github.com/gflags/gflags) (tested with 1.6
   and 2.0)
 - [glog](https://github.com/google/glog) (tested with 0.3.1)

Make sure to install glog **after** gflags, to avoid linking errors.

 - [sqlite3](http://www.sqlite.org/)
 - [leveldb](https://github.com/google/leveldb)
 - [JSON-C](https://github.com/json-c/json-c/), at least 0.11

You can specify a JSON-C library in a non-standard location using the
`JSONCLIBDIR` environment variable. Version 0.10 would work as well,
except the `json_object_iterator.h` header is not properly copied when
installing. If you can install the missing header manually, it should
work.

 - [libevent](http://libevent.org/) (tested with 2.0.21-stable)
 - [libevhtp](https://github.com/ellzey/libevhtp) (tested with 1.2.10)
 If building libevhtp from source, you may need to disable the regex support
 with the following cmake flag: `-DEVHTP_DISABLE_REGEX:STRING=ON`

You can specify a non-installed locally built library using the
`LIBEVENTDIR` environment variable to point to the local build. Note
that the FreeBSD port version 2.0.21_2 does not appear to work
correctly (it only listens on IPv6 for the HTTP server) - for that
platform we had to build from the source, specifically commit
6dba1694c89119c44cef03528945e5a5978ab43a.

 - [ldns](http://www.nlnetlabs.nl/projects/ldns/)
 - [ant](http://ant.apache.org/)
 - Python libraries:
  - pyasn1 and pyasn1-modules (optional, needed for `upload_server_cert.sh`)
  - [dnspython](http://www.dnspython.org/)

### Building ###

You can build the log server with the following commands:

    $ ./autogen.sh  # only necessary if you're building from git
    $ ./configure
    $ make

You can give the `configure` script extra parameters, to set
compilation flags, or point to custom versions of some dependencies
(notably, googlemock often needs this). For example, to compile with
Clang, using googlemock in `$HOME/gmock`, and a custom libevent in
`$HOME/libevent`:

    $ ./configure CXX=clang++ GMOCK_DIR=$HOME CPPFLAGS="-I$HOME/libevent/include" LDFLAGS="-L$HOME/libevent/.libs"

Running `./configure --help` provides more information about various
variables that can be set.

## Running Unit Tests ##

Run unit tests with this command

    $ make check

If the build still fails because of missing libraries, you may need to
set the environment variable `LD_LIBRARY_PATH`. On Linux, if you did
not change the default installation path (such as `/usr/local/lib`),
running

    $ ldconfig

or, if needed,

    $ sudo ldconfig

should resolve the problem.

## End-To-End Tests ##

For end-to-end server-client tests, you will need to install Apache
and point the tests to it. See `test/README` for how to do so.

## Testing and Logging Options ##

Note that several tests write files on disk. The default directory for
storing temporary testdata is `/tmp`. You can change this by setting
`TMPDIR=<tmpdir>` for make.

End-to-end tests also create temporary certificate and server files in
`test/tmp`. All these files are cleaned up after a successful test
run.

For logging options, see
http://google-glog.googlecode.com/svn/trunk/doc/glog.html

By default, unit tests log to stderr, and log only messages with a FATAL level
(i.e., those that result in abnormal program termination).
You can override the defaults with command-line flags.

End-to-end tests log everything at INFO level and above.
