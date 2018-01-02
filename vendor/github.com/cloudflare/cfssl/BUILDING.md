# How to Build CFSSL

## Docker 

The requirements to build `CFSSL` are:

1. A running instance of Docker 
2. The `bash` shell

To build, run:

    $ script/build    

This is will build by default all the cfssl command line utilities
for darwin (OSX), linux, and windows for i386 and amd64 and output the
binaries in the current path.

To build a specific platform and OS, run:

    $ script/build -os="darwin" -arch="amd64"

Note: for cross-compilation compatibility, the Docker build process will
build programs without PKCS #11.

## Without Docker

The requirements to build without Docker are:

1. Go version 1.5 is the minimum required version of Go.
2. A properly configured go environment
3. A properly configured GOPATH
4. With Go 1.5, you are required to set the environment variable `GO15VENDOREXPERIMENT=1`

Run:

    $ go install github.com/cloudflare/cfssl/cmd/...
