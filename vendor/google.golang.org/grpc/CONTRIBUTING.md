# How to contribute

We definitely welcome patches and contribution to grpc! Here are some guidelines
and information about how to do so.

## Sending patches

### Getting started

1. Check out the code:

        $ go get google.golang.org/grpc
        $ cd $GOPATH/src/google.golang.org/grpc

1. Create a fork of the grpc-go repository.
1. Add your fork as a remote:

        $ git remote add fork git@github.com:$YOURGITHUBUSERNAME/grpc-go.git

1. Make changes, commit them.
1. Run the test suite:

        $ make test

1. Push your changes to your fork:

        $ git push fork ...

1. Open a pull request.

## Legal requirements

In order to protect both you and ourselves, you will need to sign the
[Contributor License Agreement](https://cla.developers.google.com/clas).

## Filing Issues
When filing an issue, make sure to answer these five questions:

1. What version of Go are you using (`go version`)?
2. What operating system and processor architecture are you using?
3. What did you do?
4. What did you expect to see?
5. What did you see instead?

### Contributing code
Unless otherwise noted, the Go source files are distributed under the BSD-style license found in the LICENSE file.
