## gocompat ##

This directory contains backports of stdlib functions from later Go versions so
the filepath-securejoin can continue to be used by projects that are stuck with
Go 1.18 support. Note that often filepath-securejoin is added in security
patches for old releases, so avoiding the need to bump Go compiler requirements
is a huge plus to downstreams.

The source code is licensed under the same license as the Go stdlib. See the
source files for the precise license information.
