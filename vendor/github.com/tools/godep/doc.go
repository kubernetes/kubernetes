/*

Command godep helps build packages reproducibly by fixing
their dependencies.

Example Usage

Save currently-used dependencies to file Godeps:

	$ godep save

Build project using saved dependencies:

	$ godep go install

or

	$ GOPATH=`godep path`:$GOPATH
	$ go install

*/
package main
