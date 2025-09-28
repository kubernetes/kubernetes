# Go OLE

[![Build status](https://ci.appveyor.com/api/projects/status/qr0u2sf7q43us9fj?svg=true)](https://ci.appveyor.com/project/jacobsantos/go-ole-jgs28)
[![Build Status](https://travis-ci.org/go-ole/go-ole.svg?branch=master)](https://travis-ci.org/go-ole/go-ole)
[![GoDoc](https://godoc.org/github.com/go-ole/go-ole?status.svg)](https://godoc.org/github.com/go-ole/go-ole)

Go bindings for Windows COM using shared libraries instead of cgo.

By Yasuhiro Matsumoto.

## Install

To experiment with go-ole, you can just compile and run the example program:

```
go get github.com/go-ole/go-ole
cd /path/to/go-ole/
go test

cd /path/to/go-ole/example/excel
go run excel.go
```

## Continuous Integration

Continuous integration configuration has been added for both Travis-CI and AppVeyor. You will have to add these to your own account for your fork in order for it to run.

**Travis-CI**

Travis-CI was added to check builds on Linux to ensure that `go get` works when cross building. Currently, Travis-CI is not used to test cross-building, but this may be changed in the future. It is also not currently possible to test the library on Linux, since COM API is specific to Windows and it is not currently possible to run a COM server on Linux or even connect to a remote COM server.

**AppVeyor**

AppVeyor is used to build on Windows using the (in-development) test COM server. It is currently only used to test the build and ensure that the code works on Windows. It will be used to register a COM server and then run the test cases based on the test COM server.

The tests currently do run and do pass and this should be maintained with commits.

## Versioning

Go OLE uses [semantic versioning](http://semver.org) for version numbers, which is similar to the version contract of the Go language. Which means that the major version will always maintain backwards compatibility with minor versions. Minor versions will only add new additions and changes. Fixes will always be in patch. 

This contract should allow you to upgrade to new minor and patch versions without breakage or modifications to your existing code. Leave a ticket, if there is breakage, so that it could be fixed.

## LICENSE

Under the MIT License: http://mattn.mit-license.org/2013
