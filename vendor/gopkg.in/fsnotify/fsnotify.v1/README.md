# File system notifications for Go

[![GoDoc](https://godoc.org/gopkg.in/fsnotify.v1?status.svg)](https://godoc.org/gopkg.in/fsnotify.v1) [![Coverage](http://gocover.io/_badge/github.com/go-fsnotify/fsnotify)](http://gocover.io/github.com/go-fsnotify/fsnotify) 

Go 1.3+ required.

Cross platform: Windows, Linux, BSD and OS X.

|Adapter   |OS        |Status    |
|----------|----------|----------|
|inotify   |Linux, Android\*|Supported [![Build Status](https://travis-ci.org/go-fsnotify/fsnotify.svg?branch=master)](https://travis-ci.org/go-fsnotify/fsnotify)|
|kqueue    |BSD, OS X, iOS\*|Supported [![Build Status](https://travis-ci.org/go-fsnotify/fsnotify.svg?branch=master)](https://travis-ci.org/go-fsnotify/fsnotify)|
|ReadDirectoryChangesW|Windows|Supported [![Build status](https://ci.appveyor.com/api/projects/status/ivwjubaih4r0udeh/branch/master?svg=true)](https://ci.appveyor.com/project/NathanYoungman/fsnotify/branch/master)|
|FSEvents  |OS X          |[Planned](https://github.com/go-fsnotify/fsnotify/issues/11)|
|FEN       |Solaris 11    |[Planned](https://github.com/go-fsnotify/fsnotify/issues/12)|
|fanotify  |Linux 2.6.37+ | |
|USN Journals |Windows    |[Maybe](https://github.com/go-fsnotify/fsnotify/issues/53)|
|Polling   |*All*         |[Maybe](https://github.com/go-fsnotify/fsnotify/issues/9)|

\* Android and iOS are untested.

Please see [the documentation](https://godoc.org/gopkg.in/fsnotify.v1) for usage. Consult the [Wiki](https://github.com/go-fsnotify/fsnotify/wiki) for the FAQ and further information.

## API stability

Two major versions of fsnotify exist. 

**[fsnotify.v0](https://gopkg.in/fsnotify.v0)** is API-compatible with [howeyc/fsnotify](https://godoc.org/github.com/howeyc/fsnotify). Bugfixes *may* be backported, but I recommend upgrading to v1.

```go
import "gopkg.in/fsnotify.v0"
```

\* Refer to the package as fsnotify (without the .v0 suffix).

**[fsnotify.v1](https://gopkg.in/fsnotify.v1)** provides [a new API](https://godoc.org/gopkg.in/fsnotify.v1) based on [this design document](http://goo.gl/MrYxyA). You can import v1 with:

```go
import "gopkg.in/fsnotify.v1"
```

Further API changes are [planned](https://github.com/go-fsnotify/fsnotify/milestones), but a new major revision will be tagged, so you can depend on the v1 API.

**Master** may have unreleased changes. Use it to test the very latest code or when [contributing][], but don't expect it to remain API-compatible:

```go
import "github.com/go-fsnotify/fsnotify"
```

## Contributing

Please refer to [CONTRIBUTING][] before opening an issue or pull request.

## Example

See [example_test.go](https://github.com/go-fsnotify/fsnotify/blob/master/example_test.go).

[contributing]: https://github.com/go-fsnotify/fsnotify/blob/master/CONTRIBUTING.md

## Related Projects

* [notify](https://github.com/rjeczalik/notify)
* [fsevents](https://github.com/go-fsnotify/fsevents)

