# File system notifications for Go

[![GoDoc](https://godoc.org/github.com/fsnotify/fsnotify?status.svg)](https://godoc.org/github.com/fsnotify/fsnotify) [![Go Report Card](https://goreportcard.com/badge/github.com/fsnotify/fsnotify)](https://goreportcard.com/report/github.com/fsnotify/fsnotify)

fsnotify utilizes [golang.org/x/sys](https://godoc.org/golang.org/x/sys) rather than `syscall` from the standard library. Ensure you have the latest version installed by running:

```console
go get -u golang.org/x/sys/...
```

Cross platform: Windows, Linux, BSD and macOS.

|Adapter   |OS        |Status    |
|----------|----------|----------|
|inotify   |Linux 2.6.27 or later, Android\*|Supported [![Build Status](https://travis-ci.org/fsnotify/fsnotify.svg?branch=master)](https://travis-ci.org/fsnotify/fsnotify)|
|kqueue    |BSD, macOS, iOS\*|Supported [![Build Status](https://travis-ci.org/fsnotify/fsnotify.svg?branch=master)](https://travis-ci.org/fsnotify/fsnotify)|
|ReadDirectoryChangesW|Windows|Supported [![Build status](https://ci.appveyor.com/api/projects/status/ivwjubaih4r0udeh/branch/master?svg=true)](https://ci.appveyor.com/project/NathanYoungman/fsnotify/branch/master)|
|FSEvents  |macOS         |[Planned](https://github.com/fsnotify/fsnotify/issues/11)|
|FEN       |Solaris 11    |[In Progress](https://github.com/fsnotify/fsnotify/issues/12)|
|fanotify  |Linux 2.6.37+ | |
|USN Journals |Windows    |[Maybe](https://github.com/fsnotify/fsnotify/issues/53)|
|Polling   |*All*         |[Maybe](https://github.com/fsnotify/fsnotify/issues/9)|

\* Android and iOS are untested.

Please see [the documentation](https://godoc.org/github.com/fsnotify/fsnotify) and consult the [FAQ](#faq) for usage information.

## API stability

fsnotify is a fork of [howeyc/fsnotify](https://godoc.org/github.com/howeyc/fsnotify) with a new API as of v1.0. The API is based on [this design document](http://goo.gl/MrYxyA). 

All [releases](https://github.com/fsnotify/fsnotify/releases) are tagged based on [Semantic Versioning](http://semver.org/). Further API changes are [planned](https://github.com/fsnotify/fsnotify/milestones), and will be tagged with a new major revision number.

Go 1.6 supports dependencies located in the `vendor/` folder. Unless you are creating a library, it is recommended that you copy fsnotify into `vendor/github.com/fsnotify/fsnotify` within your project, and likewise for `golang.org/x/sys`.

## Contributing

Please refer to [CONTRIBUTING][] before opening an issue or pull request.

## Example

See [example_test.go](https://github.com/fsnotify/fsnotify/blob/master/example_test.go).

## FAQ

**When a file is moved to another directory is it still being watched?**

No (it shouldn't be, unless you are watching where it was moved to).

**When I watch a directory, are all subdirectories watched as well?**

No, you must add watches for any directory you want to watch (a recursive watcher is on the roadmap [#18][]).

**Do I have to watch the Error and Event channels in a separate goroutine?**

As of now, yes. Looking into making this single-thread friendly (see [howeyc #7][#7])

**Why am I receiving multiple events for the same file on OS X?**

Spotlight indexing on OS X can result in multiple events (see [howeyc #62][#62]). A temporary workaround is to add your folder(s) to the *Spotlight Privacy settings* until we have a native FSEvents implementation (see [#11][]).

**How many files can be watched at once?**

There are OS-specific limits as to how many watches can be created:
* Linux: /proc/sys/fs/inotify/max_user_watches contains the limit, reaching this limit results in a "no space left on device" error.
* BSD / OSX: sysctl variables "kern.maxfiles" and "kern.maxfilesperproc", reaching these limits results in a "too many open files" error.

[#62]: https://github.com/howeyc/fsnotify/issues/62
[#18]: https://github.com/fsnotify/fsnotify/issues/18
[#11]: https://github.com/fsnotify/fsnotify/issues/11
[#7]: https://github.com/howeyc/fsnotify/issues/7

[contributing]: https://github.com/fsnotify/fsnotify/blob/master/CONTRIBUTING.md

## Related Projects

* [notify](https://github.com/rjeczalik/notify)
* [fsevents](https://github.com/fsnotify/fsevents)

