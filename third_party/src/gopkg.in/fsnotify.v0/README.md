# File system notifications for Go

[![Coverage](http://gocover.io/_badge/github.com/go-fsnotify/fsnotify)](http://gocover.io/github.com/go-fsnotify/fsnotify) [![GoDoc](https://godoc.org/github.com/go-fsnotify/fsnotify?status.svg)](https://godoc.org/github.com/go-fsnotify/fsnotify)

Cross platform: Windows, Linux, BSD and OS X.

|Adapter   |OS        |Status    |
|----------|----------|----------|
|inotify   |Linux, Android|Supported|
|kqueue    |BSD, OS X, iOS|Supported|
|ReadDirectoryChangesW|Windows|Supported|
|FSEvents  |OS X          |[Planned](https://github.com/go-fsnotify/fsnotify/issues/11)|
|FEN       |Solaris 11    |[Planned](https://github.com/go-fsnotify/fsnotify/issues/12)|
|fanotify  |Linux 2.6.37+ | |
|[Polling](https://github.com/go-fsnotify/fsnotify/issues/9)|*All*         | |
|          |Plan 9        | |


Please see [the documentation](http://godoc.org/github.com/go-fsnotify/fsnotify) for usage. The [Wiki](https://github.com/go-fsnotify/fsnotify/wiki) contains an FAQ and further information.

## API stability

The fsnotify API has changed from what exists at `github.com/howeyc/fsnotify` ([GoDoc](http://godoc.org/github.com/howeyc/fsnotify)).

Further changes are expected. You may use [gopkg.in](https://gopkg.in/fsnotify.v0) to lock to the current API: 

```go
import "gopkg.in/fsnotify.v0"
```

A new major revision will be tagged for any future API changes.

## Contributing

* Send questions to [golang-dev@googlegroups.com](mailto:golang-dev@googlegroups.com). 
* Request features and report bugs using the [GitHub Issue Tracker](https://github.com/go-fsnotify/fsnotify/issues).

A future version of Go will have [fsnotify in the standard library](https://code.google.com/p/go/issues/detail?id=4068), therefore fsnotify carries the same [LICENSE](https://github.com/go-fsnotify/fsnotify/blob/master/LICENSE) as Go. Contributors retain their copyright, so we need you to fill out a short form before we can accept your contribution: [Google Individual Contributor License Agreement](https://developers.google.com/open-source/cla/individual).

Please read [CONTRIBUTING](https://github.com/go-fsnotify/fsnotify/blob/master/CONTRIBUTING.md) before opening a pull request.

## Example

See [example_test.go](https://github.com/go-fsnotify/fsnotify/blob/master/example_test.go).
