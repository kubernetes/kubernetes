# Changelog

## v0.14.2 / 2014-07-09

* Moved to [github.com/go-fsnotify/fsnotify](https://github.com/go-fsnotify/fsnotify).
* Use os.NewSyscallError instead of returning errno (thanks @hariharan-uno)
 
## v0.14.1 / 2014-07-04

* kqueue: fix incorrect mutex used in Close()
* Update example to demonstrate usage of Op.

## v0.14.0 / 2014-06-28

* [API] Don't set the Write Op for attribute notifications [#4](https://github.com/go-fsnotify/fsnotify/issues/4)
* Fix for String() method on Event (thanks Alex Brainman)
* Don't build on Plan 9 or Solaris (thanks @4ad)

## v0.13.0 / 2014-06-21

* Events channel of type Event rather than *Event.
* Add/Remove accept a name rather than a path (same behavior).
* [API] Remove AddWatch on Windows, use Add.
* [internal] use syscall constants directly for inotify and kqueue.
* [internal] kqueue: rename events to kevents and fileEvent to event.

## v0.12.0 / 2014-06-19

* Go 1.3+ required on Windows (uses syscall.ERROR_MORE_DATA internally).
* [internal] remove cookie from Event struct (unused).
* [internal] Event struct has the same definition across every OS.
* [internal] remove internal watch and removeWatch methods.

## v0.11.0 / 2014-06-12

* [API] Renamed Watch() to Add() and RemoveWatch() to Remove().
* [API] Pluralized channel names: Events and Errors.
* [API] Renamed FileEvent struct to Event.
* [API] Op constants replace methods like IsCreate().

## v0.10.1 / 2014-06-12

* Fix data race on kevent buffer (thanks @tilaks) [#98](https://github.com/howeyc/fsnotify/pull/98)

## v0.10.0 / 2014-05-23

* [API] Remove current implementation of WatchFlags.
    * current implementation doesn't take advantage of OS for efficiency
    * provides little benefit over filtering events as they are received, but has  extra bookkeeping and mutexes
    * no tests for the current implementation
    * not fully implemented on Windows [#93](https://github.com/howeyc/fsnotify/issues/93#issuecomment-39285195)

## v0.9.0 / 2014-01-17

* IsAttrib() for events that only concern a file's metadata [#79][] (thanks @abustany)
* [Fix] kqueue: fix deadlock [#77][] (thanks @cespare)
* [NOTICE] Development has moved to `code.google.com/p/go.exp/fsnotify` in preparation for inclusion in the Go standard library.

## v0.8.12 / 2013-11-13

* [API] Remove FD_SET and friends from Linux adapter

## v0.8.11 / 2013-11-02

* [Doc] Add Changelog [#72][] (thanks @nathany)
* [Doc] Spotlight and double modify events on OS X [#62][] (reported by @paulhammond)

## v0.8.10 / 2013-10-19

* [Fix] kqueue: remove file watches when parent directory is removed [#71][] (reported by @mdwhatcott)
* [Fix] kqueue: race between Close and readEvents [#70][] (reported by @bernerdschaefer)
* [Doc] specify OS-specific limits in README (thanks @debrando)

## v0.8.9 / 2013-09-08

* [Doc] Contributing (thanks @nathany)
* [Doc] update package path in example code [#63][] (thanks @paulhammond)
* [Doc] GoCI badge in README (Linux only) [#60][]
* [Doc] Cross-platform testing with Vagrant  [#59][] (thanks @nathany)

## v0.8.8 / 2013-06-17

* [Fix] Windows: handle `ERROR_MORE_DATA` on Windows [#49][] (thanks @jbowtie)

## v0.8.7 / 2013-06-03

* [API] Make syscall flags internal
* [Fix] inotify: ignore event changes
* [Fix] race in symlink test [#45][] (reported by @srid)
* [Fix] tests on Windows
* lower case error messages

## v0.8.6 / 2013-05-23

* kqueue: Use EVT_ONLY flag on Darwin
* [Doc] Update README with full example

## v0.8.5 / 2013-05-09

* [Fix] inotify: allow monitoring of "broken" symlinks (thanks @tsg)

## v0.8.4 / 2013-04-07

* [Fix] kqueue: watch all file events [#40][] (thanks @ChrisBuchholz)

## v0.8.3 / 2013-03-13

* [Fix] inoitfy/kqueue memory leak [#36][] (reported by @nbkolchin)
* [Fix] kqueue: use fsnFlags for watching a directory [#33][] (reported by @nbkolchin)

## v0.8.2 / 2013-02-07

* [Doc] add Authors
* [Fix] fix data races for map access [#29][] (thanks @fsouza)

## v0.8.1 / 2013-01-09

* [Fix] Windows path separators
* [Doc] BSD License

## v0.8.0 / 2012-11-09

* kqueue: directory watching improvements (thanks @vmirage)
* inotify: add `IN_MOVED_TO` [#25][] (requested by @cpisto)
* [Fix] kqueue: deleting watched directory [#24][] (reported by @jakerr)

## v0.7.4 / 2012-10-09

* [Fix] inotify: fixes from https://codereview.appspot.com/5418045/ (ugorji)
* [Fix] kqueue: preserve watch flags when watching for delete [#21][] (reported by @robfig)
* [Fix] kqueue: watch the directory even if it isn't a new watch (thanks @robfig)
* [Fix] kqueue: modify after recreation of file

## v0.7.3 / 2012-09-27

* [Fix] kqueue: watch with an existing folder inside the watched folder (thanks @vmirage)
* [Fix] kqueue: no longer get duplicate CREATE events

## v0.7.2 / 2012-09-01

* kqueue: events for created directories

## v0.7.1 / 2012-07-14

* [Fix] for renaming files

## v0.7.0 / 2012-07-02

* [Feature] FSNotify flags
* [Fix] inotify: Added file name back to event path

## v0.6.0 / 2012-06-06

* kqueue: watch files after directory created (thanks @tmc)

## v0.5.1 / 2012-05-22

* [Fix] inotify: remove all watches before Close()

## v0.5.0 / 2012-05-03

* [API] kqueue: return errors during watch instead of sending over channel
* kqueue: match symlink behavior on Linux
* inotify: add `DELETE_SELF` (requested by @taralx)
* [Fix] kqueue: handle EINTR (reported by @robfig)
* [Doc] Godoc example [#1][] (thanks @davecheney)

## v0.4.0 / 2012-03-30

* Go 1 released: build with go tool
* [Feature] Windows support using winfsnotify
* Windows does not have attribute change notifications
* Roll attribute notifications into IsModify

## v0.3.0 / 2012-02-19

* kqueue: add files when watch directory

## v0.2.0 / 2011-12-30

* update to latest Go weekly code

## v0.1.0 / 2011-10-19

* kqueue: add watch on file creation to match inotify
* kqueue: create file event
* inotify: ignore `IN_IGNORED` events
* event String()
* linux: common FileEvent functions
* initial commit

[#79]: https://github.com/howeyc/fsnotify/pull/79
[#77]: https://github.com/howeyc/fsnotify/pull/77
[#72]: https://github.com/howeyc/fsnotify/issues/72
[#71]: https://github.com/howeyc/fsnotify/issues/71
[#70]: https://github.com/howeyc/fsnotify/issues/70
[#63]: https://github.com/howeyc/fsnotify/issues/63
[#62]: https://github.com/howeyc/fsnotify/issues/62
[#60]: https://github.com/howeyc/fsnotify/issues/60
[#59]: https://github.com/howeyc/fsnotify/issues/59
[#49]: https://github.com/howeyc/fsnotify/issues/49
[#45]: https://github.com/howeyc/fsnotify/issues/45
[#40]: https://github.com/howeyc/fsnotify/issues/40
[#36]: https://github.com/howeyc/fsnotify/issues/36
[#33]: https://github.com/howeyc/fsnotify/issues/33
[#29]: https://github.com/howeyc/fsnotify/issues/29
[#25]: https://github.com/howeyc/fsnotify/issues/25
[#24]: https://github.com/howeyc/fsnotify/issues/24
[#21]: https://github.com/howeyc/fsnotify/issues/21

