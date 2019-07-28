# Changelog

## v1.4.7 / 2018-01-09

* BSD/macOS: Fix possible deadlock on closing the watcher on kqueue (thanks @nhooyr and @glycerine)
* Tests: Fix missing verb on format string (thanks @rchiossi)
* Linux: Fix deadlock in Remove (thanks @aarondl)
* Linux: Watch.Add improvements (avoid race, fix consistency, reduce garbage) (thanks @twpayne)
* Docs: Moved FAQ into the README (thanks @vahe)
* Linux: Properly handle inotify's IN_Q_OVERFLOW event (thanks @zeldovich)
* Docs: replace references to OS X with macOS

## v1.4.2 / 2016-10-10

* Linux: use InotifyInit1 with IN_CLOEXEC to stop leaking a file descriptor to a child process when using fork/exec [#178](https://github.com/fsnotify/fsnotify/pull/178) (thanks @pattyshack)

## v1.4.1 / 2016-10-04

* Fix flaky inotify stress test on Linux [#177](https://github.com/fsnotify/fsnotify/pull/177) (thanks @pattyshack)

## v1.4.0 / 2016-10-01

* add a String() method to Event.Op [#165](https://github.com/fsnotify/fsnotify/pull/165) (thanks @oozie)

## v1.3.1 / 2016-06-28

* Windows: fix for double backslash when watching the root of a drive [#151](https://github.com/fsnotify/fsnotify/issues/151) (thanks @brunoqc)

## v1.3.0 / 2016-04-19

* Support linux/arm64 by [patching](https://go-review.googlesource.com/#/c/21971/) x/sys/unix and switching to to it from syscall (thanks @suihkulokki) [#135](https://github.com/fsnotify/fsnotify/pull/135)

## v1.2.10 / 2016-03-02

* Fix golint errors in windows.go [#121](https://github.com/fsnotify/fsnotify/pull/121) (thanks @tiffanyfj)

## v1.2.9 / 2016-01-13

kqueue: Fix logic for CREATE after REMOVE [#111](https://github.com/fsnotify/fsnotify/pull/111) (thanks @bep)

## v1.2.8 / 2015-12-17

* kqueue: fix race condition in Close [#105](https://github.com/fsnotify/fsnotify/pull/105) (thanks @djui for reporting the issue and @ppknap for writing a failing test)
* inotify: fix race in test
* enable race detection for continuous integration (Linux, Mac, Windows)

## v1.2.5 / 2015-10-17

* inotify: use epoll_create1 for arm64 support (requires Linux 2.6.27 or later) [#100](https://github.com/fsnotify/fsnotify/pull/100) (thanks @suihkulokki)
* inotify: fix path leaks [#73](https://github.com/fsnotify/fsnotify/pull/73) (thanks @chamaken)
* kqueue: watch for rename events on subdirectories [#83](https://github.com/fsnotify/fsnotify/pull/83) (thanks @guotie)
* kqueue: avoid infinite loops from symlinks cycles [#101](https://github.com/fsnotify/fsnotify/pull/101) (thanks @illicitonion)

## v1.2.1 / 2015-10-14

* kqueue: don't watch named pipes [#98](https://github.com/fsnotify/fsnotify/pull/98) (thanks @evanphx)

## v1.2.0 / 2015-02-08

* inotify: use epoll to wake up readEvents [#66](https://github.com/fsnotify/fsnotify/pull/66) (thanks @PieterD)
* inotify: closing watcher should now always shut down goroutine [#63](https://github.com/fsnotify/fsnotify/pull/63) (thanks @PieterD)
* kqueue: close kqueue after removing watches, fixes [#59](https://github.com/fsnotify/fsnotify/issues/59)

## v1.1.1 / 2015-02-05

* inotify: Retry read on EINTR [#61](https://github.com/fsnotify/fsnotify/issues/61) (thanks @PieterD)

## v1.1.0 / 2014-12-12

* kqueue: rework internals [#43](https://github.com/fsnotify/fsnotify/pull/43)
    * add low-level functions
    * only need to store flags on directories
    * less mutexes [#13](https://github.com/fsnotify/fsnotify/issues/13)
    * done can be an unbuffered channel
    * remove calls to os.NewSyscallError
* More efficient string concatenation for Event.String() [#52](https://github.com/fsnotify/fsnotify/pull/52) (thanks @mdlayher)
* kqueue: fix regression in  rework causing subdirectories to be watched [#48](https://github.com/fsnotify/fsnotify/issues/48)
* kqueue: cleanup internal watch before sending remove event [#51](https://github.com/fsnotify/fsnotify/issues/51)

## v1.0.4 / 2014-09-07

* kqueue: add dragonfly to the build tags.
* Rename source code files, rearrange code so exported APIs are at the top.
* Add done channel to example code. [#37](https://github.com/fsnotify/fsnotify/pull/37) (thanks @chenyukang)

## v1.0.3 / 2014-08-19

* [Fix] Windows MOVED_TO now translates to Create like on BSD and Linux. [#36](https://github.com/fsnotify/fsnotify/issues/36)

## v1.0.2 / 2014-08-17

* [Fix] Missing create events on macOS. [#14](https://github.com/fsnotify/fsnotify/issues/14) (thanks @zhsso)
* [Fix] Make ./path and path equivalent. (thanks @zhsso)

## v1.0.0 / 2014-08-15

* [API] Remove AddWatch on Windows, use Add.
* Improve documentation for exported identifiers. [#30](https://github.com/fsnotify/fsnotify/issues/30)
* Minor updates based on feedback from golint.

## dev / 2014-07-09

* Moved to [github.com/fsnotify/fsnotify](https://github.com/fsnotify/fsnotify).
* Use os.NewSyscallError instead of returning errno (thanks @hariharan-uno)

## dev / 2014-07-04

* kqueue: fix incorrect mutex used in Close()
* Update example to demonstrate usage of Op.

## dev / 2014-06-28

* [API] Don't set the Write Op for attribute notifications [#4](https://github.com/fsnotify/fsnotify/issues/4)
* Fix for String() method on Event (thanks Alex Brainman)
* Don't build on Plan 9 or Solaris (thanks @4ad)

## dev / 2014-06-21

* Events channel of type Event rather than *Event.
* [internal] use syscall constants directly for inotify and kqueue.
* [internal] kqueue: rename events to kevents and fileEvent to event.

## dev / 2014-06-19

* Go 1.3+ required on Windows (uses syscall.ERROR_MORE_DATA internally).
* [internal] remove cookie from Event struct (unused).
* [internal] Event struct has the same definition across every OS.
* [internal] remove internal watch and removeWatch methods.

## dev / 2014-06-12

* [API] Renamed Watch() to Add() and RemoveWatch() to Remove().
* [API] Pluralized channel names: Events and Errors.
* [API] Renamed FileEvent struct to Event.
* [API] Op constants replace methods like IsCreate().

## dev / 2014-06-12

* Fix data race on kevent buffer (thanks @tilaks) [#98](https://github.com/howeyc/fsnotify/pull/98)

## dev / 2014-05-23

* [API] Remove current implementation of WatchFlags.
    * current implementation doesn't take advantage of OS for efficiency
    * provides little benefit over filtering events as they are received, but has  extra bookkeeping and mutexes
    * no tests for the current implementation
    * not fully implemented on Windows [#93](https://github.com/howeyc/fsnotify/issues/93#issuecomment-39285195)

## v0.9.3 / 2014-12-31

* kqueue: cleanup internal watch before sending remove event [#51](https://github.com/fsnotify/fsnotify/issues/51)

## v0.9.2 / 2014-08-17

* [Backport] Fix missing create events on macOS. [#14](https://github.com/fsnotify/fsnotify/issues/14) (thanks @zhsso)

## v0.9.1 / 2014-06-12

* Fix data race on kevent buffer (thanks @tilaks) [#98](https://github.com/howeyc/fsnotify/pull/98)

## v0.9.0 / 2014-01-17

* IsAttrib() for events that only concern a file's metadata [#79][] (thanks @abustany)
* [Fix] kqueue: fix deadlock [#77][] (thanks @cespare)
* [NOTICE] Development has moved to `code.google.com/p/go.exp/fsnotify` in preparation for inclusion in the Go standard library.

## v0.8.12 / 2013-11-13

* [API] Remove FD_SET and friends from Linux adapter

## v0.8.11 / 2013-11-02

* [Doc] Add Changelog [#72][] (thanks @nathany)
* [Doc] Spotlight and double modify events on macOS [#62][] (reported by @paulhammond)

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
