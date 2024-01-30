# Changelog

Unreleased
----------
Nothing yet.

1.7.0 - 2023-10-22
------------------
This version of fsnotify needs Go 1.17.

### Additions

- illumos: add FEN backend to support illumos and Solaris. ([#371])

- all: add `NewBufferedWatcher()` to use a buffered channel, which can be useful
  in cases where you can't control the kernel buffer and receive a large number
  of events in bursts. ([#550], [#572])

- all: add `AddWith()`, which is identical to `Add()` but allows passing
  options. ([#521])

- windows: allow setting the ReadDirectoryChangesW() buffer size with
  `fsnotify.WithBufferSize()`; the default of 64K is the highest value that
  works on all platforms and is enough for most purposes, but in some cases a
  highest buffer is needed. ([#521])

### Changes and fixes

- inotify: remove watcher if a watched path is renamed ([#518])

  After a rename the reported name wasn't updated, or even an empty string.
  Inotify doesn't provide any good facilities to update it, so just remove the
  watcher. This is already how it worked on kqueue and FEN.

  On Windows this does work, and remains working.

- windows: don't listen for file attribute changes ([#520])

  File attribute changes are sent as `FILE_ACTION_MODIFIED` by the Windows API,
  with no way to see if they're a file write or attribute change, so would show
  up as a fsnotify.Write event. This is never useful, and could result in many
  spurious Write events.

- windows: return `ErrEventOverflow` if the buffer is full ([#525])

  Before it would merely return "short read", making it hard to detect this
  error.

- kqueue: make sure events for all files are delivered properly when removing a
  watched directory ([#526])

  Previously they would get sent with `""` (empty string) or `"."` as the path
  name.

- kqueue: don't emit spurious Create events for symbolic links ([#524])

  The link would get resolved but kqueue would "forget" it already saw the link
  itself, resulting on a Create for every Write event for the directory.

- all: return `ErrClosed` on `Add()` when the watcher is closed ([#516])

- other: add `Watcher.Errors` and `Watcher.Events` to the no-op `Watcher` in
  `backend_other.go`, making it easier to use on unsupported platforms such as
  WASM, AIX, etc. ([#528])

- other: use the `backend_other.go` no-op if the `appengine` build tag is set;
  Google AppEngine forbids usage of the unsafe package so the inotify backend
  won't compile there.

[#371]: https://github.com/fsnotify/fsnotify/pull/371
[#516]: https://github.com/fsnotify/fsnotify/pull/516
[#518]: https://github.com/fsnotify/fsnotify/pull/518
[#520]: https://github.com/fsnotify/fsnotify/pull/520
[#521]: https://github.com/fsnotify/fsnotify/pull/521
[#524]: https://github.com/fsnotify/fsnotify/pull/524
[#525]: https://github.com/fsnotify/fsnotify/pull/525
[#526]: https://github.com/fsnotify/fsnotify/pull/526
[#528]: https://github.com/fsnotify/fsnotify/pull/528
[#537]: https://github.com/fsnotify/fsnotify/pull/537
[#550]: https://github.com/fsnotify/fsnotify/pull/550
[#572]: https://github.com/fsnotify/fsnotify/pull/572

1.6.0 - 2022-10-13
------------------
This version of fsnotify needs Go 1.16 (this was already the case since 1.5.1,
but not documented). It also increases the minimum Linux version to 2.6.32.

### Additions

- all: add `Event.Has()` and `Op.Has()` ([#477])

  This makes checking events a lot easier; for example:

	    if event.Op&Write == Write && !(event.Op&Remove == Remove) {
	    }

	Becomes:

	    if event.Has(Write) && !event.Has(Remove) {
	    }

- all: add cmd/fsnotify ([#463])

  A command-line utility for testing and some examples.

### Changes and fixes

- inotify: don't ignore events for files that don't exist ([#260], [#470])

  Previously the inotify watcher would call `os.Lstat()` to check if a file
  still exists before emitting events.

  This was inconsistent with other platforms and resulted in inconsistent event
  reporting (e.g. when a file is quickly removed and re-created), and generally
  a source of confusion. It was added in 2013 to fix a memory leak that no
  longer exists.

- all: return `ErrNonExistentWatch` when `Remove()` is called on a path that's
  not watched ([#460])

- inotify: replace epoll() with non-blocking inotify ([#434])

  Non-blocking inotify was not generally available at the time this library was
  written in 2014, but now it is. As a result, the minimum Linux version is
  bumped from 2.6.27 to 2.6.32. This hugely simplifies the code and is faster.

- kqueue: don't check for events every 100ms ([#480])

  The watcher would wake up every 100ms, even when there was nothing to do. Now
  it waits until there is something to do.

- macos: retry opening files on EINTR ([#475])

- kqueue: skip unreadable files ([#479])

  kqueue requires a file descriptor for every file in a directory; this would
  fail if a file was unreadable by the current user. Now these files are simply
  skipped.

- windows: fix renaming a watched directory if the parent is also watched ([#370])

- windows: increase buffer size from 4K to 64K ([#485])

- windows: close file handle on Remove() ([#288])

- kqueue: put pathname in the error if watching a file fails ([#471])

- inotify, windows: calling Close() more than once could race ([#465])

- kqueue: improve Close() performance ([#233])

- all: various documentation additions and clarifications.

[#233]: https://github.com/fsnotify/fsnotify/pull/233
[#260]: https://github.com/fsnotify/fsnotify/pull/260
[#288]: https://github.com/fsnotify/fsnotify/pull/288
[#370]: https://github.com/fsnotify/fsnotify/pull/370
[#434]: https://github.com/fsnotify/fsnotify/pull/434
[#460]: https://github.com/fsnotify/fsnotify/pull/460
[#463]: https://github.com/fsnotify/fsnotify/pull/463
[#465]: https://github.com/fsnotify/fsnotify/pull/465
[#470]: https://github.com/fsnotify/fsnotify/pull/470
[#471]: https://github.com/fsnotify/fsnotify/pull/471
[#475]: https://github.com/fsnotify/fsnotify/pull/475
[#477]: https://github.com/fsnotify/fsnotify/pull/477
[#479]: https://github.com/fsnotify/fsnotify/pull/479
[#480]: https://github.com/fsnotify/fsnotify/pull/480
[#485]: https://github.com/fsnotify/fsnotify/pull/485

## [1.5.4] - 2022-04-25

* Windows: add missing defer to `Watcher.WatchList` [#447](https://github.com/fsnotify/fsnotify/pull/447)
* go.mod: use latest x/sys [#444](https://github.com/fsnotify/fsnotify/pull/444)
* Fix compilation for OpenBSD [#443](https://github.com/fsnotify/fsnotify/pull/443)

## [1.5.3] - 2022-04-22

* This version is retracted. An incorrect branch is published accidentally [#445](https://github.com/fsnotify/fsnotify/issues/445)

## [1.5.2] - 2022-04-21

* Add a feature to return the directories and files that are being monitored [#374](https://github.com/fsnotify/fsnotify/pull/374)
* Fix potential crash on windows if `raw.FileNameLength` exceeds `syscall.MAX_PATH` [#361](https://github.com/fsnotify/fsnotify/pull/361)
* Allow build on unsupported GOOS [#424](https://github.com/fsnotify/fsnotify/pull/424)
* Don't set `poller.fd` twice in `newFdPoller` [#406](https://github.com/fsnotify/fsnotify/pull/406)
* fix go vet warnings: call to `(*T).Fatalf` from a non-test goroutine [#416](https://github.com/fsnotify/fsnotify/pull/416)

## [1.5.1] - 2021-08-24

* Revert Add AddRaw to not follow symlinks [#394](https://github.com/fsnotify/fsnotify/pull/394)

## [1.5.0] - 2021-08-20

* Go: Increase minimum required version to Go 1.12 [#381](https://github.com/fsnotify/fsnotify/pull/381)
* Feature: Add AddRaw method which does not follow symlinks when adding a watch [#289](https://github.com/fsnotify/fsnotify/pull/298)
* Windows: Follow symlinks by default like on all other systems [#289](https://github.com/fsnotify/fsnotify/pull/289)
* CI: Use GitHub Actions for CI and cover go 1.12-1.17
   [#378](https://github.com/fsnotify/fsnotify/pull/378)
   [#381](https://github.com/fsnotify/fsnotify/pull/381)
   [#385](https://github.com/fsnotify/fsnotify/pull/385)
* Go 1.14+: Fix unsafe pointer conversion [#325](https://github.com/fsnotify/fsnotify/pull/325)

## [1.4.9] - 2020-03-11

* Move example usage to the readme #329. This may resolve #328.

## [1.4.8] - 2020-03-10

* CI: test more go versions (@nathany 1d13583d846ea9d66dcabbfefbfb9d8e6fb05216)
* Tests: Queued inotify events could have been read by the test before max_queued_events was hit (@matthias-stone #265)
* Tests:  t.Fatalf -> t.Errorf in go routines (@gdey #266)
* CI: Less verbosity (@nathany #267)
* Tests: Darwin: Exchangedata is deprecated on 10.13 (@nathany #267)
* Tests: Check if channels are closed in the example (@alexeykazakov #244)
* CI: Only run golint on latest version of go and fix issues (@cpuguy83 #284)
* CI: Add windows to travis matrix (@cpuguy83 #284)
* Docs: Remover appveyor badge (@nathany 11844c0959f6fff69ba325d097fce35bd85a8e93)
* Linux: create epoll and pipe fds with close-on-exec (@JohannesEbke #219)
* Linux: open files with close-on-exec (@linxiulei #273)
* Docs: Plan to support fanotify (@nathany ab058b44498e8b7566a799372a39d150d9ea0119 )
* Project: Add go.mod (@nathany #309)
* Project: Revise editor config (@nathany #309)
* Project: Update copyright for 2019 (@nathany #309)
* CI: Drop go1.8 from CI matrix (@nathany #309)
* Docs: Updating the FAQ section for supportability with NFS & FUSE filesystems (@Pratik32 4bf2d1fec78374803a39307bfb8d340688f4f28e )

## [1.4.7] - 2018-01-09

* BSD/macOS: Fix possible deadlock on closing the watcher on kqueue (thanks @nhooyr and @glycerine)
* Tests: Fix missing verb on format string (thanks @rchiossi)
* Linux: Fix deadlock in Remove (thanks @aarondl)
* Linux: Watch.Add improvements (avoid race, fix consistency, reduce garbage) (thanks @twpayne)
* Docs: Moved FAQ into the README (thanks @vahe)
* Linux: Properly handle inotify's IN_Q_OVERFLOW event (thanks @zeldovich)
* Docs: replace references to OS X with macOS

## [1.4.2] - 2016-10-10

* Linux: use InotifyInit1 with IN_CLOEXEC to stop leaking a file descriptor to a child process when using fork/exec [#178](https://github.com/fsnotify/fsnotify/pull/178) (thanks @pattyshack)

## [1.4.1] - 2016-10-04

* Fix flaky inotify stress test on Linux [#177](https://github.com/fsnotify/fsnotify/pull/177) (thanks @pattyshack)

## [1.4.0] - 2016-10-01

* add a String() method to Event.Op [#165](https://github.com/fsnotify/fsnotify/pull/165) (thanks @oozie)

## [1.3.1] - 2016-06-28

* Windows: fix for double backslash when watching the root of a drive [#151](https://github.com/fsnotify/fsnotify/issues/151) (thanks @brunoqc)

## [1.3.0] - 2016-04-19

* Support linux/arm64 by [patching](https://go-review.googlesource.com/#/c/21971/) x/sys/unix and switching to to it from syscall (thanks @suihkulokki) [#135](https://github.com/fsnotify/fsnotify/pull/135)

## [1.2.10] - 2016-03-02

* Fix golint errors in windows.go [#121](https://github.com/fsnotify/fsnotify/pull/121) (thanks @tiffanyfj)

## [1.2.9] - 2016-01-13

kqueue: Fix logic for CREATE after REMOVE [#111](https://github.com/fsnotify/fsnotify/pull/111) (thanks @bep)

## [1.2.8] - 2015-12-17

* kqueue: fix race condition in Close [#105](https://github.com/fsnotify/fsnotify/pull/105) (thanks @djui for reporting the issue and @ppknap for writing a failing test)
* inotify: fix race in test
* enable race detection for continuous integration (Linux, Mac, Windows)

## [1.2.5] - 2015-10-17

* inotify: use epoll_create1 for arm64 support (requires Linux 2.6.27 or later) [#100](https://github.com/fsnotify/fsnotify/pull/100) (thanks @suihkulokki)
* inotify: fix path leaks [#73](https://github.com/fsnotify/fsnotify/pull/73) (thanks @chamaken)
* kqueue: watch for rename events on subdirectories [#83](https://github.com/fsnotify/fsnotify/pull/83) (thanks @guotie)
* kqueue: avoid infinite loops from symlinks cycles [#101](https://github.com/fsnotify/fsnotify/pull/101) (thanks @illicitonion)

## [1.2.1] - 2015-10-14

* kqueue: don't watch named pipes [#98](https://github.com/fsnotify/fsnotify/pull/98) (thanks @evanphx)

## [1.2.0] - 2015-02-08

* inotify: use epoll to wake up readEvents [#66](https://github.com/fsnotify/fsnotify/pull/66) (thanks @PieterD)
* inotify: closing watcher should now always shut down goroutine [#63](https://github.com/fsnotify/fsnotify/pull/63) (thanks @PieterD)
* kqueue: close kqueue after removing watches, fixes [#59](https://github.com/fsnotify/fsnotify/issues/59)

## [1.1.1] - 2015-02-05

* inotify: Retry read on EINTR [#61](https://github.com/fsnotify/fsnotify/issues/61) (thanks @PieterD)

## [1.1.0] - 2014-12-12

* kqueue: rework internals [#43](https://github.com/fsnotify/fsnotify/pull/43)
    * add low-level functions
    * only need to store flags on directories
    * less mutexes [#13](https://github.com/fsnotify/fsnotify/issues/13)
    * done can be an unbuffered channel
    * remove calls to os.NewSyscallError
* More efficient string concatenation for Event.String() [#52](https://github.com/fsnotify/fsnotify/pull/52) (thanks @mdlayher)
* kqueue: fix regression in  rework causing subdirectories to be watched [#48](https://github.com/fsnotify/fsnotify/issues/48)
* kqueue: cleanup internal watch before sending remove event [#51](https://github.com/fsnotify/fsnotify/issues/51)

## [1.0.4] - 2014-09-07

* kqueue: add dragonfly to the build tags.
* Rename source code files, rearrange code so exported APIs are at the top.
* Add done channel to example code. [#37](https://github.com/fsnotify/fsnotify/pull/37) (thanks @chenyukang)

## [1.0.3] - 2014-08-19

* [Fix] Windows MOVED_TO now translates to Create like on BSD and Linux. [#36](https://github.com/fsnotify/fsnotify/issues/36)

## [1.0.2] - 2014-08-17

* [Fix] Missing create events on macOS. [#14](https://github.com/fsnotify/fsnotify/issues/14) (thanks @zhsso)
* [Fix] Make ./path and path equivalent. (thanks @zhsso)

## [1.0.0] - 2014-08-15

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

## [0.9.3] - 2014-12-31

* kqueue: cleanup internal watch before sending remove event [#51](https://github.com/fsnotify/fsnotify/issues/51)

## [0.9.2] - 2014-08-17

* [Backport] Fix missing create events on macOS. [#14](https://github.com/fsnotify/fsnotify/issues/14) (thanks @zhsso)

## [0.9.1] - 2014-06-12

* Fix data race on kevent buffer (thanks @tilaks) [#98](https://github.com/howeyc/fsnotify/pull/98)

## [0.9.0] - 2014-01-17

* IsAttrib() for events that only concern a file's metadata [#79][] (thanks @abustany)
* [Fix] kqueue: fix deadlock [#77][] (thanks @cespare)
* [NOTICE] Development has moved to `code.google.com/p/go.exp/fsnotify` in preparation for inclusion in the Go standard library.

## [0.8.12] - 2013-11-13

* [API] Remove FD_SET and friends from Linux adapter

## [0.8.11] - 2013-11-02

* [Doc] Add Changelog [#72][] (thanks @nathany)
* [Doc] Spotlight and double modify events on macOS [#62][] (reported by @paulhammond)

## [0.8.10] - 2013-10-19

* [Fix] kqueue: remove file watches when parent directory is removed [#71][] (reported by @mdwhatcott)
* [Fix] kqueue: race between Close and readEvents [#70][] (reported by @bernerdschaefer)
* [Doc] specify OS-specific limits in README (thanks @debrando)

## [0.8.9] - 2013-09-08

* [Doc] Contributing (thanks @nathany)
* [Doc] update package path in example code [#63][] (thanks @paulhammond)
* [Doc] GoCI badge in README (Linux only) [#60][]
* [Doc] Cross-platform testing with Vagrant  [#59][] (thanks @nathany)

## [0.8.8] - 2013-06-17

* [Fix] Windows: handle `ERROR_MORE_DATA` on Windows [#49][] (thanks @jbowtie)

## [0.8.7] - 2013-06-03

* [API] Make syscall flags internal
* [Fix] inotify: ignore event changes
* [Fix] race in symlink test [#45][] (reported by @srid)
* [Fix] tests on Windows
* lower case error messages

## [0.8.6] - 2013-05-23

* kqueue: Use EVT_ONLY flag on Darwin
* [Doc] Update README with full example

## [0.8.5] - 2013-05-09

* [Fix] inotify: allow monitoring of "broken" symlinks (thanks @tsg)

## [0.8.4] - 2013-04-07

* [Fix] kqueue: watch all file events [#40][] (thanks @ChrisBuchholz)

## [0.8.3] - 2013-03-13

* [Fix] inoitfy/kqueue memory leak [#36][] (reported by @nbkolchin)
* [Fix] kqueue: use fsnFlags for watching a directory [#33][] (reported by @nbkolchin)

## [0.8.2] - 2013-02-07

* [Doc] add Authors
* [Fix] fix data races for map access [#29][] (thanks @fsouza)

## [0.8.1] - 2013-01-09

* [Fix] Windows path separators
* [Doc] BSD License

## [0.8.0] - 2012-11-09

* kqueue: directory watching improvements (thanks @vmirage)
* inotify: add `IN_MOVED_TO` [#25][] (requested by @cpisto)
* [Fix] kqueue: deleting watched directory [#24][] (reported by @jakerr)

## [0.7.4] - 2012-10-09

* [Fix] inotify: fixes from https://codereview.appspot.com/5418045/ (ugorji)
* [Fix] kqueue: preserve watch flags when watching for delete [#21][] (reported by @robfig)
* [Fix] kqueue: watch the directory even if it isn't a new watch (thanks @robfig)
* [Fix] kqueue: modify after recreation of file

## [0.7.3] - 2012-09-27

* [Fix] kqueue: watch with an existing folder inside the watched folder (thanks @vmirage)
* [Fix] kqueue: no longer get duplicate CREATE events

## [0.7.2] - 2012-09-01

* kqueue: events for created directories

## [0.7.1] - 2012-07-14

* [Fix] for renaming files

## [0.7.0] - 2012-07-02

* [Feature] FSNotify flags
* [Fix] inotify: Added file name back to event path

## [0.6.0] - 2012-06-06

* kqueue: watch files after directory created (thanks @tmc)

## [0.5.1] - 2012-05-22

* [Fix] inotify: remove all watches before Close()

## [0.5.0] - 2012-05-03

* [API] kqueue: return errors during watch instead of sending over channel
* kqueue: match symlink behavior on Linux
* inotify: add `DELETE_SELF` (requested by @taralx)
* [Fix] kqueue: handle EINTR (reported by @robfig)
* [Doc] Godoc example [#1][] (thanks @davecheney)

## [0.4.0] - 2012-03-30

* Go 1 released: build with go tool
* [Feature] Windows support using winfsnotify
* Windows does not have attribute change notifications
* Roll attribute notifications into IsModify

## [0.3.0] - 2012-02-19

* kqueue: add files when watch directory

## [0.2.0] - 2011-12-30

* update to latest Go weekly code

## [0.1.0] - 2011-10-19

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
