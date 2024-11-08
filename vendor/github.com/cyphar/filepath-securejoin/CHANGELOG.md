# Changelog #
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/)
and this project adheres to [Semantic Versioning](http://semver.org/).

## [Unreleased] ##

## [0.3.4] - 2024-10-09 ##

### Fixed ###
- Previously, some testing mocks we had resulted in us doing `import "testing"`
  in non-`_test.go` code, which made some downstreams like Kubernetes unhappy.
  This has been fixed. (#32)

## [0.3.3] - 2024-09-30 ##

### Fixed ###
- The mode and owner verification logic in `MkdirAll` has been removed. This
  was originally intended to protect against some theoretical attacks but upon
  further consideration these protections don't actually buy us anything and
  they were causing spurious errors with more complicated filesystem setups.
- The "is the created directory empty" logic in `MkdirAll` has also been
  removed. This was not causing us issues yet, but some pseudofilesystems (such
  as `cgroup`) create non-empty directories and so this logic would've been
  wrong for such cases.

## [0.3.2] - 2024-09-13 ##

### Changed ###
- Passing the `S_ISUID` or `S_ISGID` modes to `MkdirAllInRoot` will now return
  an explicit error saying that those bits are ignored by `mkdirat(2)`. In the
  past a different error was returned, but since the silent ignoring behaviour
  is codified in the man pages a more explicit error seems apt. While silently
  ignoring these bits would be the most compatible option, it could lead to
  users thinking their code sets these bits when it doesn't. Programs that need
  to deal with compatibility can mask the bits themselves. (#23, #25)

### Fixed ###
- If a directory has `S_ISGID` set, then all child directories will have
  `S_ISGID` set when created and a different gid will be used for any inode
  created under the directory. Previously, the "expected owner and mode"
  validation in `securejoin.MkdirAll` did not correctly handle this. We now
  correctly handle this case. (#24, #25)

## [0.3.1] - 2024-07-23 ##

### Changed ###
- By allowing `Open(at)InRoot` to opt-out of the extra work done by `MkdirAll`
  to do the necessary "partial lookups", `Open(at)InRoot` now does less work
  for both implementations (resulting in a many-fold decrease in the number of
  operations for `openat2`, and a modest improvement for non-`openat2`) and is
  far more guaranteed to match the correct `openat2(RESOLVE_IN_ROOT)`
  behaviour.
- We now use `readlinkat(fd, "")` where possible. For `Open(at)InRoot` this
  effectively just means that we no longer risk getting spurious errors during
  rename races. However, for our hardened procfs handler, this in theory should
  prevent mount attacks from tricking us when doing magic-link readlinks (even
  when using the unsafe host `/proc` handle). Unfortunately `Reopen` is still
  potentially vulnerable to those kinds of somewhat-esoteric attacks.

  Technically this [will only work on post-2.6.39 kernels][linux-readlinkat-emptypath]
  but it seems incredibly unlikely anyone is using `filepath-securejoin` on a
  pre-2011 kernel.

### Fixed ###
- Several improvements were made to the errors returned by `Open(at)InRoot` and
  `MkdirAll` when dealing with invalid paths under the emulated (ie.
  non-`openat2`) implementation. Previously, some paths would return the wrong
  error (`ENOENT` when the last component was a non-directory), and other paths
  would be returned as though they were acceptable (trailing-slash components
  after a non-directory would be ignored by `Open(at)InRoot`).

  These changes were done to match `openat2`'s behaviour and purely is a
  consistency fix (most users are going to be using `openat2` anyway).

[linux-readlinkat-emptypath]: https://git.kernel.org/pub/scm/linux/kernel/git/torvalds/linux.git/commit/?id=65cfc6722361570bfe255698d9cd4dccaf47570d

## [0.3.0] - 2024-07-11 ##

### Added ###
- A new set of `*os.File`-based APIs have been added. These are adapted from
  [libpathrs][] and we strongly suggest using them if possible (as they provide
  far more protection against attacks than `SecureJoin`):

   - `Open(at)InRoot` resolves a path inside a rootfs and returns an `*os.File`
     handle to the path. Note that the handle returned is an `O_PATH` handle,
     which cannot be used for reading or writing (as well as some other
     operations -- [see open(2) for more details][open.2])

   - `Reopen` takes an `O_PATH` file handle and safely re-opens it to upgrade
     it to a regular handle. This can also be used with non-`O_PATH` handles,
     but `O_PATH` is the most obvious application.

   - `MkdirAll` is an implementation of `os.MkdirAll` that is safe to use to
     create a directory tree within a rootfs.

  As these are new APIs, they may change in the future. However, they should be
  safe to start migrating to as we have extensive tests ensuring they behave
  correctly and are safe against various races and other attacks.

[libpathrs]: https://github.com/openSUSE/libpathrs
[open.2]: https://www.man7.org/linux/man-pages/man2/open.2.html

## [0.2.5] - 2024-05-03 ##

### Changed ###
- Some minor changes were made to how lexical components (like `..` and `.`)
  are handled during path generation in `SecureJoin`. There is no behaviour
  change as a result of this fix (the resulting paths are the same).

### Fixed ###
- The error returned when we hit a symlink loop now references the correct
  path. (#10)

## [0.2.4] - 2023-09-06 ##

### Security ###
- This release fixes a potential security issue in filepath-securejoin when
  used on Windows ([GHSA-6xv5-86q9-7xr8][], which could be used to generate
  paths outside of the provided rootfs in certain cases), as well as improving
  the overall behaviour of filepath-securejoin when dealing with Windows paths
  that contain volume names. Thanks to Paulo Gomes for discovering and fixing
  these issues.

### Fixed ###
- Switch to GitHub Actions for CI so we can test on Windows as well as Linux
  and MacOS.

[GHSA-6xv5-86q9-7xr8]: https://github.com/advisories/GHSA-6xv5-86q9-7xr8

## [0.2.3] - 2021-06-04 ##

### Changed ###
- Switch to Go 1.13-style `%w` error wrapping, letting us drop the dependency
  on `github.com/pkg/errors`.

## [0.2.2] - 2018-09-05 ##

### Changed ###
- Use `syscall.ELOOP` as the base error for symlink loops, rather than our own
  (internal) error. This allows callers to more easily use `errors.Is` to check
  for this case.

## [0.2.1] - 2018-09-05 ##

### Fixed ###
- Use our own `IsNotExist` implementation, which lets us handle `ENOTDIR`
  properly within `SecureJoin`.

## [0.2.0] - 2017-07-19 ##

We now have 100% test coverage!

### Added ###
- Add a `SecureJoinVFS` API that can be used for mocking (as we do in our new
  tests) or for implementing custom handling of lookup operations (such as for
  rootless containers, where work is necessary to access directories with weird
  modes because we don't have `CAP_DAC_READ_SEARCH` or `CAP_DAC_OVERRIDE`).

## 0.1.0 - 2017-07-19

This is our first release of `github.com/cyphar/filepath-securejoin`,
containing a full implementation with a coverage of 93.5% (the only missing
cases are the error cases, which are hard to mocktest at the moment).

[Unreleased]: https://github.com/cyphar/filepath-securejoin/compare/v0.3.4...HEAD
[0.3.3]: https://github.com/cyphar/filepath-securejoin/compare/v0.3.3...v0.3.4
[0.3.3]: https://github.com/cyphar/filepath-securejoin/compare/v0.3.2...v0.3.3
[0.3.2]: https://github.com/cyphar/filepath-securejoin/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/cyphar/filepath-securejoin/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.5...v0.3.0
[0.2.5]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.4...v0.2.5
[0.2.4]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.3...v0.2.4
[0.2.3]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.2...v0.2.3
[0.2.2]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/cyphar/filepath-securejoin/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/cyphar/filepath-securejoin/compare/v0.1.0...v0.2.0
