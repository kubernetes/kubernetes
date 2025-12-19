## `filepath-securejoin` ##

[![Go Documentation](https://pkg.go.dev/badge/github.com/cyphar/filepath-securejoin.svg)](https://pkg.go.dev/github.com/cyphar/filepath-securejoin)
[![Build Status](https://github.com/cyphar/filepath-securejoin/actions/workflows/ci.yml/badge.svg)](https://github.com/cyphar/filepath-securejoin/actions/workflows/ci.yml)

### Old API ###

This library was originally just an implementation of `SecureJoin` which was
[intended to be included in the Go standard library][go#20126] as a safer
`filepath.Join` that would restrict the path lookup to be inside a root
directory.

The implementation was based on code that existed in several container
runtimes. Unfortunately, this API is **fundamentally unsafe** against attackers
that can modify path components after `SecureJoin` returns and before the
caller uses the path, allowing for some fairly trivial TOCTOU attacks.

`SecureJoin` (and `SecureJoinVFS`) are still provided by this library to
support legacy users, but new users are strongly suggested to avoid using
`SecureJoin` and instead use the [new api](#new-api) or switch to
[libpathrs][libpathrs].

With the above limitations in mind, this library guarantees the following:

* If no error is set, the resulting string **must** be a child path of
  `root` and will not contain any symlink path components (they will all be
  expanded).

* When expanding symlinks, all symlink path components **must** be resolved
  relative to the provided root. In particular, this can be considered a
  userspace implementation of how `chroot(2)` operates on file paths. Note that
  these symlinks will **not** be expanded lexically (`filepath.Clean` is not
  called on the input before processing).

* Non-existent path components are unaffected by `SecureJoin` (similar to
  `filepath.EvalSymlinks`'s semantics).

* The returned path will always be `filepath.Clean`ed and thus not contain any
  `..` components.

A (trivial) implementation of this function on GNU/Linux systems could be done
with the following (note that this requires root privileges and is far more
opaque than the implementation in this library, and also requires that
`readlink` is inside the `root` path and is trustworthy):

```go
package securejoin

import (
	"os/exec"
	"path/filepath"
)

func SecureJoin(root, unsafePath string) (string, error) {
	unsafePath = string(filepath.Separator) + unsafePath
	cmd := exec.Command("chroot", root,
		"readlink", "--canonicalize-missing", "--no-newline", unsafePath)
	output, err := cmd.CombinedOutput()
	if err != nil {
		return "", err
	}
	expanded := string(output)
	return filepath.Join(root, expanded), nil
}
```

[libpathrs]: https://github.com/openSUSE/libpathrs
[go#20126]: https://github.com/golang/go/issues/20126

### <a name="new-api" /> New API ###
[#new-api]: #new-api

While we recommend users switch to [libpathrs][libpathrs] as soon as it has a
stable release, some methods implemented by libpathrs have been ported to this
library to ease the transition. These APIs are only supported on Linux.

These APIs are implemented such that `filepath-securejoin` will
opportunistically use certain newer kernel APIs that make these operations far
more secure. In particular:

* All of the lookup operations will use [`openat2`][openat2.2] on new enough
  kernels (Linux 5.6 or later) to restrict lookups through magic-links and
  bind-mounts (for certain operations) and to make use of `RESOLVE_IN_ROOT` to
  efficiently resolve symlinks within a rootfs.

* The APIs provide hardening against a malicious `/proc` mount to either detect
  or avoid being tricked by a `/proc` that is not legitimate. This is done
  using [`openat2`][openat2.2] for all users, and privileged users will also be
  further protected by using [`fsopen`][fsopen.2] and [`open_tree`][open_tree.2]
  (Linux 5.2 or later).

[openat2.2]: https://www.man7.org/linux/man-pages/man2/openat2.2.html
[fsopen.2]: https://github.com/brauner/man-pages-md/blob/main/fsopen.md
[open_tree.2]: https://github.com/brauner/man-pages-md/blob/main/open_tree.md

#### `OpenInRoot` ####

```go
func OpenInRoot(root, unsafePath string) (*os.File, error)
func OpenatInRoot(root *os.File, unsafePath string) (*os.File, error)
func Reopen(handle *os.File, flags int) (*os.File, error)
```

`OpenInRoot` is a much safer version of

```go
path, err := securejoin.SecureJoin(root, unsafePath)
file, err := os.OpenFile(path, unix.O_PATH|unix.O_CLOEXEC)
```

that protects against various race attacks that could lead to serious security
issues, depending on the application. Note that the returned `*os.File` is an
`O_PATH` file descriptor, which is quite restricted. Callers will probably need
to use `Reopen` to get a more usable handle (this split is done to provide
useful features like PTY spawning and to avoid users accidentally opening bad
inodes that could cause a DoS).

Callers need to be careful in how they use the returned `*os.File`. Usually it
is only safe to operate on the handle directly, and it is very easy to create a
security issue. [libpathrs][libpathrs] provides far more helpers to make using
these handles safer -- there is currently no plan to port them to
`filepath-securejoin`.

`OpenatInRoot` is like `OpenInRoot` except that the root is provided using an
`*os.File`. This allows you to ensure that multiple `OpenatInRoot` (or
`MkdirAllHandle`) calls are operating on the same rootfs.

> **NOTE**: Unlike `SecureJoin`, `OpenInRoot` will error out as soon as it hits
> a dangling symlink or non-existent path. This is in contrast to `SecureJoin`
> which treated non-existent components as though they were real directories,
> and would allow for partial resolution of dangling symlinks. These behaviours
> are at odds with how Linux treats non-existent paths and dangling symlinks,
> and so these are no longer allowed.

#### `MkdirAll` ####

```go
func MkdirAll(root, unsafePath string, mode int) error
func MkdirAllHandle(root *os.File, unsafePath string, mode int) (*os.File, error)
```

`MkdirAll` is a much safer version of

```go
path, err := securejoin.SecureJoin(root, unsafePath)
err = os.MkdirAll(path, mode)
```

that protects against the same kinds of races that `OpenInRoot` protects
against.

`MkdirAllHandle` is like `MkdirAll` except that the root is provided using an
`*os.File` (the reason for this is the same as with `OpenatInRoot`) and an
`*os.File` of the final created directory is returned (this directory is
guaranteed to be effectively identical to the directory created by
`MkdirAllHandle`, which is not possible to ensure by just using `OpenatInRoot`
after `MkdirAll`).

> **NOTE**: Unlike `SecureJoin`, `MkdirAll` will error out as soon as it hits
> a dangling symlink or non-existent path. This is in contrast to `SecureJoin`
> which treated non-existent components as though they were real directories,
> and would allow for partial resolution of dangling symlinks. These behaviours
> are at odds with how Linux treats non-existent paths and dangling symlinks,
> and so these are no longer allowed. This means that `MkdirAll` will not
> create non-existent directories referenced by a dangling symlink.

### License ###

`SPDX-License-Identifier: BSD-3-Clause AND MPL-2.0`

Some of the code in this project is derived from Go, and is licensed under a
BSD 3-clause license (available in `LICENSE.BSD`). Other files (many of which
are derived from [libpathrs][libpathrs]) are licensed under the Mozilla Public
License version 2.0 (available in `LICENSE.MPL-2.0`). If you are using the
["New API" described above][#new-api], you are probably using code from files
released under this license.

Every source file in this project has a copyright header describing its
license. Please check the license headers of each file to see what license
applies to it.

See [COPYING.md](./COPYING.md) for some more details.

[umoci]: https://github.com/opencontainers/umoci
