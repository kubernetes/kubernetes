## `pathrs-lite` ##

`github.com/cyphar/filepath-securejoin/pathrs-lite` provides a minimal **pure
Go** implementation of the core bits of [libpathrs][]. This is not intended to
be a complete replacement for libpathrs, instead it is mainly intended to be
useful as a transition tool for existing Go projects.

`pathrs-lite` also provides a very easy way to switch to `libpathrs` (even for
downstreams where `pathrs-lite` is being used in a third-party package and is
not interested in using CGo). At build time, if you use the `libpathrs` build
tag then `pathrs-lite` will use `libpathrs` directly instead of the pure Go
implementation. The two backends are functionally equivalent (and we have
integration tests to verify this), so this migration should be very easy with
no user-visible impact.

[libpathrs]: https://github.com/cyphar/libpathrs

### License ###

Most of this subpackage is licensed under the Mozilla Public License (version
2.0). For more information, see the top-level [COPYING.md][] and
[LICENSE.MPL-2.0][] files, as well as the individual license headers for each
file.

```
Copyright (C) 2024-2025 Aleksa Sarai <cyphar@cyphar.com>
Copyright (C) 2024-2025 SUSE LLC

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.
```

[COPYING.md]: ../COPYING.md
[LICENSE.MPL-2.0]: ../LICENSE.MPL-2.0
