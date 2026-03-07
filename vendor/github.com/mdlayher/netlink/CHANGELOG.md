# CHANGELOG

## v1.7.2

- [Improvement]: updated dependencies, test with Go 1.20.

## v1.7.1

- [Bug Fix]: test only changes to avoid failures on big endian machines.

## v1.7.0

**This is the first release of package netlink that only supports Go 1.18+.
Users on older versions of Go must use v1.6.2.**

- [Improvement]: drop support for older versions of Go so we can begin using
  modern versions of `x/sys` and other dependencies.

## v1.6.2

**This is the last release of package netlink that supports Go 1.17 and below.**

- [Bug Fix] [commit](https://github.com/mdlayher/netlink/commit/9f7f860d9865069cd1a6b4dee32a3095f0b841fc):
  undo update to `golang.org/x/sys` which would force the minimum Go version of
  this package to Go 1.17 due to use of `unsafe.Slice`. We encourage users to
  use the latest stable version of Go where possible, but continue to maintain
  some compatibility with older versions of Go as long as it is reasonable to do
  so.

## v1.6.1

- [Deprecation] [commit](https://github.com/mdlayher/netlink/commit/d1b69ea8697d721415c259ef8513ab699c6d3e96): 
  the `netlink.Socket` interface has been marked as deprecated. The abstraction
  is awkward to use properly and disables much of the functionality of the Conn
  type when the basic interface is implemented. Do not use.

## v1.6.0

**This is the first release of package netlink that only supports Go 1.13+.
Users on older versions of Go must use v1.5.0.**

- [New API] [commit](https://github.com/mdlayher/netlink/commit/ad9e2c41caa993e3f4b68831d6cb2cb05818275d):
  the `netlink.Config.Strict` field can be used to apply a more strict default
  set of options to a `netlink.Conn`. This is recommended for applications
  running on modern Linux kernels, but cannot be enabled by default because the
  options may require a more recent kernel than the minimum kernel version that
  Go supports. See the documentation for details.
- [Improvement]: broke some integration tests into a separate Go module so the
  default `go.mod` for package `netlink` has fewer dependencies.

## v1.5.0

**This is the last release of package netlink that supports Go 1.12.**

- [New API] [commit](https://github.com/mdlayher/netlink/commit/53a1c10065e51077659ceedf921c8f0807abe8c0):
  the `netlink.Config.PID` field can be used to specify an explicit port ID when
  binding the netlink socket. This is intended for advanced use cases and most
  callers should leave this field set to 0.
- [Improvement]: more low-level functionality ported to
  `github.com/mdlayher/socket`, reducing package complexity.

## v1.4.2

- [Documentation] [commit](https://github.com/mdlayher/netlink/commit/177e6364fb170d465d681c7c8a6283417a6d3e49):
  the `netlink.Config.DisableNSLockThread` now properly uses Go's deprecated
  identifier convention. This option has been a noop for a long time and should
  not be used.
- [Improvement] [#189](https://github.com/mdlayher/netlink/pull/189): the
  package now uses Go 1.17's `//go:build` identifiers. Thanks @tklauser.
- [Bug Fix]
  [commit](https://github.com/mdlayher/netlink/commit/fe6002e030928bd1f2a446c0b6c65e8f2df4ed5e):
  the `netlink.AttributeEncoder`'s `Bytes`, `String`, and `Do` methods now
  properly reject byte slices and strings which are too large to fit in the
  value of a netlink attribute. Thanks @ubiquitousbyte for the report.

## v1.4.1

- [Improvement]: significant runtime network poller integration cleanup through
  the use of `github.com/mdlayher/socket`.

## v1.4.0

- [New API] [#185](https://github.com/mdlayher/netlink/pull/185): the
  `netlink.AttributeDecoder` and `netlink.AttributeEncoder` types now have
  methods for dealing with signed integers: `Int8`, `Int16`, `Int32`, and
  `Int64`. These are necessary for working with rtnetlink's XDP APIs. Thanks
  @fbegyn.

## v1.3.2

- [Improvement]
  [commit](https://github.com/mdlayher/netlink/commit/ebc6e2e28bcf1a0671411288423d8116ff924d6d):
  `github.com/google/go-cmp` is no longer a (non-test) dependency of this module.

## v1.3.1

- [Improvement]: many internal cleanups and simplifications. The library is now
  slimmer and features less internal indirection. There are no user-facing
  changes in this release.

## v1.3.0

- [New API] [#176](https://github.com/mdlayher/netlink/pull/176):
  `netlink.OpError` now has `Message` and `Offset` fields which are populated
  when the kernel returns netlink extended acknowledgement data along with an
  error code. The caller can turn on this option by using
  `netlink.Conn.SetOption(netlink.ExtendedAcknowledge, true)`.
- [New API]
  [commit](https://github.com/mdlayher/netlink/commit/beba85e0372133b6d57221191d2c557727cd1499):
  the `netlink.GetStrictCheck` option can be used to tell the kernel to be more
  strict when parsing requests. This enables more safety checks and can allow
  the kernel to perform more advanced request filtering in subsystems such as
  route netlink.

## v1.2.1

- [Bug Fix]
  [commit](https://github.com/mdlayher/netlink/commit/d81418f81b0bfa2465f33790a85624c63d6afe3d):
  `netlink.SetBPF` will no longer panic if an empty BPF filter is set.
- [Improvement]
  [commit](https://github.com/mdlayher/netlink/commit/8014f9a7dbf4fd7b84a1783dd7b470db9113ff36):
  the library now uses https://github.com/josharian/native to provide the
  system's native endianness at compile time, rather than re-computing it many
  times at runtime.

## v1.2.0

**This is the first release of package netlink that only supports Go 1.12+.
Users on older versions of Go must use v1.1.1.**

- [Improvement] [#173](https://github.com/mdlayher/netlink/pull/173): support
  for Go 1.11 and below has been dropped. All users are highly recommended to
  use a stable and supported release of Go for their applications.
- [Performance] [#171](https://github.com/mdlayher/netlink/pull/171):
  `netlink.Conn` no longer requires a locked OS thread for the vast majority of
  operations, which should result in a significant speedup for highly concurrent
  callers. Thanks @ti-mo.
- [Bug Fix] [#169](https://github.com/mdlayher/netlink/pull/169): calls to
  `netlink.Conn.Close` are now able to unblock concurrent calls to
  `netlink.Conn.Receive` and other blocking operations.

## v1.1.1

**This is the last release of package netlink that supports Go 1.11.**

- [Improvement] [#165](https://github.com/mdlayher/netlink/pull/165):
  `netlink.Conn` `SetReadBuffer` and `SetWriteBuffer` methods now attempt the
  `SO_*BUFFORCE` socket options to possibly ignore system limits given elevated
  caller permissions. Thanks @MarkusBauer.
- [Note]
  [commit](https://github.com/mdlayher/netlink/commit/c5f8ab79aa345dcfcf7f14d746659ca1b80a0ecc):
  `netlink.Conn.Close` has had a long-standing bug
  [#162](https://github.com/mdlayher/netlink/pull/162) related to internal
  concurrency handling where a call to `Close` is not sufficient to unblock
  pending reads. To effectively fix this issue, it is necessary to drop support
  for Go 1.11 and below. This will be fixed in a future release, but a
  workaround is noted in the method documentation as of now.

## v1.1.0

- [New API] [#157](https://github.com/mdlayher/netlink/pull/157): the
  `netlink.AttributeDecoder.TypeFlags` method enables retrieval of the type bits
  stored in a netlink attribute's type field, because the existing `Type` method
  masks away these bits. Thanks @ti-mo!
- [Performance] [#157](https://github.com/mdlayher/netlink/pull/157): `netlink.AttributeDecoder`
  now decodes netlink attributes on demand, enabling callers who only need a
  limited number of attributes to exit early from decoding loops. Thanks @ti-mo!
- [Improvement] [#161](https://github.com/mdlayher/netlink/pull/161): `netlink.Conn`
  system calls are now ready for Go 1.14+'s changes to goroutine preemption.
  See the PR for details.

## v1.0.0

- Initial stable commit.
