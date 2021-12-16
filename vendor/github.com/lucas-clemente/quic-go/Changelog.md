# Changelog

## v0.22.0 (2021-07-25)

- Use `ReadBatch` to read multiple UDP packets from the socket with a single syscall
- Add a config option (`Config.DisableVersionNegotiationPackets`) to disable sending of Version Negotiation packets
- Drop support for QUIC draft versions 32 and 34
- Remove the `RetireBugBackwardsCompatibilityMode`, which was intended to mitigate a bug when retiring connection IDs in quic-go in v0.17.2 and ealier

## v0.21.2 (2021-07-15)

- Update qtls (for Go 1.15, 1.16 and 1.17rc1) to include the fix for the crypto/tls panic (see https://groups.google.com/g/golang-dev/c/5LJ2V7rd-Ag/m/YGLHVBZ6AAAJ for details)

## v0.21.0 (2021-06-01)

- quic-go now supports RFC 9000!

## v0.20.0 (2021-03-19)

- Remove the `quic.Config.HandshakeTimeout`. Introduce a `quic.Config.HandshakeIdleTimeout`.

## v0.17.1 (2020-06-20)

- Supports QUIC WG draft-29.
- Improve bundling of ACK frames (#2543).

## v0.16.0 (2020-05-31)

- Supports QUIC WG draft-28.

## v0.15.0 (2020-03-01)

- Supports QUIC WG draft-27.
- Add support for 0-RTT.
- Remove `Session.Close()`. Applications need to pass an application error code to the transport using `Session.CloseWithError()`.
- Make the TLS Cipher Suites configurable (via `tls.Config.CipherSuites`).

## v0.14.0 (2019-12-04)

- Supports QUIC WG draft-24.

## v0.13.0 (2019-11-05)

- Supports QUIC WG draft-23.
- Add an `EarlyListener` that allows sending of 0.5-RTT data.
- Add a `TokenStore` to store address validation tokens.
- Issue and use new connection IDs during a connection.

## v0.12.0 (2019-08-05)

- Implement HTTP/3.
- Rename `quic.Cookie` to `quic.Token` and `quic.Config.AcceptCookie` to `quic.Config.AcceptToken`.
- Distinguish between Retry tokens and tokens sent in NEW_TOKEN frames.
- Enforce application protocol negotiation (via `tls.Config.NextProtos`).
- Use a varint for error codes.
- Add support for [quic-trace](https://github.com/google/quic-trace).
- Add a context to `Listener.Accept`, `Session.Accept{Uni}Stream` and `Session.Open{Uni}StreamSync`.
- Implement TLS key updates.

## v0.11.0 (2019-04-05)

- Drop support for gQUIC. For qQUIC support, please switch to the *gquic* branch.
- Implement QUIC WG draft-19.
- Use [qtls](https://github.com/marten-seemann/qtls) for TLS 1.3.
- Return a `tls.ConnectionState` from `quic.Session.ConnectionState()`.
- Remove the error return values from `quic.Stream.CancelRead()` and `quic.Stream.CancelWrite()`

## v0.10.0 (2018-08-28)

- Add support for QUIC 44, drop support for QUIC 42.

## v0.9.0 (2018-08-15)

- Add a `quic.Config` option for the length of the connection ID (for IETF QUIC).
- Split Session.Close into one method for regular closing and one for closing with an error.

## v0.8.0 (2018-06-26)

- Add support for unidirectional streams (for IETF QUIC).
- Add a `quic.Config` option for the maximum number of incoming streams.
- Add support for QUIC 42 and 43.
- Add dial functions that use a context.
- Multiplex clients on a net.PacketConn, when using Dial(conn).

## v0.7.0 (2018-02-03)

- The lower boundary for packets included in ACKs is now derived, and the value sent in STOP_WAITING frames is ignored.
- Remove `DialNonFWSecure` and `DialAddrNonFWSecure`.
- Expose the `ConnectionState` in the `Session` (experimental API).
- Implement packet pacing.

## v0.6.0 (2017-12-12)

- Add support for QUIC 39, drop support for QUIC 35 - 37
- Added `quic.Config` options for maximal flow control windows
- Add a `quic.Config` option for QUIC versions
- Add a `quic.Config` option to request omission of the connection ID from a server
- Add a `quic.Config` option to configure the source address validation
- Add a `quic.Config` option to configure the handshake timeout
- Add a `quic.Config` option to configure the idle timeout
- Add a `quic.Config` option to configure keep-alive
- Rename the STK to Cookie
- Implement `net.Conn`-style deadlines for streams
- Remove the `tls.Config` from the `quic.Config`. The `tls.Config` must now be passed to the `Dial` and `Listen` functions as a separate parameter. See the [Godoc](https://godoc.org/github.com/lucas-clemente/quic-go) for details.
- Changed the log level environment variable to only accept strings ("DEBUG", "INFO", "ERROR"), see [the wiki](https://github.com/lucas-clemente/quic-go/wiki/Logging) for more details.
- Rename the `h2quic.QuicRoundTripper` to `h2quic.RoundTripper`
- Changed `h2quic.Server.Serve()` to accept a `net.PacketConn`
- Drop support for Go 1.7 and 1.8.
- Various bugfixes
