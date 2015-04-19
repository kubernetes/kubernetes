
0.9.17 / 2014-05-22
===================

 * use static channels for remote syncing instead of subscribing/unsubscribing 5 channels for every connection
 * Use destroy buffer size on websocket transport method as well
 * http-polling : adding 'X-XSS-Protection : 0;' to headers necessary not only to jsonp-polling but http-polling

0.9.16 / 2013-06-06
===================

  * transports: added tests for htmlfile escaping/unescaping

0.9.15 / 2013-06-06
===================

  * transports: added escaping to htmlfile (fixes #1251)

0.9.14 / 2013-03-29
===================

  * manager: fix memory leak with SSL [jpallen]

0.9.13 / 2012-12-13
===================

  * package: fixed `base64id` requirement

0.9.12 / 2012-12-13
===================

  * manager: fix for latest node which is returning a clone with `listeners` [viirya]

0.9.11 / 2012-11-02
===================

  * package: move redis to optionalDependenices [3rd-Eden]
  * bumped client

0.9.10 / 2012-08-10
===================

  * Don't lowercase log messages
  * Always set the HTTP response in case an error should be returned to the client
  * Create or destroy the flash policy server on configuration change
  * Honour configuration to disable flash policy server
  * Add express 3.0 instructions on Readme.md
  * Bump client

0.9.9 / 2012-08-01
==================

  * Fixed sync disconnect xhrs handling
  * Put license text in its own file (#965)
  * Add warning to .listen() to ease the migration to Express 3.x
  * Restored compatibility with node 0.4.x

0.9.8 / 2012-07-24
==================

  * Bumped client.

0.9.7 / 2012-07-24
==================

  * Prevent crash when socket leaves a room twice.
  * Corrects unsafe usage of for..in
  * Fix for node 0.8 with `gzip compression` [vadimi]
  * Update redis to support Node 0.8.x
  * Made ID generation securely random
  * Fix Redis Store race condition in manager onOpen unsubscribe callback
  * Fix for EventEmitters always reusing the same Array instance for listeners

0.9.6 / 2012-04-17
==================

  * Fixed XSS in jsonp-polling.

0.9.5 / 2012-04-05
==================

  * Added test for polling and socket close.
  * Ensure close upon request close.
  * Fix disconnection reason being lost for polling transports.
  * Ensure that polling transports work with Connection: close.
  * Log disconnection reason.

0.9.4 / 2012-04-01
==================

  * Disconnecting from namespace improvement (#795) [DanielBaulig]
  * Bumped client with polling reconnection loop (#438)

0.9.3 / 2012-03-28
==================

  * Fix "Syntax error" on FF Web Console with XHR Polling [mikito]

0.9.2 / 2012-03-13
==================

  * More sensible close `timeout default` (fixes disconnect issue)

0.9.1-1 / 2012-03-02
====================

  * Bumped client with NPM dependency fix.

0.9.1 / 2012-03-02
==================

  * Changed heartbeat timeout and interval defaults (60 and 25 seconds)
  * Make tests work both on 0.4 and 0.6
  * Updated client (improvements + bug fixes).

0.9.0 / 2012-02-26
==================

  * Make it possible to use a regexp to match the socket.io resource URL.
    We need this because we have to prefix the socket.io URL with a variable ID.
  * Supplemental fix to gavinuhma/authfix, it looks like the same Access-Control-Origin logic is needed in the http and xhr-polling transports
  * Updated express dep for windows compatibility.
  * Combine two substr calls into one in decodePayload to improve performance
  * Minor documentation fix
  * Minor. Conform to style of other files.
  * Switching setting to 'match origin protocol'
  * Revert "Fixes leaking Redis subscriptions for #663. The local flag was not getting passed through onClientDisconnect()."
  * Revert "Handle leaked dispatch:[id] subscription."
  * Merge pull request #667 from dshaw/patch/redis-disconnect
  * Handle leaked dispatch:[id] subscription.
  * Fixes leaking Redis subscriptions for #663. The local flag was not getting passed through onClientDisconnect().
  * Prevent memory leaking on uncompleted requests & add max post size limitation
  * Fix for testcase
  * Set Access-Control-Allow-Credentials true, regardless of cookie
  * Remove assertvarnish from package as it breaks on 0.6
  * Correct irc channel
  * Added proper return after reserved field error
  * Fixes manager.js failure to close connection after transport error has happened
  * Added implicit port 80 for origin checks. fixes #638
  * Fixed bug #432 in 0.8.7
  * Set Access-Control-Allow-Origin header to origin to enable withCredentials
  * Adding configuration variable matchOriginProtocol
  * Fixes location mismatch error in Safari.
  * Use tty to detect if we should add colors or not by default.
  * Updated the package location.

0.8.7 / 2011-11-05
==================

  * Fixed memory leaks in closed clients.
  * Fixed memory leaks in namespaces.
  * Fixed websocket handling for malformed requests from proxies. [einaros]
  * Node 0.6 compatibility. [einaros] [3rd-Eden]
  * Adapted tests and examples.

0.8.6 / 2011-10-27 
==================

  * Added JSON decoding on jsonp-polling transport.
  * Fixed README example.
  * Major speed optimizations [3rd-Eden] [einaros] [visionmedia]
  * Added decode/encode benchmarks [visionmedia]
  * Added support for black-listing client sent events.
  * Fixed logging options, closes #540 [3rd-Eden]
  * Added vary header for gzip [3rd-Eden]
  * Properly cleaned up async websocket / flashsocket tests, after patching node-websocket-client
  * Patched to properly shut down when a finishClose call is made during connection establishment
  * Added support for socket.io version on url and far-future Expires [3rd-Eden] [getify]
  * Began IE10 compatibility [einaros] [tbranyen]
  * Misc WebSocket fixes [einaros]
  * Added UTF8 to respone headers for htmlfile [3rd-Eden]

0.8.5 / 2011-10-07
==================

  * Added websocket draft HyBi-16 support. [einaros]
  * Fixed websocket continuation bugs. [einaros]
  * Fixed flashsocket transport name.
  * Fixed websocket tests.
  * Ensured `parser#decodePayload` doesn't choke.
  * Added http referrer verification to manager verifyOrigin.
  * Added access control for cross domain xhr handshakes [3rd-Eden]
  * Added support for automatic generation of socket.io files [3rd-Eden]
  * Added websocket binary support [einaros]
  * Added gzip support for socket.io.js [3rd-Eden]
  * Expose socket.transport [3rd-Eden]
  * Updated client.

0.8.4 / 2011-09-06
==================

  * Client build

0.8.3 / 2011-09-03
==================

  * Fixed `\n` parsing for non-JSON packets (fixes #479).
  * Fixed parsing of certain unicode characters (fixes #451).
  * Fixed transport message packet logging.
  * Fixed emission of `error` event resulting in an uncaught exception if unhandled (fixes #476).
  * Fixed; allow for falsy values as the configuration value of `log level` (fixes #491).
  * Fixed repository URI in `package.json`. Fixes #504.
  * Added text/plain content-type to handshake responses [einaros]
  * Improved single byte writes [einaros]
  * Updated socket.io-flashsocket default port from 843 to 10843 [3rd-Eden]
  * Updated client.

0.8.2 / 2011-08-29
==================

  * Updated client.

0.8.1 / 2011-08-29
==================

  * Fixed utf8 bug in send framing in websocket [einaros]
  * Fixed typo in docs [Znarkus]
  * Fixed bug in send framing for over 64kB of data in websocket [einaros]
  * Corrected ping handling in websocket transport [einaros]

0.8.0 / 2011-08-28
==================

  * Updated to work with two-level websocket versioning. [einaros]
  * Added hybi07 support. [einaros]
  * Added hybi10 support. [einaros]
  * Added http referrer verification to manager.js verifyOrigin. [einaors]

0.7.11 / 2011-08-27
===================

  * Updated socket.io-client.

0.7.10 / 2011-08-27
===================

  * Updated socket.io-client.

0.7.9 / 2011-08-12
==================

  * Updated socket.io-client.
  * Make sure we only do garbage collection when the server we receive is actually run.

0.7.8 / 2011-08-08
==================

  * Changed; make sure sio#listen passes options to both HTTP server and socket.io manager.
  * Added docs for sio#listen.
  * Added options parameter support for Manager constructor.
  * Added memory leaks tests and test-leaks Makefile task.
  * Removed auto npm-linking from make test.
  * Make sure that you can disable heartbeats. [3rd-Eden]
  * Fixed rooms memory leak [3rd-Eden]
  * Send response once we got all POST data, not immediately [Pita]
  * Fixed onLeave behavior with missing clientsk [3rd-Eden]
  * Prevent duplicate references in rooms.
  * Added alias for `to` to `in` and `in` to `to`.
  * Fixed roomClients definition.
  * Removed dependency on redis for installation without npm [3rd-Eden]
  * Expose path and querystring in handshakeData [3rd-Eden]

0.7.7 / 2011-07-12
==================

  * Fixed double dispatch handling with emit to closed clients.
  * Added test for emitting to closed clients to prevent regression.
  * Fixed race condition in redis test.
  * Changed Transport#end instrumentation.
  * Leveraged $emit instead of emit internally.
  * Made tests faster.
  * Fixed double disconnect events.
  * Fixed disconnect logic
  * Simplified remote events handling in Socket.
  * Increased testcase timeout.
  * Fixed unknown room emitting (GH-291). [3rd-Eden]
  * Fixed `address` in handshakeData. [3rd-Eden]
  * Removed transports definition in chat example.
  * Fixed room cleanup
  * Fixed; make sure the client is cleaned up after booting.
  * Make sure to mark the client as non-open if the connection is closed.
  * Removed unneeded `buffer` declarations.
  * Fixed; make sure to clear socket handlers and subscriptions upon transport close.

0.7.6 / 2011-06-30
==================

  * Fixed general dispatching when a client has closed.

0.7.5 / 2011-06-30
==================

  * Fixed dispatching to clients that are disconnected.

0.7.4 / 2011-06-30
==================

  * Fixed; only clear handlers if they were set. [level09]

0.7.3 / 2011-06-30
==================

  * Exposed handshake data to clients.
  * Refactored dispatcher interface.
  * Changed; Moved id generation method into the manager.
  * Added sub-namespace authorization. [3rd-Eden]
  * Changed; normalized SocketNamespace local eventing [dvv]
  * Changed; Use packet.reason or default to 'packet' [3rd-Eden]
  * Changed console.error to console.log.
  * Fixed; bind both servers at the same time do that the test never times out.
  * Added 304 support.
  * Removed `Transport#name` for abstract interface.
  * Changed; lazily require http and https module only when needed. [3rd-Eden]

0.7.2 / 2011-06-22
==================

  * Make sure to write a packet (of type `noop`) when closing a poll.
    This solves a problem with cross-domain requests being flagged as aborted and
    reconnection being triggered.
  * Added `noop` message type.

0.7.1 / 2011-06-21 
==================

  * Fixed cross-domain XHR.
  * Added CORS test to xhr-polling suite.

0.7.0 / 2010-06-21
==================

  * http://socket.io/announcement.html
