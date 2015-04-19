
0.9.16 / 2013-06-06
===================

  * transports: fix escaping for tests

0.9.15 / 2013-06-06
===================

  * transports: added unescaping for escaped htmlfile
  * skipped 12-14 to match socket.io server version

0.9.11 / 2012-11-02
===================

  * Enable use of 'xhr' transport in Node.js
  * Fix the problem with disconnecting xhr-polling users
  * Add should to devDependencies
  * Prefer XmlHttpRequest if CORS is available
  * Make client compatible with AMD loaders.

0.9.10 / 2012-08-10
===================

  * fix removeAllListeners to behave as expected.
  * set withCredentials to true only if xdomain.
  * socket: disable disconnect on unload by default.

0.9.9 / 2012-08-01
==================

  * socket: fixed disconnect xhr url and made it actually sync
  * *: bump xmlhttprequest dep

0.9.8 / 2012-07-24
==================

  * Fixed build.

0.9.7 / 2012-07-24
==================

  * iOS websocket crash fix.
  * Fixed potential `open` collision.
  * Fixed disconnectSync.

0.9.6 / 2012-04-17
==================

  * Don't position the jsonp form off the screen (android fix).

0.9.5 / 2012-04-05
==================

  * Bumped version.

0.9.4 / 2012-04-01
==================

  * Fixes polling loop upon reconnect advice (fixes #438).

0.9.3 / 2012-03-28
==================

  * Fix XHR.check, which was throwing an error transparently and causing non-IE browsers to fall back to JSONP [mikito]
  * Fixed forced disconnect on window close [zzzaaa]

0.9.2 / 2012-03-13
==================

  * Transport order set by "options" [zzzaaa]

0.9.1-1 / 2012-03-02
====================

  * Fixed active-x-obfuscator NPM dependency.

0.9.1 / 2012-03-02
==================

  * Misc corrections.
  * Added warning within Firefox about webworker test in test runner.
  * Update ws dependency [einaros]
  * Implemented client side heartbeat checks. [felixge]
  * Improved Firewall support with ActiveX obfuscation. [felixge]
  * Fixed error handling during connection process. [Outsideris]

0.9.0 / 2012-02-26
==================

  * Added DS_Store to gitignore.
  * Updated depedencies.
  * Bumped uglify
  * Tweaking code so it doesn't throw an exception when used inside a WebWorker in Firefox
  * Do not rely on Array.prototype.indexOf as it breaks with pages that use the Prototype.js library.
  * Windows support landed
  * Use @einaros ws module instead of the old crap one
  * Fix for broken closeTimeout and 'IE + xhr' goes into infinite loop on disconnection
  * Disabled reconnection on error if reconnect option is set to false
  * Set withCredentials to true before xhr to fix authentication
  * Clears the timeout from reconnection attempt when there is a successful or failed reconnection. 
    This fixes the issue of setTimeout's carrying over from previous reconnection
    and changing (skipping) values of self.reconnectionDelay in the newer reconnection.
  * Removed decoding of parameters when chunking the query string.
    This was used later on to construct the url to post to the socket.io server
    for connection and if we're adding custom parameters of our own to this url
    (for example for OAuth authentication) they were being sent decoded, which is wrong.

0.8.7 / 2011-11-05
==================

  * Bumped client

0.8.6 / 2011-10-27 
==================

  * Added WebWorker support.
  * Fixed swfobject and web_socket.js to not assume window.
  * Fixed CORS detection for webworker.
  * Fix `defer` for webkit in a webworker.
  * Fixed io.util.request to not rely on window.
  * FIxed; use global instead of window and dont rely on document.
  * Fixed; JSON-P handshake if CORS is not available.
  * Made underlying Transport disconnection trigger immediate socket.io disconnect.
  * Fixed warning when compressing with Google Closure Compiler.
  * Fixed builder's uglify utf-8 support.
  * Added workaround for loading indicator in FF jsonp-polling. [3rd-Eden]
  * Fixed host discovery lookup. [holic]
  * Fixed close timeout when disconnected/reconnecting. [jscharlach]
  * Fixed jsonp-polling feature detection.
  * Fixed jsonp-polling client POSTing of \n.
  * Fixed test runner on IE6/7

0.8.5 / 2011-10-07
==================

  * Bumped client

0.8.4 / 2011-09-06
==================

  * Corrected build

0.8.3 / 2011-09-03
==================

  * Fixed `\n` parsing for non-JSON packets.
  * Fixed; make Socket.IO XHTML doctype compatible (fixes #460 from server)
  * Fixed support for Node.JS running `socket.io-client`.
  * Updated repository name in `package.json`.
  * Added support for different policy file ports without having to port
    forward 843 on the server side [3rd-Eden]

0.8.2 / 2011-08-29
==================

  * Fixed flashsocket detection.

0.8.1 / 2011-08-29
==================

  * Bump version.

0.8.0 / 2011-08-28
==================

  * Added MozWebSocket support (hybi-10 doesn't require API changes) [einaros].

0.7.11 / 2011-08-27
===================

  * Corrected previous release (missing build).

0.7.10 / 2011-08-27
===================

  * Fix for failing fallback in websockets

0.7.9 / 2011-08-12
==================

  * Added check on `Socket#onConnect` to prevent double `connect` events on the main manager.
  * Fixed socket namespace connect test. Remove broken alternative namespace connect test.
  * Removed test handler for removed test.
  * Bumped version to match `socket.io` server.

0.7.5 / 2011-08-08
==================

  * Added querystring support for `connect` [3rd-Eden]
  * Added partial Node.JS transports support [3rd-Eden, josephg]
  * Fixed builder test.
  * Changed `util.inherit` to replicate Object.create / __proto__.
  * Changed and cleaned up some acceptance tests.
  * Fixed race condition with a test that could not be run multiple times.
  * Added test for encoding a payload.
  * Added the ability to override the transport to use in acceptance test [3rd-Eden]
  * Fixed multiple connect packets [DanielBaulig]
  * Fixed jsonp-polling over-buffering [3rd-Eden]
  * Fixed ascii preservation in minified socket.io client [3rd-Eden]
  * Fixed socket.io in situations where the page is not served through utf8.
  * Fixed namespaces not reconnecting after disconnect [3rd-Eden]
  * Fixed default port for secure connections.

0.7.4 / 2011-07-12
==================

  * Added `SocketNamespace#of` shortcut. [3rd-Eden]
  * Fixed a IE payload decoding bug. [3rd-Eden]
  * Honor document protocol, unless overriden. [dvv]
  * Fixed new builder dependencies. [3rd-Eden]

0.7.3 / 2011-06-30 
==================

  * Fixed; acks don't depend on arity. They're automatic for `.send` and
    callback based for `.emit`. [dvv]
  * Added support for sub-sockets authorization. [3rd-Eden]
  * Added BC support for `new io.connect`. [fat]
  * Fixed double `connect` events. [3rd-Eden]
  * Fixed reconnection with jsonp-polling maintaining old sessionid. [franck34]

0.7.2 / 2011-06-22
==================

  * Added `noop` message type.

0.7.1 / 2011-06-21
==================

  * Bumped socket.io dependency version for acceptance tests.

0.7.0 / 2011-06-21
==================

  * http://socket.io/announcement.html

