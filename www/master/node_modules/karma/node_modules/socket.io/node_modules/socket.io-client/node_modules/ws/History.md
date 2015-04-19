v0.4.31 - September 23th, 2013
=====================

* Component support

v0.4.30 - August 30th, 2013
=====================

* BufferedAmount could be undefined, default to 0 [TooTallNate]
* Support protocols as second argument and options as third [TooTallNate]
* Proper browserify shim [mcollina]
* Broadcasting example in README [stefanocudini]

v0.4.29 - August 23th, 2013
=====================
* Small clean up of the Node 0.11 support by using NAN from the NPM registry [kkoopa]
* Support for custom `Agent`'s through the options. [gramakri] & [TooTallNate]
* Support for custom headers through the options [3rd-Eden]
* Added a `gypfile` flag to the package.json for compiled module discovery [wolfeidau]

v0.4.28 - August 16th, 2013
=====================
* Node 0.11 support. [kkoopa]
* Authorization headers are sent when basic auth is used in the url [jcrugzz]
* Origin header will now include the port number [Jason Plum]
* Race condition fixed where data was received before the readyState was updated. [saschagehlich]

v0.4.27 - June 27th, 2013
=====================
* Frames are no longer masked in `wscat`. [slaskis]
* Don't retrain reference to large slab buffers. [jmatthewsr-msi]
* Don't use Buffer.byteLength for ArrayBuffer's. [Anthony Pesch]
* Fix browser field in package.json. [shtylman]
* Client-side certificate support & documentation improvements. [Lukas Berns]
* WebSocket readyState's is added to the prototype for spec compatiblity. [BallBearing]
* Use Object.defineProperty. [arlolra]
* Autodetect ArrayBuffers as binary when sending. [BallBearing]
* Check instanceof Buffer for binary data. [arlolra]
* Emit the close event before destroying the internal socket. [3rd-Eden]
* Don't setup multiply timeouts for one connection. [AndreasMadsen]
* Allow support for binding to ethereal port. [wpreul]
* Fix broken terminate reference. [3rd-Eden]
* Misc node 0.10 test fixes and documentation improvements. [3rd-Eden]
* Ensure ssl options are propagated to request. [einaros]
* Add 'Host' and 'Origin' to request header. [Lars-Magnus Skog]
* Subprotocol support. [kanaka]
* Honor ArrayBufferView's byteOffset when sending. [Anthony Pesch]
* Added target attribute for events. [arlolra]

v0.4.26 - Skipped
=====================

v0.4.25 - December 17th, 2012
=====================
* Removed install.js. [shtylman]
* Added browser field to package.json. [shtylman]
* Support overwriting host header. [Raynos]
* Emit 'listening' also with custom http server. [sebiq]

v0.4.24 - December 6th, 2012
=====================
* Yet another intermediate release, to  not delay minor features any longer.
* Native support installation issues further circumvented. [einaros]

v0.4.23 - November 19th, 2012
=====================
* Service release - last before major upgrade.
* Changes default host from 127.0.0.1 to 0.0.0.0. [einaros]

v0.4.22 - October 3rd, 2012
=====================
* clear failsafe cleanup timeout once cleanup is called [AndreasMadsen]
* added w3c compatible CloseEvent for onclose / addEventListener("close", ...). [einaros]
* fix the sub protocol header handler [sonnyp]
* fix unhandled exception if socket closes and 'error' is emitted [jmatthewsr-ms]

v0.4.21 - July 14th, 2012
=====================
* Emit error if server reponds with anything other than status code 101. [einaros]
* Added 'headers' event to server. [rauchg]
* path.exists moved to fs.exists. [blakmatrix]

v0.4.20 - June 26th, 2012
=====================
* node v0.8.0 compatibility release.

v0.4.19 - June 19th, 2012
=====================
* Change sender to merge buffers for relatively small payloads, may improve perf in some cases [einaros]
* Avoid EventEmitter for Receiver classes. As above this may improve perf. [einaros]
* Renamed fallback files from the somewhat misleading '*Windows'. [einaros]

v0.4.18 - June 14th 2012
=====================
* Fixed incorrect md5 digest encoding in Hixie handshake [nicokaiser]
* Added example of use with Express 3 [einaros]
* Change installation procedure to not require --ws:native to build native extensions. They will now build if a compiler is available. [einaros]

v0.4.17 - June 13th 2012
=====================
* Improve error handling during connection handshaking [einaros]
* Ensure that errors are caught also after connection teardown [nicokaiser]
* Update 'mocha' version to 1.1.0. [einaros]
* Stop showing 'undefined' for some error logs. [tricknotes]
* Update 'should' version to 0.6.3 [tricknotes]

v0.4.16 - June 1st 2012
=====================
* Build fix for Windows. [einaros]

v0.4.15 - May 20th 2012
=====================
* Enable fauxe streaming for hixie tansport. [einaros]
* Allow hixie sender to deal with buffers. [einaros/pigne]
* Allow error code 1011. [einaros]
* Fix framing for empty packets (empty pings and pongs might break). [einaros]
* Improve error and close handling, to avoid connections lingering in CLOSING state. [einaros]

v0.4.14 - Apr 30th 2012
=====================
* use node-gyp instead of node-waf [TooTallNate]
* remove old windows compatibility makefile, and silently fall back to native modules [einaros]
* ensure connection status [nicokaiser]
* websocket client updated to use port 443 by default for wss:// connections [einaros]
* support unix sockets [kschzt]

v0.4.13 - Apr 12th 2012
=====================

* circumvent node 0.6+ related memory leak caused by Object.defineProperty [nicokaiser]
* improved error handling, improving stability in massive load use cases [nicokaiser]

v0.4.12 - Mar 30th 2012
=====================

* various memory leak / possible memory leak cleanups [einaros]
* api documentation [nicokaiser]
* add option to disable client tracking [nicokaiser]

v0.4.11 - Mar 24th 2012
=====================

* node v0.7 compatibillity release
* gyp support [TooTallNate]
* commander dependency update [jwueller]
* loadbalancer support [nicokaiser]

v0.4.10 - Mar 22th 2012
=====================

* Final hixie close frame fixes. [nicokaiser]

v0.4.9 - Mar 21st 2012
=====================

* Various hixie bugfixes (such as proper close frame handling). [einaros]

v0.4.8 - Feb 29th 2012
=====================

* Allow verifyClient to run asynchronously [karlsequin]
* Various bugfixes and cleanups. [einaros]

v0.4.7 - Feb 21st 2012
=====================

* Exposed bytesReceived from websocket client object, which makes it possible to implement bandwidth sampling. [einaros]
* Updated browser based file upload example to include and output per websocket channel bandwidth sampling. [einaros]
* Changed build scripts to check which architecture is currently in use. Required after the node.js changes to have prebuilt packages target ia32 by default. [einaros]

v0.4.6 - Feb 9th 2012
=====================

* Added browser based file upload example. [einaros]
* Added server-to-browser status push example. [einaros]
* Exposed pause() and resume() on WebSocket object, to enable client stream shaping. [einaros]

v0.4.5 - Feb 7th 2012
=====================

* Corrected regression bug in handling of connections with the initial frame delivered across both http upgrade head and a standalone packet. This would lead to a race condition, which in some cases could cause message corruption. [einaros]

v0.4.4 - Feb 6th 2012
=====================

* Pass original request object to verifyClient, for cookie or authentication verifications. [einaros]
* Implemented addEventListener and slightly improved the emulation API by adding a MessageEvent with a readonly data attribute. [aslakhellesoy]
* Rewrite parts of hybi receiver to avoid stack overflows for large amounts of packets bundled in the same buffer / packet. [einaros]

v0.4.3 - Feb 4th 2012
=====================

* Prioritized update: Corrected issue which would cause sockets to stay open longer than necessary, and resource leakage because of this. [einaros]

v0.4.2 - Feb 4th 2012
=====================

* Breaking change: WebSocketServer's verifyOrigin option has been renamed to verifyClient. [einaros]
* verifyClient now receives { origin: 'origin header', secure: true/false }, where 'secure' will be true for ssl connections. [einaros]
* Split benchmark, in preparation for more thorough case. [einaros]
* Introduced hixie-76 draft support for server, since Safari (iPhone / iPad / OS X) and Opera still aren't updated to use Hybi. [einaros]
* Expose 'supports' object from WebSocket, to indicate e.g. the underlying transport's support for binary data. [einaros]
* Test and code cleanups. [einaros]

v0.4.1 - Jan 25th 2012
=====================

* Use readline in wscat [tricknotes]
* Refactor _state away, in favor of the new _readyState [tricknotes]
* travis-ci integration [einaros]
* Fixed race condition in testsuite, causing a few tests to fail (without actually indicating errors) on travis [einaros]
* Expose pong event [paddybyers]
* Enabled running of WebSocketServer in noServer-mode, meaning that upgrades are passed in manually. [einaros]
* Reworked connection procedure for WebSocketServer, and cleaned up tests. [einaros]

v0.4.0 - Jan 2nd 2012
=====================

* Windows compatibility [einaros]
* Windows compatible test script [einaros]

v0.3.9 - Jan 1st 2012
======================

* Improved protocol framing performance [einaros]
* WSS support [kazuyukitanimura]
* WSS tests [einaros]
* readyState exposed [justinlatimer, tricknotes]
* url property exposed [justinlatimer]
* Removed old 'state' property [einaros]
* Test cleanups [einaros]

v0.3.8 - Dec 27th 2011
======================

* Made it possible to listen on specific paths, which is especially good to have for precreated http servers [einaros]
* Extensive WebSocket / WebSocketServer cleanup, including changing all internal properties to unconfigurable, unenumerable properties [einaros]
* Receiver modifications to ensure even better performance with fragmented sends [einaros]
* Fixed issue in sender.js, which would cause SlowBuffer instances (such as returned from the crypto library's randomBytes) to be copied (and thus be dead slow) [einaros]
* Removed redundant buffer copy in sender.js, which should improve server performance [einaros]

v0.3.7 - Dec 25nd 2011
======================

* Added a browser based API which uses EventEmitters internally [3rd-Eden]
* Expose request information from upgrade event for websocket server clients [mmalecki]

v0.3.6 - Dec 19th 2011
======================

* Added option to let WebSocket.Server use an already existing http server [mmalecki]
* Migrating various option structures to use options.js module [einaros]
* Added a few more tests, options and handshake verifications to ensure that faulty connections are dealt with [einaros]
* Code cleanups in Sender and Receiver, to ensure even faster parsing [einaros]

v0.3.5 - Dec 13th 2011
======================

* Optimized Sender.js, Receiver.js and bufferutil.cc:
 * Apply loop-unrolling-like small block copies rather than use node.js Buffer#copy() (which is slow).
 * Mask blocks of data using combination of 32bit xor and loop-unrolling, instead of single bytes.
 * Keep pre-made send buffer for small transfers.
* Leak fixes and code cleanups.

v0.3.3 - Dec 12th 2011
======================

* Compile fix for Linux.
* Rewrote parts of WebSocket.js, to avoid try/catch and thus avoid optimizer bailouts.

v0.3.2 - Dec 11th 2011
======================

* Further performance updates, including the additions of a native BufferUtil module, which deals with several of the cpu intensive WebSocket operations.

v0.3.1 - Dec 8th 2011
======================

* Service release, fixing broken tests.

v0.3.0 - Dec 8th 2011
======================

* Node.js v0.4.x compatibility.
* Code cleanups and efficiency improvements.
* WebSocket server added, although this will still mainly be a client library.
* WebSocket server certified to pass the Autobahn test suite.
* Protocol improvements and corrections - such as handling (redundant) masks for empty fragments.
* 'wscat' command line utility added, which can act as either client or server.

v0.2.6 - Dec 3rd 2011
======================

* Renamed to 'ws'. Big woop, right -- but easy-websocket really just doesn't cut it anymore!

v0.2.5 - Dec 3rd 2011
======================

  * Rewrote much of the WebSocket parser, to ensure high speed for highly fragmented messages.
  * Added a BufferPool, as a start to more efficiently deal with allocations for WebSocket connections. More work to come, in that area.
  * Updated the Autobahn report, at http://einaros.github.com/easy-websocket, with comparisons against WebSocket-Node 1.0.2 and Chrome 16.

v0.2.0 - Nov 25th 2011
======================

  * Major rework to make sure all the Autobahn test cases pass. Also updated the internal tests to cover more corner cases.

v0.1.2 - Nov 14th 2011
======================

  * Back and forth, back and forth: now settled on keeping the api (event names, methods) closer to the websocket browser api. This will stick now.
  * Started keeping this history record. Better late than never, right?
