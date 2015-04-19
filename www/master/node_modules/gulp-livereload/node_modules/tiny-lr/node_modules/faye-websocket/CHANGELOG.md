### 0.7.3 / 2014-10-04

* Allow sockets to be closed when they are in any state other than `CLOSED`


### 0.7.2 / 2013-12-29

* Make sure the `close` event is emitted by clients on Node v0.10


### 0.7.1 / 2013-12-03

* Support the `maxLength` websocket-driver option
* Make the client emit `error` events on network errors


### 0.7.0 / 2013-09-09

* Allow the server to send custom headers with EventSource responses


### 0.6.1 / 2013-07-05

* Add `ca` option to the client for specifying certificate authorities
* Start the server driver asynchronously so that `onopen` handlers can be added


### 0.6.0 / 2013-05-12

* Add support for custom headers


### 0.5.0 / 2013-05-05

* Extract the protocol handlers into the `websocket-driver` library
* Support the Node streaming API


### 0.4.4 / 2013-02-14

* Emit the `close` event if TCP is closed before CLOSE frame is acked


### 0.4.3 / 2012-07-09

* Add `Connection: close` to EventSource response
* Handle situations where `request.socket` is undefined


### 0.4.2 / 2012-04-06

* Add WebSocket error code `1011`.
* Handle URLs with no path correctly by sending `GET /`


### 0.4.1 / 2012-02-26

* Treat anything other than a `Buffer` as a string when calling `send()`


### 0.4.0 / 2012-02-13

* Add `ping()` method to server-side `WebSocket` and `EventSource`
* Buffer `send()` calls until the draft-76 handshake is complete
* Fix HTTPS problems on Node 0.7


### 0.3.1 / 2012-01-16

* Call `setNoDelay(true)` on `net.Socket` objects to reduce latency


### 0.3.0 / 2012-01-13

* Add support for `EventSource` connections


### 0.2.0 / 2011-12-21

* Add support for `Sec-WebSocket-Protocol` negotiation
* Support `hixie-76` close frames and 75/76 ignored segments
* Improve performance of HyBi parsing/framing functions
* Decouple parsers from TCP and reduce write volume


### 0.1.2 / 2011-12-05

* Detect closed sockets on the server side when TCP connection breaks
* Make `hixie-76` sockets work through HAProxy


### 0.1.1 / 2011-11-30

* Fix `addEventListener()` interface methods


### 0.1.0 / 2011-11-27

* Initial release, based on WebSocket components from Faye
