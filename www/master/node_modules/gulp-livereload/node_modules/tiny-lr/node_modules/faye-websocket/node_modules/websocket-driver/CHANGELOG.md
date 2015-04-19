### 0.5.4 / 2015-03-29

* Don't emit extra close frames if we receive a close frame after we already
  sent one
* Fail the connection when the driver receives an invalid
  `Sec-WebSocket-Extensions` header

### 0.5.3 / 2015-02-22

* Don't treat incoming data as WebSocket frames if a client driver is closed
  before receiving the server handshake

### 0.5.2 / 2015-02-19

* Fix compatibility with the HTTP parser on io.js
* Use `websocket-extensions` to make sure messages and close frames are kept in
  order
* Don't emit multiple `error` events

### 0.5.1 / 2014-12-18

* Don't allow drivers to be created with unrecognized options

### 0.5.0 / 2014-12-13

* Support protocol extensions via the websocket-extensions module

### 0.4.0 / 2014-11-08

* Support connection via HTTP proxies using `CONNECT`

### 0.3.6 / 2014-10-04

* It is now possible to call `close()` before `start()` and close the driver

### 0.3.5 / 2014-07-06

* Don't hold references to frame buffers after a message has been emitted
* Make sure that `protocol` and `version` are exposed properly by the TCP driver

### 0.3.4 / 2014-05-08

* Don't hold memory-leaking references to I/O buffers after they have been
  parsed

### 0.3.3 / 2014-04-24

* Correct the draft-76 status line reason phrase

### 0.3.2 / 2013-12-29

* Expand `maxLength` to cover sequences of continuation frames and
  `draft-{75,76}`
* Decrease default maximum frame buffer size to 64MB
* Stop parsing when the protocol enters a failure mode, to save CPU cycles

### 0.3.1 / 2013-12-03

* Add a `maxLength` option to limit allowed frame size
* Don't pre-allocate a message buffer until the whole frame has arrived
* Fix compatibility with Node v0.11 `HTTPParser`

### 0.3.0 / 2013-09-09

* Support client URLs with Basic Auth credentials

### 0.2.2 / 2013-07-05

* No functional changes, just updates to package.json

### 0.2.1 / 2013-05-17

* Export the isSecureRequest() method since faye-websocket relies on it
* Queue sent messages in the client's initial state

### 0.2.0 / 2013-05-12

* Add API for setting and reading headers
* Add Driver.server() method for getting a driver for TCP servers

### 0.1.0 / 2013-05-04

* First stable release
