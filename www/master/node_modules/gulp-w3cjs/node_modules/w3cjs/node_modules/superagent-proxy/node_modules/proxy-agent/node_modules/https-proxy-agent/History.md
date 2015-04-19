
0.3.5 / 2014-06-11
==================

  * package: update "debug" to v1.0.0

0.3.4 / 2014-04-09
==================

  * gitignore: ignore root level ?.js files
  * package: update outdated dependencies

0.3.3 / 2014-01-13
==================

  * https-proxy-agnet: use debug() instead of console.error()
  * https-proxy-agent: fix debug() call
  * History: fix whitespace

0.3.2 / 2013-11-18
==================

  * https-proxy-agent: allow "https" without trailing colon
  * README: fix typo

0.3.1 / 2013-11-16
==================

  * test: enable the HTTPS over HTTPS test on node v0.11.8
  * https-proxy-agent: create the proxy socket connection first
  * https-proxy-agent: delete `pathname` from the proxy opts as well
  * https-proxy-agent: remove dead "end"-emitting code

0.3.0 / 2013-09-16
==================

  * https-proxy-agent: use "debug" module
  * https-proxy-agent: update to the "agent-base" v1 API
  * https-proxy-agent: default the "port" to 443 if not set
  * https-proxy-agent: augment the `opts` object for the `tls.connect` function
  * https-proxy-agent: use "extend" module
  * https-proxy-agent: remove use of `this` as much as possible
  * https-proxy-agent: listen for the "error" event of the socket
  * test: refactor of tests to use "proxy" module
  * test: add "error" event catching test
  * test: add 407 proxy response test
  * test: use "semver" module, disable the HTTPS over HTTPS test for node >= v0.11.3

0.2.0 / 2013-09-03
==================

  * Add initial "Proxy-Authorization" Basic authentication support

0.1.0 / 2013-07-21
==================

  * rename `secure` to `secureProxy`
  * added `secureEndpoint` option
  * various optimizations
  * README improvements

0.0.2 / 2013-07-11
==================

  * test: add mocha tests
  * don't use `socket.ondata`, use the official API instead
  * throw an Error when no proxy info is given
  * add support for passing options to net/tls .connect()

0.0.1 / 2013-07-09
==================

  * Initial release
