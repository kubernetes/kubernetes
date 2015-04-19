
0.2.0 / 2013-11-20
==================

  * index: use "proxy-agent"

0.1.0 / 2013-11-16
==================

  * index: implement an LRU cache to lower the creation of `http.Agent` instances
  * index: include the ":" when generating the cache URI
  * index: add support for calling the proxy function directly
  * index: cleanup and refactor to allow for an opts object to be used
  * test: add some basic tests for the .proxy() function

0.0.2 / 2013-11-15
==================

  * package: add "superagent" to `devDependencies`
  * only invoke `url.parse()` when the proxy is a string
  * README++

0.0.1 / 2013-07-11
==================

 * Initial release, currently supports:
   * `http:`
   * `https:`
   * `socks:` (version 4a)
