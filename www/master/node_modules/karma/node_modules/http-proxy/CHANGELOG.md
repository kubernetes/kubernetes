## ChangeLog for: node-http-proxy

## Version 0.10.0 - 3/18/2013

- Breaking change: `proxyResponse` events are emitted on the `HttpProxy` or `RoutingProxy` instances as originally was intended in `0.9.x`.

## Version 0.9.1 - 3/9/2013

- Ensure that `webSocketProxyError` and `proxyError` both receive the error (indexzero).

## Version 0.9.0 - 3/9/2013
- Fix #276 Ensure response.headers.location is defined (indexzero)
- Fix #248 Make options immutable in RoutingProxy (indexzero)
- Fix #359 Do not modify the protocol in redirect request for external sites. (indexzero)
- Fix #373 Do not use "Transfer-Encoding: chunked" header for proxied DELETE requests with no "Content-Length" header. (indexzero)
- Fix #338 Set "content-length" header to "0" if it is not already set on DELETE requests. (indexzero)
- Updates to README.md and Examples (ramitos, jamie-stackhouse, oost, indexzero)
- Fixes to ProxyTable and Routing Proxy (adjohnson916, otavoijr)
- New API for ProxyTable (mikkel, tglines)
- Add `options.timeout` for specifying socket timeouts (pdoran)
- Improve bin/node-http-proxy (niallo)
- Don't emit `proxyError` twice (erasmospunk)
- Fix memory leaks in WebSocket proxying 
- Support UNIX Sockets (yosefd)
- Fix truncated chunked respones (jpetazzo)
- Allow upstream listeners to get `proxyResponse` (colinmollenhour)

## Version 0.8.1 - 6/5/2012
- Fix re-emitting of events in RoutingProxy                (coderarity)
- New load balancer and middleware examples                (marak)
- Docs updated including changelog                         (lot of gently people)

## Version 0.8.0 - 12/23/2011
- Improve support and tests for url segment routing        (maxogden)
- Fix aborting connections when request close              (c4milo)
- Avoid 'Transfer-Encoding' on HTTP/1.0 clients            (koichik).
- Support for Node.js 0.6.x                                (mmalecki)

## Version 0.7.3 - 10/4/2011
- Fix setting x-forwarded headers                          (jesusabdullah)
- Updated examples                                         (AvianFlu)

## Version 0.7.0 - 9/10/2011
- Handles to every throw-able resume() call                (isaacs)
- Updated tests, README and package.json                   (indexzero)
- Added HttpProxy.close() method                           (indexzero)

## Version 0.6.6 - 8/31/2011
- Add more examples                                        (dominictarr)
- Use of 'pkginfo'                                         (indexzero)
- Handle cases where res.write throws                      (isaacs)
- Handles to every throw-able res.end call                 (isaacs)

## Version 0.5.11 - 6/21/2011
- Add more examples with WebSockets                        (indexzero)
- Update the documentation                                 (indexzero)

## Version 0.5.7 - 5/19/2011
- Fix to README related to markup and fix some examples    (benatkin)
- Improve WebSockets handling                              (indexzero)
- Improve WebSockets tests                                 (indexzero)
- Improve https tests                                      (olauzon)
- Add devDependencies to package.json                      (olauzon)
- Add 'proxyError' event                                   (indexzero)
- Add 'x-forwarded-{port|proto}' headers support           (indexzero)
- Keep-Alive connection supported                          (indexzero)

## Version 0.5.0 - 4/15/2011
- Remove winston in favor of custom events                 (indexzero)
- Add x-forwarded-for Header                               (indexzero)
- Fix WebSocket support                                    (indexzero)
- Add tests / examples for WebSocket support               (indexzero)
- Update .proxyRequest() and .proxyWebSocketRequest() APIs (indexzero)
- Add HTTPS support                                        (indexzero)
- Add tests / examples for HTTPS support                   (indexzero)

## Version 0.4.1 - 3/20/2011
- Include missing dependency in package.json                                  (indexzero)

## Version 0.4.0 - 3/20/2011
- Update for node.js 0.4.0                                                    (indexzero)
- Remove pool dependency in favor of http.Agent                               (indexzero)
- Store buffered data using `.buffer()` instead of on the HttpProxy instance  (indexzero)
- Change the ProxyTable to be a lookup table instead of actively proxying     (indexzero)
- Allow for pure host-only matching in ProxyTable                             (indexzero)
- Use winston for logging                                                     (indexzero)
- Improve tests with async setup and more coverage                            (indexzero)
- Improve code documentation                                                  (indexzero)

### Version 0.3.1 - 11/22/2010
- Added node-http-proxy binary script                      (indexzero)
- Added experimental WebSocket support                     (indutny)
- Added forward proxy functionality                        (indexzero)
- Added proxy table for multiple target lookup             (indexzero)
- Simplified tests using helpers.js                        (indexzero)
- Fixed uncaughtException bug with invalid proxy target    (indutny)
- Added configurable logging for HttpProxy and ProxyTable  (indexzero) 