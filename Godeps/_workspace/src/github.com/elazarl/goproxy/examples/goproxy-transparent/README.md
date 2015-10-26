# Transparent Proxy

This transparent example in goproxy is meant to show how to transparenty proxy and hijack all http and https connections while doing a man-in-the-middle to the TLS session.  It requires that goproxy sees all the packets traversing out to the internet.  Linux iptables rules deal with changing the source/destination IPs to act transparently, but you do need to setup your network configuration so that goproxy is a mandatory stop on the outgoing route.  Primarily you can do this by placing the proxy inline.  goproxy does not have any WCCP support itself; patches welcome.

## Why not explicit?

Transparent proxies are more difficult to maintain and setup from a server side, but they require no configuration on the client(s) which could be in unmanaged systems or systems that don't support a proxy configuration.  See the [eavesdropper example](https://github.com/elazarl/goproxy/blob/master/examples/goproxy-eavesdropper/main.go) if you want to see an explicit proxy example.

## Potential Issues

Support for very old clients using HTTPS will fail.  Clients need to send the SNI value in the TLS ClientHello which most modern clients do these days, but old clients will break.

If you're routing table allows for it, an explicit http request to goproxy will cause it to fail in an endless loop since it will try to request resources from itself repeatedly.  This could be solved in the goproxy code by looking up the hostnames, but it adds a delay that is much easier/faster to handle on the routing side.

## Routing Rules

Example routing rules are included in [proxy.sh](https://github.com/elazarl/goproxy/blob/master/examples/goproxy-transparent/proxy.sh) but are best when setup using your distribution's configuration.
