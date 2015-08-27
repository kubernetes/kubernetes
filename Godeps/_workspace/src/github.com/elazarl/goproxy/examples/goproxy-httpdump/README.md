# Trace HTTP Requests and Responses

`goproxy-httpdump` starts an HTTP proxy on :8080. It handles explicit CONNECT
requests and traces them in a "db" directory created in the proxy working
directory.  Each request type and headers are logged in a "log" file, while
their bodies are dumped in files prefixed with the request session identifier.

Additionally, the example demonstrates how to:
- Log information asynchronously (see HttpLogger)
- Allow the proxy to be stopped manually while ensuring all pending requests
  have been processed (in this case, logged).

Start it in one shell:

```sh
goproxy-httpdump
```

Fetch goproxy homepage in another:

```sh
http_proxy=http://127.0.0.1:8080 wget -O - \
	http://ripper234.com/p/introducing-goproxy-light-http-proxy/
```

A "db" directory should have appeared where you started the proxy, containing
two files:
- log: the request/response traces
- 1\_resp: the first response body

