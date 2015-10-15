# Simple HTTP Proxy

`goproxy-basic` starts an HTTP proxy on :8080. It only handles explicit CONNECT
requests.

Start it in one shell:

```sh
goproxy-basic -v
```

Fetch goproxy homepage in another:

```sh
http_proxy=http://127.0.0.1:8080 wget -O - \
	http://ripper234.com/p/introducing-goproxy-light-http-proxy/
```

The homepage HTML content should be displayed in the console. The proxy should
have logged the request being processed:

```sh
2015/04/09 18:19:17 [001] INFO: Got request /p/introducing-goproxy-light-http-proxy/ ripper234.com GET http://ripper234.com/p/introducing-goproxy-light-http-proxy/
2015/04/09 18:19:17 [001] INFO: Sending request GET http://ripper234.com/p/introducing-goproxy-light-http-proxy/
2015/04/09 18:19:18 [001] INFO: Received response 200 OK
2015/04/09 18:19:18 [001] INFO: Copying response to client 200 OK [200]
2015/04/09 18:19:18 [001] INFO: Copied 44333 bytes to client error=<nil>
```

