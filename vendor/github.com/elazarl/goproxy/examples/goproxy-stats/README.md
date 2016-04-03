# Gather Browsing Statistics

`goproxy-stats` starts an HTTP proxy on :8080, counts the bytes received for
web resources and prints the cumulative sum per URL every 20 seconds.

Start it in one shell:

```sh
goproxy-stats
```

Fetch goproxy homepage in another:

```sh
mkdir tmp
cd tmp
http_proxy=http://127.0.0.1:8080 wget -r -l 1 -H \
	http://ripper234.com/p/introducing-goproxy-light-http-proxy/
```

Stop it after a moment. `goproxy-stats` should eventually print:
```sh
listening on :8080
statistics
http://www.telerik.com/fiddler -> 84335
http://msmvps.com/robots.txt -> 157
http://eli.thegreenplace.net/robots.txt -> 294
http://www.phdcomics.com/robots.txt -> 211
http://resharper.blogspot.com/robots.txt -> 221
http://idanz.blogli.co.il/robots.txt -> 271
http://ripper234.com/p/introducing-goproxy-light-http-proxy/ -> 44407
http://live.gnome.org/robots.txt -> 298
http://ponetium.wordpress.com/robots.txt -> 178
http://pilaheleg.blogli.co.il/robots.txt -> 321
http://pilaheleg.wordpress.com/robots.txt -> 178
http://blogli.co.il/ -> 9165
http://nimrod-code.org/robots.txt -> 289
http://www.joelonsoftware.com/robots.txt -> 1245
http://top-performance.blogspot.com/robots.txt -> 227
http://ooc-lang.org/robots.txt -> 345
http://blogs.jetbrains.com/robots.txt -> 293
```

