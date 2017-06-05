// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build h2demo

package main

import (
	"bytes"
	"crypto/tls"
	"flag"
	"fmt"
	"hash/crc32"
	"image"
	"image/jpeg"
	"io"
	"io/ioutil"
	"log"
	"net"
	"net/http"
	"path"
	"regexp"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"camlistore.org/pkg/googlestorage"
	"go4.org/syncutil/singleflight"
	"golang.org/x/net/http2"
)

var (
	prod = flag.Bool("prod", false, "Whether to configure itself to be the production http2.golang.org server.")

	httpsAddr = flag.String("https_addr", "localhost:4430", "TLS address to listen on ('host:port' or ':port'). Required.")
	httpAddr  = flag.String("http_addr", "", "Plain HTTP address to listen on ('host:port', or ':port'). Empty means no HTTP.")

	hostHTTP  = flag.String("http_host", "", "Optional host or host:port to use for http:// links to this service. By default, this is implied from -http_addr.")
	hostHTTPS = flag.String("https_host", "", "Optional host or host:port to use for http:// links to this service. By default, this is implied from -https_addr.")
)

func homeOldHTTP(w http.ResponseWriter, r *http.Request) {
	io.WriteString(w, `<html>
<body>
<h1>Go + HTTP/2</h1>
<p>Welcome to <a href="https://golang.org/">the Go language</a>'s <a href="https://http2.github.io/">HTTP/2</a> demo & interop server.</p>
<p>Unfortunately, you're <b>not</b> using HTTP/2 right now. To do so:</p>
<ul>
   <li>Use Firefox Nightly or go to <b>about:config</b> and enable "network.http.spdy.enabled.http2draft"</li>
   <li>Use Google Chrome Canary and/or go to <b>chrome://flags/#enable-spdy4</b> to <i>Enable SPDY/4</i> (Chrome's name for HTTP/2)</li>
</ul>
<p>See code & instructions for connecting at <a href="https://github.com/golang/net/tree/master/http2">https://github.com/golang/net/tree/master/http2</a>.</p>

</body></html>`)
}

func home(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	io.WriteString(w, `<html>
<body>
<h1>Go + HTTP/2</h1>

<p>Welcome to <a href="https://golang.org/">the Go language</a>'s <a
href="https://http2.github.io/">HTTP/2</a> demo & interop server.</p>

<p>Congratulations, <b>you're using HTTP/2 right now</b>.</p>

<p>This server exists for others in the HTTP/2 community to test their HTTP/2 client implementations and point out flaws in our server.</p>

<p>
The code is at <a href="https://golang.org/x/net/http2">golang.org/x/net/http2</a> and
is used transparently by the Go standard library from Go 1.6 and later.
</p>

<p>Contact info: <i>bradfitz@golang.org</i>, or <a
href="https://golang.org/s/http2bug">file a bug</a>.</p>

<h2>Handlers for testing</h2>
<ul>
  <li>GET <a href="/reqinfo">/reqinfo</a> to dump the request + headers received</li>
  <li>GET <a href="/clockstream">/clockstream</a> streams the current time every second</li>
  <li>GET <a href="/gophertiles">/gophertiles</a> to see a page with a bunch of images</li>
  <li>GET <a href="/file/gopher.png">/file/gopher.png</a> for a small file (does If-Modified-Since, Content-Range, etc)</li>
  <li>GET <a href="/file/go.src.tar.gz">/file/go.src.tar.gz</a> for a larger file (~10 MB)</li>
  <li>GET <a href="/redirect">/redirect</a> to redirect back to / (this page)</li>
  <li>GET <a href="/goroutines">/goroutines</a> to see all active goroutines in this server</li>
  <li>PUT something to <a href="/crc32">/crc32</a> to get a count of number of bytes and its CRC-32</li>
  <li>PUT something to <a href="/ECHO">/ECHO</a> and it will be streamed back to you capitalized</li>
</ul>

</body></html>`)
}

func reqInfoHandler(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain")
	fmt.Fprintf(w, "Method: %s\n", r.Method)
	fmt.Fprintf(w, "Protocol: %s\n", r.Proto)
	fmt.Fprintf(w, "Host: %s\n", r.Host)
	fmt.Fprintf(w, "RemoteAddr: %s\n", r.RemoteAddr)
	fmt.Fprintf(w, "RequestURI: %q\n", r.RequestURI)
	fmt.Fprintf(w, "URL: %#v\n", r.URL)
	fmt.Fprintf(w, "Body.ContentLength: %d (-1 means unknown)\n", r.ContentLength)
	fmt.Fprintf(w, "Close: %v (relevant for HTTP/1 only)\n", r.Close)
	fmt.Fprintf(w, "TLS: %#v\n", r.TLS)
	fmt.Fprintf(w, "\nHeaders:\n")
	r.Header.Write(w)
}

func crcHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "PUT" {
		http.Error(w, "PUT required.", 400)
		return
	}
	crc := crc32.NewIEEE()
	n, err := io.Copy(crc, r.Body)
	if err == nil {
		w.Header().Set("Content-Type", "text/plain")
		fmt.Fprintf(w, "bytes=%d, CRC32=%x", n, crc.Sum(nil))
	}
}

type capitalizeReader struct {
	r io.Reader
}

func (cr capitalizeReader) Read(p []byte) (n int, err error) {
	n, err = cr.r.Read(p)
	for i, b := range p[:n] {
		if b >= 'a' && b <= 'z' {
			p[i] = b - ('a' - 'A')
		}
	}
	return
}

type flushWriter struct {
	w io.Writer
}

func (fw flushWriter) Write(p []byte) (n int, err error) {
	n, err = fw.w.Write(p)
	if f, ok := fw.w.(http.Flusher); ok {
		f.Flush()
	}
	return
}

func echoCapitalHandler(w http.ResponseWriter, r *http.Request) {
	if r.Method != "PUT" {
		http.Error(w, "PUT required.", 400)
		return
	}
	io.Copy(flushWriter{w}, capitalizeReader{r.Body})
}

var (
	fsGrp   singleflight.Group
	fsMu    sync.Mutex // guards fsCache
	fsCache = map[string]http.Handler{}
)

// fileServer returns a file-serving handler that proxies URL.
// It lazily fetches URL on the first access and caches its contents forever.
func fileServer(url string) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		hi, err := fsGrp.Do(url, func() (interface{}, error) {
			fsMu.Lock()
			if h, ok := fsCache[url]; ok {
				fsMu.Unlock()
				return h, nil
			}
			fsMu.Unlock()

			res, err := http.Get(url)
			if err != nil {
				return nil, err
			}
			defer res.Body.Close()
			slurp, err := ioutil.ReadAll(res.Body)
			if err != nil {
				return nil, err
			}

			modTime := time.Now()
			var h http.Handler = http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				http.ServeContent(w, r, path.Base(url), modTime, bytes.NewReader(slurp))
			})
			fsMu.Lock()
			fsCache[url] = h
			fsMu.Unlock()
			return h, nil
		})
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		hi.(http.Handler).ServeHTTP(w, r)
	})
}

func clockStreamHandler(w http.ResponseWriter, r *http.Request) {
	clientGone := w.(http.CloseNotifier).CloseNotify()
	w.Header().Set("Content-Type", "text/plain")
	ticker := time.NewTicker(1 * time.Second)
	defer ticker.Stop()
	fmt.Fprintf(w, "# ~1KB of junk to force browsers to start rendering immediately: \n")
	io.WriteString(w, strings.Repeat("# xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx\n", 13))

	for {
		fmt.Fprintf(w, "%v\n", time.Now())
		w.(http.Flusher).Flush()
		select {
		case <-ticker.C:
		case <-clientGone:
			log.Printf("Client %v disconnected from the clock", r.RemoteAddr)
			return
		}
	}
}

func registerHandlers() {
	tiles := newGopherTilesHandler()

	mux2 := http.NewServeMux()
	http.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.TLS == nil {
			if r.URL.Path == "/gophertiles" {
				tiles.ServeHTTP(w, r)
				return
			}
			http.Redirect(w, r, "https://"+httpsHost()+"/", http.StatusFound)
			return
		}
		if r.ProtoMajor == 1 {
			if r.URL.Path == "/reqinfo" {
				reqInfoHandler(w, r)
				return
			}
			homeOldHTTP(w, r)
			return
		}
		mux2.ServeHTTP(w, r)
	})
	mux2.HandleFunc("/", home)
	mux2.Handle("/file/gopher.png", fileServer("https://golang.org/doc/gopher/frontpage.png"))
	mux2.Handle("/file/go.src.tar.gz", fileServer("https://storage.googleapis.com/golang/go1.4.1.src.tar.gz"))
	mux2.HandleFunc("/reqinfo", reqInfoHandler)
	mux2.HandleFunc("/crc32", crcHandler)
	mux2.HandleFunc("/ECHO", echoCapitalHandler)
	mux2.HandleFunc("/clockstream", clockStreamHandler)
	mux2.Handle("/gophertiles", tiles)
	mux2.HandleFunc("/redirect", func(w http.ResponseWriter, r *http.Request) {
		http.Redirect(w, r, "/", http.StatusFound)
	})
	stripHomedir := regexp.MustCompile(`/(Users|home)/\w+`)
	mux2.HandleFunc("/goroutines", func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/plain; charset=utf-8")
		buf := make([]byte, 2<<20)
		w.Write(stripHomedir.ReplaceAll(buf[:runtime.Stack(buf, true)], nil))
	})
}

func newGopherTilesHandler() http.Handler {
	const gopherURL = "https://blog.golang.org/go-programming-language-turns-two_gophers.jpg"
	res, err := http.Get(gopherURL)
	if err != nil {
		log.Fatal(err)
	}
	if res.StatusCode != 200 {
		log.Fatalf("Error fetching %s: %v", gopherURL, res.Status)
	}
	slurp, err := ioutil.ReadAll(res.Body)
	res.Body.Close()
	if err != nil {
		log.Fatal(err)
	}
	im, err := jpeg.Decode(bytes.NewReader(slurp))
	if err != nil {
		if len(slurp) > 1024 {
			slurp = slurp[:1024]
		}
		log.Fatalf("Failed to decode gopher image: %v (got %q)", err, slurp)
	}

	type subImager interface {
		SubImage(image.Rectangle) image.Image
	}
	const tileSize = 32
	xt := im.Bounds().Max.X / tileSize
	yt := im.Bounds().Max.Y / tileSize
	var tile [][][]byte // y -> x -> jpeg bytes
	for yi := 0; yi < yt; yi++ {
		var row [][]byte
		for xi := 0; xi < xt; xi++ {
			si := im.(subImager).SubImage(image.Rectangle{
				Min: image.Point{xi * tileSize, yi * tileSize},
				Max: image.Point{(xi + 1) * tileSize, (yi + 1) * tileSize},
			})
			buf := new(bytes.Buffer)
			if err := jpeg.Encode(buf, si, &jpeg.Options{Quality: 90}); err != nil {
				log.Fatal(err)
			}
			row = append(row, buf.Bytes())
		}
		tile = append(tile, row)
	}
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		ms, _ := strconv.Atoi(r.FormValue("latency"))
		const nanosPerMilli = 1e6
		if r.FormValue("x") != "" {
			x, _ := strconv.Atoi(r.FormValue("x"))
			y, _ := strconv.Atoi(r.FormValue("y"))
			if ms <= 1000 {
				time.Sleep(time.Duration(ms) * nanosPerMilli)
			}
			if x >= 0 && x < xt && y >= 0 && y < yt {
				http.ServeContent(w, r, "", time.Time{}, bytes.NewReader(tile[y][x]))
				return
			}
		}
		io.WriteString(w, "<html><body onload='showtimes()'>")
		fmt.Fprintf(w, "A grid of %d tiled images is below. Compare:<p>", xt*yt)
		for _, ms := range []int{0, 30, 200, 1000} {
			d := time.Duration(ms) * nanosPerMilli
			fmt.Fprintf(w, "[<a href='https://%s/gophertiles?latency=%d'>HTTP/2, %v latency</a>] [<a href='http://%s/gophertiles?latency=%d'>HTTP/1, %v latency</a>]<br>\n",
				httpsHost(), ms, d,
				httpHost(), ms, d,
			)
		}
		io.WriteString(w, "<p>\n")
		cacheBust := time.Now().UnixNano()
		for y := 0; y < yt; y++ {
			for x := 0; x < xt; x++ {
				fmt.Fprintf(w, "<img width=%d height=%d src='/gophertiles?x=%d&y=%d&cachebust=%d&latency=%d'>",
					tileSize, tileSize, x, y, cacheBust, ms)
			}
			io.WriteString(w, "<br/>\n")
		}
		io.WriteString(w, `<p><div id='loadtimes'></div></p>
<script>
function showtimes() {
	var times = 'Times from connection start:<br>'
	times += 'DOM loaded: ' + (window.performance.timing.domContentLoadedEventEnd - window.performance.timing.connectStart) + 'ms<br>'
	times += 'DOM complete (images loaded): ' + (window.performance.timing.domComplete - window.performance.timing.connectStart) + 'ms<br>'
	document.getElementById('loadtimes').innerHTML = times
}
</script>
<hr><a href='/'>&lt;&lt Back to Go HTTP/2 demo server</a></body></html>`)
	})
}

func httpsHost() string {
	if *hostHTTPS != "" {
		return *hostHTTPS
	}
	if v := *httpsAddr; strings.HasPrefix(v, ":") {
		return "localhost" + v
	} else {
		return v
	}
}

func httpHost() string {
	if *hostHTTP != "" {
		return *hostHTTP
	}
	if v := *httpAddr; strings.HasPrefix(v, ":") {
		return "localhost" + v
	} else {
		return v
	}
}

func serveProdTLS() error {
	c, err := googlestorage.NewServiceClient()
	if err != nil {
		return err
	}
	slurp := func(key string) ([]byte, error) {
		const bucket = "http2-demo-server-tls"
		rc, _, err := c.GetObject(&googlestorage.Object{
			Bucket: bucket,
			Key:    key,
		})
		if err != nil {
			return nil, fmt.Errorf("Error fetching GCS object %q in bucket %q: %v", key, bucket, err)
		}
		defer rc.Close()
		return ioutil.ReadAll(rc)
	}
	certPem, err := slurp("http2.golang.org.chained.pem")
	if err != nil {
		return err
	}
	keyPem, err := slurp("http2.golang.org.key")
	if err != nil {
		return err
	}
	cert, err := tls.X509KeyPair(certPem, keyPem)
	if err != nil {
		return err
	}
	srv := &http.Server{
		TLSConfig: &tls.Config{
			Certificates: []tls.Certificate{cert},
		},
	}
	http2.ConfigureServer(srv, &http2.Server{})
	ln, err := net.Listen("tcp", ":443")
	if err != nil {
		return err
	}
	return srv.Serve(tls.NewListener(tcpKeepAliveListener{ln.(*net.TCPListener)}, srv.TLSConfig))
}

type tcpKeepAliveListener struct {
	*net.TCPListener
}

func (ln tcpKeepAliveListener) Accept() (c net.Conn, err error) {
	tc, err := ln.AcceptTCP()
	if err != nil {
		return
	}
	tc.SetKeepAlive(true)
	tc.SetKeepAlivePeriod(3 * time.Minute)
	return tc, nil
}

func serveProd() error {
	errc := make(chan error, 2)
	go func() { errc <- http.ListenAndServe(":80", nil) }()
	go func() { errc <- serveProdTLS() }()
	return <-errc
}

const idleTimeout = 5 * time.Minute
const activeTimeout = 10 * time.Minute

// TODO: put this into the standard library and actually send
// PING frames and GOAWAY, etc: golang.org/issue/14204
func idleTimeoutHook() func(net.Conn, http.ConnState) {
	var mu sync.Mutex
	m := map[net.Conn]*time.Timer{}
	return func(c net.Conn, cs http.ConnState) {
		mu.Lock()
		defer mu.Unlock()
		if t, ok := m[c]; ok {
			delete(m, c)
			t.Stop()
		}
		var d time.Duration
		switch cs {
		case http.StateNew, http.StateIdle:
			d = idleTimeout
		case http.StateActive:
			d = activeTimeout
		default:
			return
		}
		m[c] = time.AfterFunc(d, func() {
			log.Printf("closing idle conn %v after %v", c.RemoteAddr(), d)
			go c.Close()
		})
	}
}

func main() {
	var srv http.Server
	flag.BoolVar(&http2.VerboseLogs, "verbose", false, "Verbose HTTP/2 debugging.")
	flag.Parse()
	srv.Addr = *httpsAddr
	srv.ConnState = idleTimeoutHook()

	registerHandlers()

	if *prod {
		*hostHTTP = "http2.golang.org"
		*hostHTTPS = "http2.golang.org"
		log.Fatal(serveProd())
	}

	url := "https://" + httpsHost() + "/"
	log.Printf("Listening on " + url)
	http2.ConfigureServer(&srv, &http2.Server{})

	if *httpAddr != "" {
		go func() {
			log.Printf("Listening on http://" + httpHost() + "/ (for unencrypted HTTP/1)")
			log.Fatal(http.ListenAndServe(*httpAddr, nil))
		}()
	}

	go func() {
		log.Fatal(srv.ListenAndServeTLS("server.crt", "server.key"))
	}()
	select {}
}
