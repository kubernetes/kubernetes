package main

import (
	"bytes"
	"io/ioutil"
	"log"
	"net/http"
	"net/http/httptest"
	"net/url"
	"strings"
	"testing"
)

func equal(u, v []string) bool {
	if len(u) != len(v) {
		return false
	}
	for i, _ := range u {
		if u[i] != v[i] {
			return false
		}
	}
	return true
}

func readFile(fname string, t *testing.T) string {
	b, err := ioutil.ReadFile(fname)
	if err != nil {
		t.Fatal("readFile", err)
	}
	return string(b)
}

func TestDefectiveScriptParser(t *testing.T) {
	if l := len(findScriptSrc(`<!DOCTYPE HTML>
    <html>
    <body>

    <video width="320" height="240" controls="controls">
      <source src="movie.mp4" type="video/mp4" />
	<source src="movie.ogg" type="video/ogg" />
	  <source src="movie.webm" type="video/webm" />
	  Your browser does not support the video tag.
	  </video>

	  </body>
	  </html>`)); l != 0 {
		t.Fail()
	}
	urls := findScriptSrc(readFile("w3schools.html", t))
	if !equal(urls, []string{"http://partner.googleadservices.com/gampad/google_service.js",
		"//translate.google.com/translate_a/element.js?cb=googleTranslateElementInit"}) {
		t.Error("w3schools.html", "src scripts are not recognized", urls)
	}
	urls = findScriptSrc(readFile("jquery_homepage.html", t))
	if !equal(urls, []string{"http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js",
		"http://code.jquery.com/jquery-1.4.2.min.js",
		"http://static.jquery.com/files/rocker/scripts/custom.js",
		"http://static.jquery.com/donate/donate.js"}) {
		t.Error("jquery_homepage.html", "src scripts are not recognized", urls)
	}
}

func proxyWithLog() (*http.Client, *bytes.Buffer) {
	proxy := NewJqueryVersionProxy()
	proxyServer := httptest.NewServer(proxy)
	buf := new(bytes.Buffer)
	proxy.Logger = log.New(buf, "", 0)
	proxyUrl, _ := url.Parse(proxyServer.URL)
	tr := &http.Transport{Proxy: http.ProxyURL(proxyUrl)}
	client := &http.Client{Transport: tr}
	return client, buf
}

func get(t *testing.T, server *httptest.Server, client *http.Client, url string) {
	resp, err := client.Get(server.URL + url)
	if err != nil {
		t.Fatal("cannot get proxy", err)
	}
	ioutil.ReadAll(resp.Body)
	resp.Body.Close()
}

func TestProxyServiceTwoVersions(t *testing.T) {
	var fs = httptest.NewServer(http.FileServer(http.Dir(".")))
	defer fs.Close()

	client, buf := proxyWithLog()

	get(t, fs, client, "/w3schools.html")
	get(t, fs, client, "/php_man.html")
	if buf.String() != "" &&
		!strings.Contains(buf.String(), " uses jquery ") {
		t.Error("shouldn't warn on a single URL", buf.String())
	}
	get(t, fs, client, "/jquery1.html")
	warnings := buf.String()
	if !strings.Contains(warnings, "http://ajax.googleapis.com/ajax/libs/jquery/1.3.2/jquery.min.js") ||
		!strings.Contains(warnings, "jquery.1.4.js") ||
		!strings.Contains(warnings, "Contradicting") {
		t.Error("contradicting jquery versions (php_man.html, w3schools.html) does not issue warning", warnings)
	}
}

func TestProxyService(t *testing.T) {
	var fs = httptest.NewServer(http.FileServer(http.Dir(".")))
	defer fs.Close()

	client, buf := proxyWithLog()

	get(t, fs, client, "/jquery_homepage.html")
	warnings := buf.String()
	if !strings.Contains(warnings, "http://ajax.googleapis.com/ajax/libs/jquery/1.4.2/jquery.min.js") ||
		!strings.Contains(warnings, "http://code.jquery.com/jquery-1.4.2.min.js") ||
		!strings.Contains(warnings, "Contradicting") {
		t.Error("contradicting jquery versions does not issue warning")
	}
}
