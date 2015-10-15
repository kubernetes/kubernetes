package goproxy_html_test

import (
	"github.com/elazarl/goproxy"
	"github.com/elazarl/goproxy/ext/html"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"
)

type ConstantServer int

func (s ConstantServer) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "text/plain; charset=iso-8859-8")
	//w.Header().Set("Content-Type","text/plain; charset=cp-1255")
	w.Write([]byte{0xe3, 0xf3})
}

func TestCharset(t *testing.T) {
	s := httptest.NewServer(ConstantServer(1))
	defer s.Close()

	ch := make(chan string, 2)
	proxy := goproxy.NewProxyHttpServer()
	proxy.OnResponse().Do(goproxy_html.HandleString(
		func(s string, ctx *goproxy.ProxyCtx) string {
			ch <- s
			return s
		}))
	proxyServer := httptest.NewServer(proxy)
	defer proxyServer.Close()

	proxyUrl, _ := url.Parse(proxyServer.URL)
	client := &http.Client{Transport: &http.Transport{Proxy: http.ProxyURL(proxyUrl)}}

	resp, err := client.Get(s.URL + "/cp1255.txt")
	if err != nil {
		t.Fatal("GET:", err)
	}
	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		t.Fatal("readAll:", err)
	}
	resp.Body.Close()

	inHandleString := ""
	select {
	case inHandleString = <-ch:
	default:
	}

	if len(b) != 2 || b[0] != 0xe3 || b[1] != 0xf3 {
		t.Error("Did not translate back to 0xe3,0xf3, instead", b)
	}
	if inHandleString != "דף" {
		t.Error("HandleString did not convert DALET & PEH SOFIT (דף) from ISO-8859-8 to utf-8, got", []byte(inHandleString))
	}
}
