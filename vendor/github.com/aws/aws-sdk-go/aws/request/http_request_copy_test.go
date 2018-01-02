package request

import (
	"bytes"
	"io/ioutil"
	"net/http"
	"net/url"
	"sync"
	"testing"
)

func TestRequestCopyRace(t *testing.T) {
	origReq := &http.Request{URL: &url.URL{}, Header: http.Header{}}
	origReq.Header.Set("Header", "OrigValue")

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func() {
			req := copyHTTPRequest(origReq, ioutil.NopCloser(&bytes.Buffer{}))
			req.Header.Set("Header", "Value")
			go func() {
				req2 := copyHTTPRequest(req, ioutil.NopCloser(&bytes.Buffer{}))
				req2.Header.Add("Header", "Value2")
			}()
			_ = req.Header.Get("Header")
			wg.Done()
		}()
		_ = origReq.Header.Get("Header")
	}
	origReq.Header.Get("Header")

	wg.Wait()
}
