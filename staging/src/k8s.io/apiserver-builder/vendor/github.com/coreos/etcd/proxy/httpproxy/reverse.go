// Copyright 2015 The etcd Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package httpproxy

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strings"
	"sync/atomic"

	"time"

	"github.com/coreos/etcd/etcdserver/api/v2http/httptypes"
	"github.com/coreos/etcd/pkg/httputil"
	"github.com/coreos/pkg/capnslog"
)

var (
	plog = capnslog.NewPackageLogger("github.com/coreos/etcd/proxy", "httpproxy")

	// Hop-by-hop headers. These are removed when sent to the backend.
	// http://www.w3.org/Protocols/rfc2616/rfc2616-sec13.html
	// This list of headers borrowed from stdlib httputil.ReverseProxy
	singleHopHeaders = []string{
		"Connection",
		"Keep-Alive",
		"Proxy-Authenticate",
		"Proxy-Authorization",
		"Te", // canonicalized version of "TE"
		"Trailers",
		"Transfer-Encoding",
		"Upgrade",
	}
)

func removeSingleHopHeaders(hdrs *http.Header) {
	for _, h := range singleHopHeaders {
		hdrs.Del(h)
	}
}

type reverseProxy struct {
	director  *director
	transport http.RoundTripper
}

func (p *reverseProxy) ServeHTTP(rw http.ResponseWriter, clientreq *http.Request) {
	reportIncomingRequest(clientreq)
	proxyreq := new(http.Request)
	*proxyreq = *clientreq
	startTime := time.Now()

	var (
		proxybody []byte
		err       error
	)

	if clientreq.Body != nil {
		proxybody, err = ioutil.ReadAll(clientreq.Body)
		if err != nil {
			msg := fmt.Sprintf("failed to read request body: %v", err)
			plog.Println(msg)
			e := httptypes.NewHTTPError(http.StatusInternalServerError, "httpproxy: "+msg)
			if we := e.WriteTo(rw); we != nil {
				plog.Debugf("error writing HTTPError (%v) to %s", we, clientreq.RemoteAddr)
			}
			return
		}
	}

	// deep-copy the headers, as these will be modified below
	proxyreq.Header = make(http.Header)
	copyHeader(proxyreq.Header, clientreq.Header)

	normalizeRequest(proxyreq)
	removeSingleHopHeaders(&proxyreq.Header)
	maybeSetForwardedFor(proxyreq)

	endpoints := p.director.endpoints()
	if len(endpoints) == 0 {
		msg := "zero endpoints currently available"
		reportRequestDropped(clientreq, zeroEndpoints)

		// TODO: limit the rate of the error logging.
		plog.Println(msg)
		e := httptypes.NewHTTPError(http.StatusServiceUnavailable, "httpproxy: "+msg)
		if we := e.WriteTo(rw); we != nil {
			plog.Debugf("error writing HTTPError (%v) to %s", we, clientreq.RemoteAddr)
		}
		return
	}

	var requestClosed int32
	completeCh := make(chan bool, 1)
	closeNotifier, ok := rw.(http.CloseNotifier)
	cancel := httputil.RequestCanceler(p.transport, proxyreq)
	if ok {
		closeCh := closeNotifier.CloseNotify()
		go func() {
			select {
			case <-closeCh:
				atomic.StoreInt32(&requestClosed, 1)
				plog.Printf("client %v closed request prematurely", clientreq.RemoteAddr)
				cancel()
			case <-completeCh:
			}
		}()

		defer func() {
			completeCh <- true
		}()
	}

	var res *http.Response

	for _, ep := range endpoints {
		if proxybody != nil {
			proxyreq.Body = ioutil.NopCloser(bytes.NewBuffer(proxybody))
		}
		redirectRequest(proxyreq, ep.URL)

		res, err = p.transport.RoundTrip(proxyreq)
		if atomic.LoadInt32(&requestClosed) == 1 {
			return
		}
		if err != nil {
			reportRequestDropped(clientreq, failedSendingRequest)
			plog.Printf("failed to direct request to %s: %v", ep.URL.String(), err)
			ep.Failed()
			continue
		}

		break
	}

	if res == nil {
		// TODO: limit the rate of the error logging.
		msg := fmt.Sprintf("unable to get response from %d endpoint(s)", len(endpoints))
		reportRequestDropped(clientreq, failedGettingResponse)
		plog.Println(msg)
		e := httptypes.NewHTTPError(http.StatusBadGateway, "httpproxy: "+msg)
		if we := e.WriteTo(rw); we != nil {
			plog.Debugf("error writing HTTPError (%v) to %s", we, clientreq.RemoteAddr)
		}
		return
	}

	defer res.Body.Close()
	reportRequestHandled(clientreq, res, startTime)
	removeSingleHopHeaders(&res.Header)
	copyHeader(rw.Header(), res.Header)

	rw.WriteHeader(res.StatusCode)
	io.Copy(rw, res.Body)
}

func copyHeader(dst, src http.Header) {
	for k, vv := range src {
		for _, v := range vv {
			dst.Add(k, v)
		}
	}
}

func redirectRequest(req *http.Request, loc url.URL) {
	req.URL.Scheme = loc.Scheme
	req.URL.Host = loc.Host
}

func normalizeRequest(req *http.Request) {
	req.Proto = "HTTP/1.1"
	req.ProtoMajor = 1
	req.ProtoMinor = 1
	req.Close = false
}

func maybeSetForwardedFor(req *http.Request) {
	clientIP, _, err := net.SplitHostPort(req.RemoteAddr)
	if err != nil {
		return
	}

	// If we aren't the first proxy retain prior
	// X-Forwarded-For information as a comma+space
	// separated list and fold multiple headers into one.
	if prior, ok := req.Header["X-Forwarded-For"]; ok {
		clientIP = strings.Join(prior, ", ") + ", " + clientIP
	}
	req.Header.Set("X-Forwarded-For", clientIP)
}
