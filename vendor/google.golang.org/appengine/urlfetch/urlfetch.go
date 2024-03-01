// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by the Apache 2.0
// license that can be found in the LICENSE file.

// Package urlfetch provides an http.RoundTripper implementation
// for fetching URLs via App Engine's urlfetch service.
package urlfetch // import "google.golang.org/appengine/urlfetch"

import (
	"context"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/golang/protobuf/proto"

	"google.golang.org/appengine/internal"
	pb "google.golang.org/appengine/internal/urlfetch"
)

// Transport is an implementation of http.RoundTripper for
// App Engine. Users should generally create an http.Client using
// this transport and use the Client rather than using this transport
// directly.
type Transport struct {
	Context context.Context

	// Controls whether the application checks the validity of SSL certificates
	// over HTTPS connections. A value of false (the default) instructs the
	// application to send a request to the server only if the certificate is
	// valid and signed by a trusted certificate authority (CA), and also
	// includes a hostname that matches the certificate. A value of true
	// instructs the application to perform no certificate validation.
	AllowInvalidServerCertificate bool
}

// Verify statically that *Transport implements http.RoundTripper.
var _ http.RoundTripper = (*Transport)(nil)

// Client returns an *http.Client using a default urlfetch Transport. This
// client will check the validity of SSL certificates.
//
// Any deadline of the provided context will be used for requests through this client.
// If the client does not have a deadline, then an App Engine default of 60 second is used.
func Client(ctx context.Context) *http.Client {
	return &http.Client{
		Transport: &Transport{
			Context: ctx,
		},
	}
}

type bodyReader struct {
	content   []byte
	truncated bool
	closed    bool
}

// ErrTruncatedBody is the error returned after the final Read() from a
// response's Body if the body has been truncated by App Engine's proxy.
var ErrTruncatedBody = errors.New("urlfetch: truncated body")

func statusCodeToText(code int) string {
	if t := http.StatusText(code); t != "" {
		return t
	}
	return strconv.Itoa(code)
}

func (br *bodyReader) Read(p []byte) (n int, err error) {
	if br.closed {
		if br.truncated {
			return 0, ErrTruncatedBody
		}
		return 0, io.EOF
	}
	n = copy(p, br.content)
	if n > 0 {
		br.content = br.content[n:]
		return
	}
	if br.truncated {
		br.closed = true
		return 0, ErrTruncatedBody
	}
	return 0, io.EOF
}

func (br *bodyReader) Close() error {
	br.closed = true
	br.content = nil
	return nil
}

// A map of the URL Fetch-accepted methods that take a request body.
var methodAcceptsRequestBody = map[string]bool{
	"POST":  true,
	"PUT":   true,
	"PATCH": true,
}

// urlString returns a valid string given a URL. This function is necessary because
// the String method of URL doesn't correctly handle URLs with non-empty Opaque values.
// See http://code.google.com/p/go/issues/detail?id=4860.
func urlString(u *url.URL) string {
	if u.Opaque == "" || strings.HasPrefix(u.Opaque, "//") {
		return u.String()
	}
	aux := *u
	aux.Opaque = "//" + aux.Host + aux.Opaque
	return aux.String()
}

// RoundTrip issues a single HTTP request and returns its response. Per the
// http.RoundTripper interface, RoundTrip only returns an error if there
// was an unsupported request or the URL Fetch proxy fails.
// Note that HTTP response codes such as 5xx, 403, 404, etc are not
// errors as far as the transport is concerned and will be returned
// with err set to nil.
func (t *Transport) RoundTrip(req *http.Request) (res *http.Response, err error) {
	methNum, ok := pb.URLFetchRequest_RequestMethod_value[req.Method]
	if !ok {
		return nil, fmt.Errorf("urlfetch: unsupported HTTP method %q", req.Method)
	}

	method := pb.URLFetchRequest_RequestMethod(methNum)

	freq := &pb.URLFetchRequest{
		Method:                        &method,
		Url:                           proto.String(urlString(req.URL)),
		FollowRedirects:               proto.Bool(false), // http.Client's responsibility
		MustValidateServerCertificate: proto.Bool(!t.AllowInvalidServerCertificate),
	}
	if deadline, ok := t.Context.Deadline(); ok {
		freq.Deadline = proto.Float64(deadline.Sub(time.Now()).Seconds())
	}

	for k, vals := range req.Header {
		for _, val := range vals {
			freq.Header = append(freq.Header, &pb.URLFetchRequest_Header{
				Key:   proto.String(k),
				Value: proto.String(val),
			})
		}
	}
	if methodAcceptsRequestBody[req.Method] && req.Body != nil {
		// Avoid a []byte copy if req.Body has a Bytes method.
		switch b := req.Body.(type) {
		case interface {
			Bytes() []byte
		}:
			freq.Payload = b.Bytes()
		default:
			freq.Payload, err = ioutil.ReadAll(req.Body)
			if err != nil {
				return nil, err
			}
		}
	}

	fres := &pb.URLFetchResponse{}
	if err := internal.Call(t.Context, "urlfetch", "Fetch", freq, fres); err != nil {
		return nil, err
	}

	res = &http.Response{}
	res.StatusCode = int(*fres.StatusCode)
	res.Status = fmt.Sprintf("%d %s", res.StatusCode, statusCodeToText(res.StatusCode))
	res.Header = make(http.Header)
	res.Request = req

	// Faked:
	res.ProtoMajor = 1
	res.ProtoMinor = 1
	res.Proto = "HTTP/1.1"
	res.Close = true

	for _, h := range fres.Header {
		hkey := http.CanonicalHeaderKey(*h.Key)
		hval := *h.Value
		if hkey == "Content-Length" {
			// Will get filled in below for all but HEAD requests.
			if req.Method == "HEAD" {
				res.ContentLength, _ = strconv.ParseInt(hval, 10, 64)
			}
			continue
		}
		res.Header.Add(hkey, hval)
	}

	if req.Method != "HEAD" {
		res.ContentLength = int64(len(fres.Content))
	}

	truncated := fres.GetContentWasTruncated()
	res.Body = &bodyReader{content: fres.Content, truncated: truncated}
	return
}

func init() {
	internal.RegisterErrorCodeMap("urlfetch", pb.URLFetchServiceError_ErrorCode_name)
	internal.RegisterTimeoutErrorCode("urlfetch", int32(pb.URLFetchServiceError_DEADLINE_EXCEEDED))
}
