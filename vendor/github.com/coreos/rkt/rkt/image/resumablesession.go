// Copyright 2015 The rkt Authors
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

package image

import (
	"crypto/tls"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/coreos/rkt/rkt/config"
	"github.com/coreos/rkt/version"
)

// statusAcceptedError is an error returned when resumableSession
// receives a 202 HTTP status. It is mostly used for deferring
// signature downloads.
type statusAcceptedError struct{}

func (*statusAcceptedError) Error() string {
	return "HTTP 202"
}

// cacheData holds caching-specific informations taken from various
// HTTP headers.
type cacheData struct {
	// whether we should reuse an image from store
	UseCached bool
	// image ETag, used for redownloading the obsolete images
	ETag string
	// MaxAge is a number of seconds telling when the downloaded
	// image is obsolete
	MaxAge int
}

// resumableSession is an implementation of the downloadSession
// interface, it allows sending custom headers for authentication,
// resuming interrupted downloads, handling cache data.
type resumableSession struct {
	// InsecureSkipTLSVerify tells whether TLS certificate
	// validation should be skipped.
	InsecureSkipTLSVerify bool
	// Headers are HTTP headers to be added to the HTTP
	// request. Used for ETAG.
	Headers http.Header
	// Headerers used for authentication.
	Headerers map[string]config.Headerer
	// File possibly holds the downloaded data - it is used for
	// resuming interrupted downloads.
	File *os.File
	// ETagFilePath is a path to a file holding an ETag of a
	// downloaded file. It is used for resuming interrupted
	// downloads.
	ETagFilePath string
	// Label is used for printing the type of the downloaded data
	// when printing a pretty progress bar.
	Label string

	// Cd is a cache data returned by HTTP server. It is an output
	// value.
	Cd *cacheData

	u                  *url.URL
	client             *http.Client
	amountAlreadyHere  int64
	byteRangeSupported bool
}

func (s *resumableSession) Client() (*http.Client, error) {
	s.ensureClient()
	return s.client, nil
}

func (s *resumableSession) Request(u *url.URL) (*http.Request, error) {
	s.u = u
	if err := s.maybeSetupDownloadResume(u); err != nil {
		return nil, err
	}
	return s.getRequest(u), nil
}

func (s *resumableSession) HandleStatus(res *http.Response) (bool, error) {
	switch res.StatusCode {
	case http.StatusOK, http.StatusPartialContent:
		fallthrough
	case http.StatusNotModified:
		s.Cd = &cacheData{
			ETag:      res.Header.Get("ETag"),
			MaxAge:    s.getMaxAge(res.Header.Get("Cache-Control")),
			UseCached: res.StatusCode == http.StatusNotModified,
		}
		return s.Cd.UseCached, nil
	case http.StatusAccepted:
		// If the server returns Status Accepted (HTTP 202), we should retry
		// downloading the signature later.
		return false, &statusAcceptedError{}
	case http.StatusRequestedRangeNotSatisfiable:
		return s.handleRangeNotSatisfiable()
	default:
		return false, fmt.Errorf("bad HTTP status code: %d", res.StatusCode)
	}
}

func (s *resumableSession) BodyReader(res *http.Response) (io.Reader, error) {
	reader := getIoProgressReader(s.Label, res)
	return reader, nil
}

type rangeStatus int

const (
	rangeSupported rangeStatus = iota
	rangeInvalid
	rangeUnsupported
)

func (s *resumableSession) maybeSetupDownloadResume(u *url.URL) error {
	fi, err := s.File.Stat()
	if err != nil {
		return err
	}

	size := fi.Size()
	if size < 1 {
		return nil
	}

	s.ensureClient()
	headReq := s.headRequest(u)
	res, err := s.client.Do(headReq)
	if err != nil {
		return err
	}
	if res.StatusCode != http.StatusOK {
		log.Printf("bad HTTP status code from HEAD request: %d", res.StatusCode)
		if err := s.reset(); err != nil {
			return err
		}
		return nil
	}
	status := s.verifyAcceptRange(res, fi.ModTime())
	if status == rangeSupported {
		s.byteRangeSupported = true
		s.amountAlreadyHere = size
	} else {
		if status == rangeInvalid {
			log.Printf("cannot use cached partial download, resource updated.")
		} else {
			log.Printf("cannot use cached partial download, range request unsupported.")
		}
		if err := s.reset(); err != nil {
			return err
		}
	}
	return nil
}

func (s *resumableSession) ensureClient() {
	if s.client == nil {
		s.client = s.getClient()
	}
}

func (s *resumableSession) getRequest(u *url.URL) *http.Request {
	return s.httpRequest("GET", u)
}

func (s *resumableSession) getMaxAge(headerValue string) int {
	if headerValue == "" {
		return 0
	}

	maxAge := 0
	parts := strings.Split(headerValue, ",")

	for i := 0; i < len(parts); i++ {
		parts[i] = strings.TrimSpace(parts[i])
		attr, val := parts[i], ""
		if j := strings.Index(attr, "="); j >= 0 {
			attr, val = attr[:j], attr[j+1:]
		}
		lowerAttr := strings.ToLower(attr)

		switch lowerAttr {
		case "no-store", "no-cache":
			maxAge = 0
			// TODO(krnowak): Just break out of the loop
			// at this point.
		case "max-age":
			secs, err := strconv.Atoi(val)
			if err != nil || secs != 0 && val[0] == '0' {
				// TODO(krnowak): Set maxAge to zero.
				break
			}
			if secs <= 0 {
				maxAge = 0
			} else {
				maxAge = secs
			}
		}
	}
	return maxAge
}

func (s *resumableSession) handleRangeNotSatisfiable() (bool, error) {
	if fi, err := s.File.Stat(); err != nil {
		return false, err
	} else if fi.Size() > 0 {
		if err := s.reset(); err != nil {
			return false, err
		}
		dl := &downloader{
			Session: s,
		}
		if err := dl.Download(s.u, s.File); err != nil {
			return false, err
		}
		return true, nil
	}
	code := http.StatusRequestedRangeNotSatisfiable
	return false, fmt.Errorf("bad HTTP status code: %d", code)
}

func (s *resumableSession) headRequest(u *url.URL) *http.Request {
	return s.httpRequest("HEAD", u)
}

func (s *resumableSession) verifyAcceptRange(res *http.Response, mod time.Time) rangeStatus {
	acceptRanges, hasRange := res.Header["Accept-Ranges"]
	if !hasRange {
		return rangeUnsupported
	}
	if !s.modificationTimeOK(res, mod) && !s.eTagOK(res) {
		return rangeInvalid
	}
	for _, rng := range acceptRanges {
		if rng == "bytes" {
			return rangeSupported
		}
	}
	return rangeInvalid
}

func (s *resumableSession) reset() error {
	s.amountAlreadyHere = 0
	s.byteRangeSupported = false
	if _, err := s.File.Seek(0, 0); err != nil {
		return err
	}
	if err := s.File.Truncate(0); err != nil {
		return err
	}
	if err := os.Remove(s.ETagFilePath); err != nil && !os.IsNotExist(err) {
		return err
	}
	return nil
}

func (s *resumableSession) getClient() *http.Client {
	transport := http.DefaultTransport
	if s.InsecureSkipTLSVerify {
		transport = &http.Transport{
			Proxy:           http.ProxyFromEnvironment,
			TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
		}
	}

	return &http.Client{
		Transport: transport,
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			if len(via) >= 10 {
				return fmt.Errorf("too many redirects")
			}
			stripAuth := false
			// don't propagate "Authorization" if the redirect is to a
			// different host
			previousHost := via[len(via)-1].URL.Host
			if previousHost != req.URL.Host {
				stripAuth = true
			}
			s.setHTTPHeaders(req, stripAuth)
			return nil
		},
	}
}

func (s *resumableSession) httpRequest(method string, u *url.URL) *http.Request {
	req := &http.Request{
		Method:     method,
		URL:        u,
		Proto:      "HTTP/1.1",
		ProtoMajor: 1,
		ProtoMinor: 1,
		Header:     make(http.Header),
		Host:       u.Host,
	}

	s.setHTTPHeaders(req, false)

	// Send credentials only over secure channel
	// TODO(krnowak): This could be controlled with another
	// insecure flag.
	if req.URL.Scheme != "https" {
		return req
	}

	if hostOpts, ok := s.Headerers[req.URL.Host]; ok {
		req = hostOpts.SignRequest(req)
		if req == nil {
			panic("Req is nil!")
		}
	}

	return req
}

func (s *resumableSession) modificationTimeOK(res *http.Response, mod time.Time) bool {
	lastModified := res.Header.Get("Last-Modified")
	if lastModified != "" {
		layout := "Mon, 02 Jan 2006 15:04:05 MST"
		t, err := time.Parse(layout, lastModified)
		if err == nil && t.Before(mod) {
			return true
		}
	}
	return false
}

func (s *resumableSession) eTagOK(res *http.Response) bool {
	etag := res.Header.Get("ETag")
	if etag != "" {
		savedEtag, err := ioutil.ReadFile(s.ETagFilePath)
		if err == nil && string(savedEtag) == etag {
			return true
		}
	}
	return false
}

func (s *resumableSession) setHTTPHeaders(req *http.Request, stripAuth bool) {
	for k, v := range s.Headers {
		if stripAuth && k == "Authorization" {
			continue
		}
		for _, e := range v {
			req.Header.Add(k, e)
		}
	}
	req.Header.Add("User-Agent", fmt.Sprintf("rkt/%s", version.Version))
	if s.amountAlreadyHere > 0 && s.byteRangeSupported {
		req.Header.Add("Range", fmt.Sprintf("bytes=%d-", s.amountAlreadyHere))
	}
}
