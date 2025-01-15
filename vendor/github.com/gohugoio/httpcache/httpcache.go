// Package httpcache provides a http.RoundTripper implementation that works as a
// mostly RFC-compliant cache for http responses.
//
// It is only suitable for use as a 'private' cache (i.e. for a web-browser or an API-client
// and not for a shared proxy).
package httpcache

import (
	"bufio"
	"bytes"
	"crypto/md5"
	"encoding/hex"
	"errors"
	"hash"
	"io"
	"net/http"
	"net/http/httputil"
	"strings"
	"time"
)

const (
	stale = iota
	fresh
	transparent
	// XFromCache is the header added to responses that are returned from the cache
	XFromCache = "X-From-Cache"

	// xEtags is the prefix for the header with the custom etag pair set in the cached response.
	xEtags = "X-Etags-"

	// XETag1 is the key for the first eTag value.
	XETag1 = xEtags + "1"

	// XETag2 is the key for the second eTag value.
	// Note that in the cache, XETag1 and XETag2 will always be the same.
	// In the Response returned from Response, XETag1 will be the cached value (old) and
	// XETag2 will be the eTag value from the server (new).
	XETag2 = xEtags + "2"
)

// A Cache interface is used by the Transport to store and retrieve responses.
type Cache interface {
	// Get returns the []byte representation of a cached response and a bool
	// set to set to false if the key is not found or the value is stale.
	Get(key string) (responseBytes []byte, ok bool)
	// Set stores the []byte representation of a response against a key
	Set(key string, responseBytes []byte)
	// Delete removes the value associated with the key
	Delete(key string)
}

// cacheKey returns the cache key for req.
func (t *Transport) cacheKey(req *http.Request) string {
	if t.CacheKey != nil {
		return t.CacheKey(req)
	}

	cacheable := (req.Method != http.MethodHead || req.Method == "HEAD") && req.Header.Get("range") == ""
	if !cacheable {
		return ""
	}

	if req.Method == http.MethodGet {
		return req.URL.String()
	} else {
		return req.Method + " " + req.URL.String()
	}
}

// cachedResponse returns the cached http.Response for req if present and
// a bool set to false if the value is stale.
func (t *Transport) cachedResponse(req *http.Request) (*http.Response, bool, error) {
	cachedVal, ok := t.Cache.Get(t.cacheKey(req))
	if !ok && len(cachedVal) == 0 {
		return nil, false, nil
	}
	b := bytes.NewBuffer(cachedVal)
	resp, err := http.ReadResponse(bufio.NewReader(b), req)
	if err != nil {
		return nil, false, err
	}
	return resp, ok, nil
}

// Transport is an implementation of http.RoundTripper that will return values from a cache
// where possible (avoiding a network request) and will additionally add validators (etag/if-modified-since)
// to repeated requests allowing servers to return 304 / Not Modified
type Transport struct {
	// The RoundTripper interface actually used to make requests
	// If nil, http.DefaultTransport is used
	Transport http.RoundTripper

	// The Cache interface used to store and retrieve responses.
	Cache Cache

	// If true, responses returned from the cache will be given an extra header, X-From-Cache
	MarkCachedResponses bool

	// if EnableETagPair is true, the Transport will store the pair of eTags in the response header.
	// These are stored in the X-Etags-1 and X-Etags-2 headers.
	// If these are different, the response has been modified.
	// If the server does not return an eTag, the MD5 hash of the response body is used.
	EnableETagPair bool

	// CacheKey is an optional func that returns the key to use to store the response.
	// An empty string signals that this request should not be cached.
	CacheKey func(req *http.Request) string

	// AlwaysUseCachedResponse is an optional func that when it returns true
	// a successful response from the cache will be returned without connecting to the server.
	AlwaysUseCachedResponse func(req *http.Request, key string) bool

	// ShouldCache is an optional func that when it returns false, the response will not be cached.
	ShouldCache func(req *http.Request, resp *http.Response, key string) bool

	// Around is an optional func.
	// If set, the Transport will call Around at the start of RoundTrip
	// and defer the returned func until the end of RoundTrip.
	// Typically used to implement a lock that is held for the duration of the RoundTrip.
	Around func(req *http.Request, key string) func()
}

// varyMatches will return false unless all of the cached values for the headers listed in Vary
// match the new request
func varyMatches(cachedResp *http.Response, req *http.Request) bool {
	for _, header := range headerAllCommaSepValues(cachedResp.Header, "vary") {
		header = http.CanonicalHeaderKey(header)
		if header != "" && req.Header.Get(header) != cachedResp.Header.Get("X-Varied-"+header) {
			return false
		}
	}
	return true
}

// RoundTrip takes a Request and returns a Response
//
// If there is a fresh Response already in cache, then it will be returned without connecting to
// the server.
//
// If there is a stale Response, then any validators it contains will be set on the new request
// to give the server a chance to respond with NotModified. If this happens, then the cached Response
// will be returned.
func (t *Transport) RoundTrip(req *http.Request) (resp *http.Response, err error) {
	cacheKey := t.cacheKey(req)
	if f := t.Around; f != nil {
		defer f(req, cacheKey)()
	}

	var cachedXEtag string

	cacheable := cacheKey != ""

	var (
		cachedResp    *http.Response
		hasCachedResp bool
	)
	if cacheable {
		cachedResp, hasCachedResp, err = t.cachedResponse(req)
		if err == nil && hasCachedResp && t.AlwaysUseCachedResponse != nil && t.AlwaysUseCachedResponse(req, cacheKey) {
			return cachedResp, nil
		}
	} else {
		// Need to invalidate an existing value
		t.Cache.Delete(cacheKey)
	}

	transport := t.Transport
	if transport == nil {
		transport = http.DefaultTransport
	}

	if cachedResp != nil {
		if t.EnableETagPair {
			cachedXEtag, _ = getXETags(cachedResp.Header)
		}
	}

	if cacheable && hasCachedResp && err == nil {
		if t.MarkCachedResponses {
			cachedResp.Header.Set(XFromCache, "1")
		}

		if varyMatches(cachedResp, req) {
			// Can only use cached value if the new request doesn't Vary significantly
			freshness := getFreshness(cachedResp.Header, req.Header)
			if freshness == fresh {
				return cachedResp, nil
			}

			if freshness == stale {
				var req2 *http.Request
				// Add validators if caller hasn't already done so
				etag := cachedResp.Header.Get("etag")
				if etag != "" && req.Header.Get("etag") == "" {
					req2 = cloneRequest(req)
					req2.Header.Set("if-none-match", etag)
				}
				lastModified := cachedResp.Header.Get("last-modified")
				if lastModified != "" && req.Header.Get("last-modified") == "" {
					if req2 == nil {
						req2 = cloneRequest(req)
					}
					req2.Header.Set("if-modified-since", lastModified)
				}
				if req2 != nil {
					req = req2
				}
			}
		}

		resp, err = transport.RoundTrip(req)

		if err == nil && req.Method != http.MethodHead && resp.StatusCode == http.StatusNotModified {
			// Replace the 304 response with the one from cache, but update with some new headers
			endToEndHeaders := getEndToEndHeaders(resp.Header)
			for _, header := range endToEndHeaders {
				cachedResp.Header[header] = resp.Header[header]
			}
			resp = cachedResp
		} else if (err != nil || resp.StatusCode >= 500) &&
			req.Method != http.MethodHead && canStaleOnError(cachedResp.Header, req.Header) {
			// In case of transport failure and stale-if-error activated, returns cached content
			// when available
			return cachedResp, nil
		} else {
			if err != nil || resp.StatusCode != http.StatusOK {
				t.Cache.Delete(cacheKey)
			}
			if err != nil {
				return nil, err
			}
		}
	} else {
		reqCacheControl := parseCacheControl(req.Header)
		if _, ok := reqCacheControl["only-if-cached"]; ok {
			resp = newGatewayTimeoutResponse(req)
		} else {
			resp, err = transport.RoundTrip(req)
			if err != nil {
				return nil, err
			}
		}
	}

	if cacheable && (t.ShouldCache == nil || t.ShouldCache(req, resp, cacheKey)) && canStore(parseCacheControl(req.Header), parseCacheControl(resp.Header)) {
		for _, varyKey := range headerAllCommaSepValues(resp.Header, "vary") {
			varyKey = http.CanonicalHeaderKey(varyKey)
			fakeHeader := "X-Varied-" + varyKey
			reqValue := req.Header.Get(varyKey)
			if reqValue != "" {
				resp.Header.Set(fakeHeader, reqValue)
			}
		}
		switch req.Method {
		case http.MethodHead:
			respBytes, err := httputil.DumpResponse(resp, true)
			if err == nil {
				t.Cache.Set(cacheKey, respBytes)
			}
		default:
			var (
				etagHash hash.Hash
				etag1    = cachedXEtag
				etag2    string
			)

			r := resp.Body
			if t.EnableETagPair {
				if etag := resp.Header.Get("etag"); etag != "" {
					etag1 = etag
					if etag2 == "" {
						etag2 = etag
					}
				} else {
					etagHash = md5.New()
					r = struct {
						io.Reader
						io.Closer
					}{
						io.TeeReader(r, etagHash),
						resp.Body,
					}
				}
			}

			r = &cachingReadCloser{
				R: r,
				OnEOF: func(r io.Reader) {
					if etagHash != nil {
						md5Str := hex.EncodeToString(etagHash.Sum(nil))
						etag2 = md5Str
						resp.Header.Set(XETag1, md5Str)
						resp.Header.Set(XETag2, md5Str)
						if etag1 == "" {
							etag1 = md5Str
						}
					} else {
						resp.Header.Set(XETag1, etag1)
						resp.Header.Set(XETag2, etag1)
					}

					resp := *resp
					resp.Body = io.NopCloser(r)
					respBytes, err := httputil.DumpResponse(&resp, true)
					if err == nil {
						// Signal any change back to the caller.
						resp.Header.Set(XETag1, etag1)
						t.Cache.Set(cacheKey, respBytes)
					}
				},
				buf: &bytes.Buffer{},
			}
			// Delay caching until EOF is reached.
			resp.Body = r

		}
	} else {
		t.Cache.Delete(cacheKey)
	}
	return resp, nil
}

// ErrNoDateHeader indicates that the HTTP headers contained no Date header.
var ErrNoDateHeader = errors.New("no Date header")

// date parses and returns the value of the date header.
func date(respHeaders http.Header) (date time.Time, err error) {
	dateHeader := respHeaders.Get("date")
	if dateHeader == "" {
		err = ErrNoDateHeader
		return
	}

	return time.Parse(time.RFC1123, dateHeader)
}

type realClock struct{}

func (c *realClock) since(d time.Time) time.Duration {
	return time.Since(d)
}

type timer interface {
	since(d time.Time) time.Duration
}

var clock timer = &realClock{}

func getXETags(h http.Header) (string, string) {
	return h.Get(XETag1), h.Get(XETag2)
}

// getFreshness will return one of fresh/stale/transparent based on the cache-control
// values of the request and the response
//
// fresh indicates the response can be returned
// stale indicates that the response needs validating before it is returned
// transparent indicates the response should not be used to fulfil the request
//
// Because this is only a private cache, 'public' and 'private' in cache-control aren't
// significant. Similarly, smax-age isn't used.
func getFreshness(respHeaders, reqHeaders http.Header) (freshness int) {
	respCacheControl := parseCacheControl(respHeaders)
	reqCacheControl := parseCacheControl(reqHeaders)
	if _, ok := reqCacheControl["no-cache"]; ok {
		return transparent
	}
	if _, ok := respCacheControl["no-cache"]; ok {
		return stale
	}
	if _, ok := reqCacheControl["only-if-cached"]; ok {
		return fresh
	}

	date, err := date(respHeaders)
	if err != nil {
		return stale
	}
	currentAge := clock.since(date)

	var lifetime time.Duration
	var zeroDuration time.Duration

	// If a response includes both an Expires header and a max-age directive,
	// the max-age directive overrides the Expires header, even if the Expires header is more restrictive.
	if maxAge, ok := respCacheControl["max-age"]; ok {
		lifetime, err = time.ParseDuration(maxAge + "s")
		if err != nil {
			lifetime = zeroDuration
		}
	} else {
		expiresHeader := respHeaders.Get("Expires")
		if expiresHeader != "" {
			expires, err := time.Parse(time.RFC1123, expiresHeader)
			if err != nil {
				lifetime = zeroDuration
			} else {
				lifetime = expires.Sub(date)
			}
		}
	}

	if maxAge, ok := reqCacheControl["max-age"]; ok {
		// the client is willing to accept a response whose age is no greater than the specified time in seconds
		lifetime, err = time.ParseDuration(maxAge + "s")
		if err != nil {
			lifetime = zeroDuration
		}
	}
	if minfresh, ok := reqCacheControl["min-fresh"]; ok {
		//  the client wants a response that will still be fresh for at least the specified number of seconds.
		minfreshDuration, err := time.ParseDuration(minfresh + "s")
		if err == nil {
			currentAge = time.Duration(currentAge + minfreshDuration)
		}
	}

	if maxstale, ok := reqCacheControl["max-stale"]; ok {
		// Indicates that the client is willing to accept a response that has exceeded its expiration time.
		// If max-stale is assigned a value, then the client is willing to accept a response that has exceeded
		// its expiration time by no more than the specified number of seconds.
		// If no value is assigned to max-stale, then the client is willing to accept a stale response of any age.
		//
		// Responses served only because of a max-stale value are supposed to have a Warning header added to them,
		// but that seems like a  hassle, and is it actually useful? If so, then there needs to be a different
		// return-value available here.
		if maxstale == "" {
			return fresh
		}
		maxstaleDuration, err := time.ParseDuration(maxstale + "s")
		if err == nil {
			currentAge = time.Duration(currentAge - maxstaleDuration)
		}
	}

	if lifetime > currentAge {
		return fresh
	}

	return stale
}

// Returns true if either the request or the response includes the stale-if-error
// cache control extension: https://tools.ietf.org/html/rfc5861
func canStaleOnError(respHeaders, reqHeaders http.Header) bool {
	respCacheControl := parseCacheControl(respHeaders)
	reqCacheControl := parseCacheControl(reqHeaders)

	var err error
	lifetime := time.Duration(-1)

	if staleMaxAge, ok := respCacheControl["stale-if-error"]; ok {
		if staleMaxAge != "" {
			lifetime, err = time.ParseDuration(staleMaxAge + "s")
			if err != nil {
				return false
			}
		} else {
			return true
		}
	}
	if staleMaxAge, ok := reqCacheControl["stale-if-error"]; ok {
		if staleMaxAge != "" {
			lifetime, err = time.ParseDuration(staleMaxAge + "s")
			if err != nil {
				return false
			}
		} else {
			return true
		}
	}

	if lifetime >= 0 {
		date, err := date(respHeaders)
		if err != nil {
			return false
		}
		currentAge := clock.since(date)
		if lifetime > currentAge {
			return true
		}
	}

	return false
}

func getEndToEndHeaders(respHeaders http.Header) []string {
	// These headers are always hop-by-hop
	hopByHopHeaders := map[string]struct{}{
		"Connection":          {},
		"Keep-Alive":          {},
		"Proxy-Authenticate":  {},
		"Proxy-Authorization": {},
		"Te":                  {},
		"Trailers":            {},
		"Transfer-Encoding":   {},
		"Upgrade":             {},
	}

	for _, extra := range strings.Split(respHeaders.Get("connection"), ",") {
		// any header listed in connection, if present, is also considered hop-by-hop
		if strings.Trim(extra, " ") != "" {
			hopByHopHeaders[http.CanonicalHeaderKey(extra)] = struct{}{}
		}
	}
	endToEndHeaders := []string{}
	for respHeader := range respHeaders {
		if _, ok := hopByHopHeaders[respHeader]; !ok {
			endToEndHeaders = append(endToEndHeaders, respHeader)
		}
	}
	return endToEndHeaders
}

func canStore(reqCacheControl, respCacheControl cacheControl) (canStore bool) {
	if _, ok := respCacheControl["no-store"]; ok {
		return false
	}
	if _, ok := reqCacheControl["no-store"]; ok {
		return false
	}
	return true
}

func newGatewayTimeoutResponse(req *http.Request) *http.Response {
	var braw bytes.Buffer
	braw.WriteString("HTTP/1.1 504 Gateway Timeout\r\n\r\n")
	resp, err := http.ReadResponse(bufio.NewReader(&braw), req)
	if err != nil {
		panic(err)
	}
	return resp
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
// (This function copyright goauth2 authors: https://code.google.com/p/goauth2)
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header)
	for k, s := range r.Header {
		r2.Header[k] = s
	}
	return r2
}

type cacheControl map[string]string

func parseCacheControl(headers http.Header) cacheControl {
	cc := cacheControl{}
	ccHeader := headers.Get("Cache-Control")
	for _, part := range strings.Split(ccHeader, ",") {
		part = strings.Trim(part, " ")
		if part == "" {
			continue
		}
		if strings.ContainsRune(part, '=') {
			keyval := strings.Split(part, "=")
			cc[strings.Trim(keyval[0], " ")] = strings.Trim(keyval[1], ",")
		} else {
			cc[part] = ""
		}
	}
	return cc
}

// headerAllCommaSepValues returns all comma-separated values (each
// with whitespace trimmed) for header name in headers. According to
// Section 4.2 of the HTTP/1.1 spec
// (http://www.w3.org/Protocols/rfc2616/rfc2616-sec4.html#sec4.2),
// values from multiple occurrences of a header should be concatenated, if
// the header's value is a comma-separated list.
func headerAllCommaSepValues(headers http.Header, name string) []string {
	var vals []string
	for _, val := range headers[http.CanonicalHeaderKey(name)] {
		fields := strings.Split(val, ",")
		for i, f := range fields {
			fields[i] = strings.TrimSpace(f)
		}
		vals = append(vals, fields...)
	}
	return vals
}

// cachingReadCloser is a wrapper around ReadCloser R that calls OnEOF
// handler with a full copy of the content read from R when EOF is
// reached.
type cachingReadCloser struct {
	// Underlying ReadCloser.
	R io.ReadCloser
	// OnEOF is called with a copy of the content of R when EOF is reached.
	OnEOF func(io.Reader)

	buf *bytes.Buffer // buf stores a copy of the content of R.
}

// Read reads the next len(p) bytes from R or until R is drained. The
// return value n is the number of bytes read. If R has no data to
// return, err is io.EOF and OnEOF is called with a full copy of what
// has been read so far.
func (r *cachingReadCloser) Read(p []byte) (n int, err error) {
	n, err = r.R.Read(p)
	r.buf.Write(p[:n])
	if err == io.EOF {
		r.OnEOF(r.buf)
	}
	return n, err
}

func (r *cachingReadCloser) Close() error {
	return r.R.Close()
}
