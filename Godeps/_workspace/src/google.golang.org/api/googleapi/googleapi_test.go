// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googleapi

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"testing"
	"time"

	"golang.org/x/net/context"
)

type SetOpaqueTest struct {
	in             *url.URL
	wantRequestURI string
}

var setOpaqueTests = []SetOpaqueTest{
	// no path
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.golang.org",
		},
		"http://www.golang.org",
	},
	// path
	{
		&url.URL{
			Scheme: "http",
			Host:   "www.golang.org",
			Path:   "/",
		},
		"http://www.golang.org/",
	},
	// file with hex escaping
	{
		&url.URL{
			Scheme: "https",
			Host:   "www.golang.org",
			Path:   "/file%20one&two",
		},
		"https://www.golang.org/file%20one&two",
	},
	// query
	{
		&url.URL{
			Scheme:   "http",
			Host:     "www.golang.org",
			Path:     "/",
			RawQuery: "q=go+language",
		},
		"http://www.golang.org/?q=go+language",
	},
	// file with hex escaping in path plus query
	{
		&url.URL{
			Scheme:   "https",
			Host:     "www.golang.org",
			Path:     "/file%20one&two",
			RawQuery: "q=go+language",
		},
		"https://www.golang.org/file%20one&two?q=go+language",
	},
	// query with hex escaping
	{
		&url.URL{
			Scheme:   "http",
			Host:     "www.golang.org",
			Path:     "/",
			RawQuery: "q=go%20language",
		},
		"http://www.golang.org/?q=go%20language",
	},
}

// prefixTmpl is a template for the expected prefix of the output of writing
// an HTTP request.
const prefixTmpl = "GET %v HTTP/1.1\r\nHost: %v\r\n"

func TestSetOpaque(t *testing.T) {
	for _, test := range setOpaqueTests {
		u := *test.in
		SetOpaque(&u)

		w := &bytes.Buffer{}
		r := &http.Request{URL: &u}
		if err := r.Write(w); err != nil {
			t.Errorf("write request: %v", err)
			continue
		}

		prefix := fmt.Sprintf(prefixTmpl, test.wantRequestURI, test.in.Host)
		if got := string(w.Bytes()); !strings.HasPrefix(got, prefix) {
			t.Errorf("got %q expected prefix %q", got, prefix)
		}
	}
}

type ExpandTest struct {
	in         string
	expansions map[string]string
	want       string
}

var expandTests = []ExpandTest{
	// no expansions
	{
		"http://www.golang.org/",
		map[string]string{},
		"http://www.golang.org/",
	},
	// one expansion, no escaping
	{
		"http://www.golang.org/{bucket}/delete",
		map[string]string{
			"bucket": "red",
		},
		"http://www.golang.org/red/delete",
	},
	// one expansion, with hex escapes
	{
		"http://www.golang.org/{bucket}/delete",
		map[string]string{
			"bucket": "red/blue",
		},
		"http://www.golang.org/red%2Fblue/delete",
	},
	// one expansion, with space
	{
		"http://www.golang.org/{bucket}/delete",
		map[string]string{
			"bucket": "red or blue",
		},
		"http://www.golang.org/red%20or%20blue/delete",
	},
	// expansion not found
	{
		"http://www.golang.org/{object}/delete",
		map[string]string{
			"bucket": "red or blue",
		},
		"http://www.golang.org//delete",
	},
	// multiple expansions
	{
		"http://www.golang.org/{one}/{two}/{three}/get",
		map[string]string{
			"one":   "ONE",
			"two":   "TWO",
			"three": "THREE",
		},
		"http://www.golang.org/ONE/TWO/THREE/get",
	},
	// utf-8 characters
	{
		"http://www.golang.org/{bucket}/get",
		map[string]string{
			"bucket": "£100",
		},
		"http://www.golang.org/%C2%A3100/get",
	},
	// punctuations
	{
		"http://www.golang.org/{bucket}/get",
		map[string]string{
			"bucket": `/\@:,.`,
		},
		"http://www.golang.org/%2F%5C%40%3A%2C./get",
	},
	// mis-matched brackets
	{
		"http://www.golang.org/{bucket/get",
		map[string]string{
			"bucket": "red",
		},
		"http://www.golang.org/{bucket/get",
	},
	// "+" prefix for suppressing escape
	// See also: http://tools.ietf.org/html/rfc6570#section-3.2.3
	{
		"http://www.golang.org/{+topic}",
		map[string]string{
			"topic": "/topics/myproject/mytopic",
		},
		// The double slashes here look weird, but it's intentional
		"http://www.golang.org//topics/myproject/mytopic",
	},
}

func TestExpand(t *testing.T) {
	for i, test := range expandTests {
		u := url.URL{
			Path: test.in,
		}
		Expand(&u, test.expansions)
		got := u.Path
		if got != test.want {
			t.Errorf("got %q expected %q in test %d", got, test.want, i+1)
		}
	}
}

type CheckResponseTest struct {
	in       *http.Response
	bodyText string
	want     error
	errText  string
}

var checkResponseTests = []CheckResponseTest{
	{
		&http.Response{
			StatusCode: http.StatusOK,
		},
		"",
		nil,
		"",
	},
	{
		&http.Response{
			StatusCode: http.StatusInternalServerError,
		},
		`{"error":{}}`,
		&Error{
			Code: http.StatusInternalServerError,
			Body: `{"error":{}}`,
		},
		`googleapi: got HTTP response code 500 with body: {"error":{}}`,
	},
	{
		&http.Response{
			StatusCode: http.StatusNotFound,
		},
		`{"error":{"message":"Error message for StatusNotFound."}}`,
		&Error{
			Code:    http.StatusNotFound,
			Message: "Error message for StatusNotFound.",
			Body:    `{"error":{"message":"Error message for StatusNotFound."}}`,
		},
		"googleapi: Error 404: Error message for StatusNotFound.",
	},
	{
		&http.Response{
			StatusCode: http.StatusBadRequest,
		},
		`{"error":"invalid_token","error_description":"Invalid Value"}`,
		&Error{
			Code: http.StatusBadRequest,
			Body: `{"error":"invalid_token","error_description":"Invalid Value"}`,
		},
		`googleapi: got HTTP response code 400 with body: {"error":"invalid_token","error_description":"Invalid Value"}`,
	},
	{
		&http.Response{
			StatusCode: http.StatusBadRequest,
		},
		`{"error":{"errors":[{"domain":"usageLimits","reason":"keyInvalid","message":"Bad Request"}],"code":400,"message":"Bad Request"}}`,
		&Error{
			Code: http.StatusBadRequest,
			Errors: []ErrorItem{
				{
					Reason:  "keyInvalid",
					Message: "Bad Request",
				},
			},
			Body:    `{"error":{"errors":[{"domain":"usageLimits","reason":"keyInvalid","message":"Bad Request"}],"code":400,"message":"Bad Request"}}`,
			Message: "Bad Request",
		},
		"googleapi: Error 400: Bad Request, keyInvalid",
	},
}

func TestCheckResponse(t *testing.T) {
	for _, test := range checkResponseTests {
		res := test.in
		if test.bodyText != "" {
			res.Body = ioutil.NopCloser(strings.NewReader(test.bodyText))
		}
		g := CheckResponse(res)
		if !reflect.DeepEqual(g, test.want) {
			t.Errorf("CheckResponse: got %v, want %v", g, test.want)
			gotJson, err := json.Marshal(g)
			if err != nil {
				t.Error(err)
			}
			wantJson, err := json.Marshal(test.want)
			if err != nil {
				t.Error(err)
			}
			t.Errorf("json(got):  %q\njson(want): %q", string(gotJson), string(wantJson))
		}
		if g != nil && g.Error() != test.errText {
			t.Errorf("CheckResponse: unexpected error message.\nGot:  %q\nwant: %q", g, test.errText)
		}
	}
}

type VariantPoint struct {
	Type        string
	Coordinates []float64
}

type VariantTest struct {
	in     map[string]interface{}
	result bool
	want   VariantPoint
}

var coords = []interface{}{1.0, 2.0}

var variantTests = []VariantTest{
	{
		in: map[string]interface{}{
			"type":        "Point",
			"coordinates": coords,
		},
		result: true,
		want: VariantPoint{
			Type:        "Point",
			Coordinates: []float64{1.0, 2.0},
		},
	},
	{
		in: map[string]interface{}{
			"type":  "Point",
			"bogus": coords,
		},
		result: true,
		want: VariantPoint{
			Type: "Point",
		},
	},
}

func TestVariantType(t *testing.T) {
	for _, test := range variantTests {
		if g := VariantType(test.in); g != test.want.Type {
			t.Errorf("VariantType(%v): got %v, want %v", test.in, g, test.want.Type)
		}
	}
}

func TestConvertVariant(t *testing.T) {
	for _, test := range variantTests {
		g := VariantPoint{}
		r := ConvertVariant(test.in, &g)
		if r != test.result {
			t.Errorf("ConvertVariant(%v): got %v, want %v", test.in, r, test.result)
		}
		if !reflect.DeepEqual(g, test.want) {
			t.Errorf("ConvertVariant(%v): got %v, want %v", test.in, g, test.want)
		}
	}
}

type unexpectedReader struct{}

func (unexpectedReader) Read([]byte) (int, error) {
	return 0, fmt.Errorf("unexpected read in test.")
}

var contentRangeRE = regexp.MustCompile(`^bytes (\d+)\-(\d+)/(\d+)$`)

func (t *testTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.req = req
	if rng := req.Header.Get("Content-Range"); rng != "" && !strings.HasPrefix(rng, "bytes */") { // Read the data
		m := contentRangeRE.FindStringSubmatch(rng)
		if len(m) != 4 {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		start, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		end, err := strconv.ParseInt(m[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		totalSize, err := strconv.ParseInt(m[3], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		partialSize := end - start + 1
		t.buf, err = ioutil.ReadAll(req.Body)
		if err != nil || int64(len(t.buf)) != partialSize {
			return nil, fmt.Errorf("unable to read %v bytes from request data, n=%v: %v", partialSize, len(t.buf), err)
		}
		if totalSize == end+1 {
			t.statusCode = 200 // signify completion of transfer
		}
	}
	f := ioutil.NopCloser(unexpectedReader{})
	res := &http.Response{
		Body:       f,
		StatusCode: t.statusCode,
		Header:     http.Header{},
	}
	if t.rangeVal != "" {
		res.Header.Set("Range", t.rangeVal)
	}
	return res, nil
}

type testTransport struct {
	req        *http.Request
	statusCode int
	rangeVal   string
	want       int64
	buf        []byte
}

var statusTests = []*testTransport{
	&testTransport{statusCode: 308, want: 0},
	&testTransport{statusCode: 308, rangeVal: "bytes=0-0", want: 1},
	&testTransport{statusCode: 308, rangeVal: "bytes=0-42", want: 43},
}

func TestTransferStatus(t *testing.T) {
	for _, tr := range statusTests {
		rx := &ResumableUpload{
			Client: &http.Client{Transport: tr},
		}
		g, _, err := rx.transferStatus()
		if err != nil {
			t.Error(err)
		}
		if g != tr.want {
			t.Errorf("transferStatus got %v, want %v", g, tr.want)
		}
	}
}

func (t *interruptedTransport) RoundTrip(req *http.Request) (*http.Response, error) {
	t.req = req
	if rng := req.Header.Get("Content-Range"); rng != "" && !strings.HasPrefix(rng, "bytes */") {
		t.interruptCount += 1
		if t.interruptCount%7 == 0 { // Respond with a "service unavailable" error
			res := &http.Response{
				StatusCode: http.StatusServiceUnavailable,
				Header:     http.Header{},
			}
			t.rangeVal = fmt.Sprintf("bytes=0-%v", len(t.buf)-1) // Set the response for next time
			return res, nil
		}
		m := contentRangeRE.FindStringSubmatch(rng)
		if len(m) != 4 {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		start, err := strconv.ParseInt(m[1], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		end, err := strconv.ParseInt(m[2], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		totalSize, err := strconv.ParseInt(m[3], 10, 64)
		if err != nil {
			return nil, fmt.Errorf("unable to parse content range: %v", rng)
		}
		partialSize := end - start + 1
		buf, err := ioutil.ReadAll(req.Body)
		if err != nil || int64(len(buf)) != partialSize {
			return nil, fmt.Errorf("unable to read %v bytes from request data, n=%v: %v", partialSize, len(buf), err)
		}
		t.buf = append(t.buf, buf...)
		if totalSize == end+1 {
			t.statusCode = 200 // signify completion of transfer
		}
	}
	f := ioutil.NopCloser(unexpectedReader{})
	res := &http.Response{
		Body:       f,
		StatusCode: t.statusCode,
		Header:     http.Header{},
	}
	if t.rangeVal != "" {
		res.Header.Set("Range", t.rangeVal)
	}
	return res, nil
}

type interruptedTransport struct {
	req            *http.Request
	statusCode     int
	rangeVal       string
	interruptCount int
	buf            []byte
	progressBuf    string
}

func (tr *interruptedTransport) ProgressUpdate(current, total int64) {
	tr.progressBuf += fmt.Sprintf("%v, %v\n", current, total)
}

func TestInterruptedTransferChunks(t *testing.T) {
	f, err := os.Open("googleapi.go")
	if err != nil {
		t.Fatalf("unable to open googleapi.go: %v", err)
	}
	defer f.Close()
	slurp, err := ioutil.ReadAll(f)
	if err != nil {
		t.Fatalf("unable to slurp file: %v", err)
	}
	st, err := f.Stat()
	if err != nil {
		t.Fatalf("unable to stat googleapi.go: %v", err)
	}
	tr := &interruptedTransport{
		statusCode: 308,
		buf:        make([]byte, 0, st.Size()),
	}
	oldChunkSize := chunkSize
	defer func() { chunkSize = oldChunkSize }()
	chunkSize = 100 // override to process small chunks for test.

	sleep = func(time.Duration) {} // override time.Sleep
	rx := &ResumableUpload{
		Client:        &http.Client{Transport: tr},
		Media:         f,
		MediaType:     "text/plain",
		ContentLength: st.Size(),
		Callback:      tr.ProgressUpdate,
	}
	res, err := rx.Upload(context.Background())
	if err != nil || res == nil || res.StatusCode != http.StatusOK {
		if res == nil {
			t.Errorf("transferChunks not successful, res=nil: %v", err)
		} else {
			t.Errorf("transferChunks not successful, statusCode=%v: %v", res.StatusCode, err)
		}
	}
	if len(tr.buf) != len(slurp) || bytes.Compare(tr.buf, slurp) != 0 {
		t.Errorf("transfered file corrupted:\ngot %s\nwant %s", tr.buf, slurp)
	}
	w := ""
	for i := chunkSize; i <= st.Size(); i += chunkSize {
		w += fmt.Sprintf("%v, %v\n", i, st.Size())
	}
	if st.Size()%chunkSize != 0 {
		w += fmt.Sprintf("%v, %v\n", st.Size(), st.Size())
	}
	if tr.progressBuf != w {
		t.Errorf("progress update error, got %v, want %v", tr.progressBuf, w)
	}
}

func TestCancelUpload(t *testing.T) {
	f, err := os.Open("googleapi.go")
	if err != nil {
		t.Fatalf("unable to open googleapi.go: %v", err)
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		t.Fatalf("unable to stat googleapi.go: %v", err)
	}
	tr := &interruptedTransport{
		statusCode: 308,
		buf:        make([]byte, 0, st.Size()),
	}
	oldChunkSize := chunkSize
	defer func() { chunkSize = oldChunkSize }()
	chunkSize = 100 // override to process small chunks for test.

	sleep = func(time.Duration) {} // override time.Sleep
	rx := &ResumableUpload{
		Client:        &http.Client{Transport: tr},
		Media:         f,
		MediaType:     "text/plain",
		ContentLength: st.Size(),
		Callback:      tr.ProgressUpdate,
	}
	ctx, cancelFunc := context.WithCancel(context.Background())
	cancelFunc() // stop the upload that hasn't started yet
	res, err := rx.Upload(ctx)
	if err == nil || res == nil || res.StatusCode != http.StatusRequestTimeout {
		if res == nil {
			t.Errorf("transferChunks not successful, got res=nil, err=%v, want StatusRequestTimeout", err)
		} else {
			t.Errorf("transferChunks not successful, got statusCode=%v, err=%v, want StatusRequestTimeout", res.StatusCode, err)
		}
	}
}
