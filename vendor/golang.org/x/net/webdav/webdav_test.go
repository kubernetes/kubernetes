// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package webdav

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"net/url"
	"os"
	"reflect"
	"regexp"
	"sort"
	"strings"
	"testing"

	"golang.org/x/net/context"
)

// TODO: add tests to check XML responses with the expected prefix path
func TestPrefix(t *testing.T) {
	const dst, blah = "Destination", "blah blah blah"

	// createLockBody comes from the example in Section 9.10.7.
	const createLockBody = `<?xml version="1.0" encoding="utf-8" ?>
		<D:lockinfo xmlns:D='DAV:'>
			<D:lockscope><D:exclusive/></D:lockscope>
			<D:locktype><D:write/></D:locktype>
			<D:owner>
				<D:href>http://example.org/~ejw/contact.html</D:href>
			</D:owner>
		</D:lockinfo>
	`

	do := func(method, urlStr string, body string, wantStatusCode int, headers ...string) (http.Header, error) {
		var bodyReader io.Reader
		if body != "" {
			bodyReader = strings.NewReader(body)
		}
		req, err := http.NewRequest(method, urlStr, bodyReader)
		if err != nil {
			return nil, err
		}
		for len(headers) >= 2 {
			req.Header.Add(headers[0], headers[1])
			headers = headers[2:]
		}
		res, err := http.DefaultTransport.RoundTrip(req)
		if err != nil {
			return nil, err
		}
		defer res.Body.Close()
		if res.StatusCode != wantStatusCode {
			return nil, fmt.Errorf("got status code %d, want %d", res.StatusCode, wantStatusCode)
		}
		return res.Header, nil
	}

	prefixes := []string{
		"/",
		"/a/",
		"/a/b/",
		"/a/b/c/",
	}
	ctx := context.Background()
	for _, prefix := range prefixes {
		fs := NewMemFS()
		h := &Handler{
			FileSystem: fs,
			LockSystem: NewMemLS(),
		}
		mux := http.NewServeMux()
		if prefix != "/" {
			h.Prefix = prefix
		}
		mux.Handle(prefix, h)
		srv := httptest.NewServer(mux)
		defer srv.Close()

		// The script is:
		//	MKCOL /a
		//	MKCOL /a/b
		//	PUT   /a/b/c
		//	COPY  /a/b/c /a/b/d
		//	MKCOL /a/b/e
		//	MOVE  /a/b/d /a/b/e/f
		//	LOCK  /a/b/e/g
		//	PUT   /a/b/e/g
		// which should yield the (possibly stripped) filenames /a/b/c,
		// /a/b/e/f and /a/b/e/g, plus their parent directories.

		wantA := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusMovedPermanently,
			"/a/b/":   http.StatusNotFound,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if _, err := do("MKCOL", srv.URL+"/a", "", wantA); err != nil {
			t.Errorf("prefix=%-9q MKCOL /a: %v", prefix, err)
			continue
		}

		wantB := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusMovedPermanently,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if _, err := do("MKCOL", srv.URL+"/a/b", "", wantB); err != nil {
			t.Errorf("prefix=%-9q MKCOL /a/b: %v", prefix, err)
			continue
		}

		wantC := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusMovedPermanently,
		}[prefix]
		if _, err := do("PUT", srv.URL+"/a/b/c", blah, wantC); err != nil {
			t.Errorf("prefix=%-9q PUT /a/b/c: %v", prefix, err)
			continue
		}

		wantD := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusMovedPermanently,
		}[prefix]
		if _, err := do("COPY", srv.URL+"/a/b/c", "", wantD, dst, srv.URL+"/a/b/d"); err != nil {
			t.Errorf("prefix=%-9q COPY /a/b/c /a/b/d: %v", prefix, err)
			continue
		}

		wantE := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if _, err := do("MKCOL", srv.URL+"/a/b/e", "", wantE); err != nil {
			t.Errorf("prefix=%-9q MKCOL /a/b/e: %v", prefix, err)
			continue
		}

		wantF := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if _, err := do("MOVE", srv.URL+"/a/b/d", "", wantF, dst, srv.URL+"/a/b/e/f"); err != nil {
			t.Errorf("prefix=%-9q MOVE /a/b/d /a/b/e/f: %v", prefix, err)
			continue
		}

		var lockToken string
		wantG := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if h, err := do("LOCK", srv.URL+"/a/b/e/g", createLockBody, wantG); err != nil {
			t.Errorf("prefix=%-9q LOCK /a/b/e/g: %v", prefix, err)
			continue
		} else {
			lockToken = h.Get("Lock-Token")
		}

		ifHeader := fmt.Sprintf("<%s/a/b/e/g> (%s)", srv.URL, lockToken)
		wantH := map[string]int{
			"/":       http.StatusCreated,
			"/a/":     http.StatusCreated,
			"/a/b/":   http.StatusCreated,
			"/a/b/c/": http.StatusNotFound,
		}[prefix]
		if _, err := do("PUT", srv.URL+"/a/b/e/g", blah, wantH, "If", ifHeader); err != nil {
			t.Errorf("prefix=%-9q PUT /a/b/e/g: %v", prefix, err)
			continue
		}

		got, err := find(ctx, nil, fs, "/")
		if err != nil {
			t.Errorf("prefix=%-9q find: %v", prefix, err)
			continue
		}
		sort.Strings(got)
		want := map[string][]string{
			"/":       {"/", "/a", "/a/b", "/a/b/c", "/a/b/e", "/a/b/e/f", "/a/b/e/g"},
			"/a/":     {"/", "/b", "/b/c", "/b/e", "/b/e/f", "/b/e/g"},
			"/a/b/":   {"/", "/c", "/e", "/e/f", "/e/g"},
			"/a/b/c/": {"/"},
		}[prefix]
		if !reflect.DeepEqual(got, want) {
			t.Errorf("prefix=%-9q find:\ngot  %v\nwant %v", prefix, got, want)
			continue
		}
	}
}

func TestEscapeXML(t *testing.T) {
	// These test cases aren't exhaustive, and there is more than one way to
	// escape e.g. a quot (as "&#34;" or "&quot;") or an apos. We presume that
	// the encoding/xml package tests xml.EscapeText more thoroughly. This test
	// here is just a sanity check for this package's escapeXML function, and
	// its attempt to provide a fast path (and avoid a bytes.Buffer allocation)
	// when escaping filenames is obviously a no-op.
	testCases := map[string]string{
		"":              "",
		" ":             " ",
		"&":             "&amp;",
		"*":             "*",
		"+":             "+",
		",":             ",",
		"-":             "-",
		".":             ".",
		"/":             "/",
		"0":             "0",
		"9":             "9",
		":":             ":",
		"<":             "&lt;",
		">":             "&gt;",
		"A":             "A",
		"_":             "_",
		"a":             "a",
		"~":             "~",
		"\u0201":        "\u0201",
		"&amp;":         "&amp;amp;",
		"foo&<b/ar>baz": "foo&amp;&lt;b/ar&gt;baz",
	}

	for in, want := range testCases {
		if got := escapeXML(in); got != want {
			t.Errorf("in=%q: got %q, want %q", in, got, want)
		}
	}
}

func TestFilenameEscape(t *testing.T) {
	hrefRe := regexp.MustCompile(`<D:href>([^<]*)</D:href>`)
	displayNameRe := regexp.MustCompile(`<D:displayname>([^<]*)</D:displayname>`)
	do := func(method, urlStr string) (string, string, error) {
		req, err := http.NewRequest(method, urlStr, nil)
		if err != nil {
			return "", "", err
		}
		res, err := http.DefaultClient.Do(req)
		if err != nil {
			return "", "", err
		}
		defer res.Body.Close()

		b, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return "", "", err
		}
		hrefMatch := hrefRe.FindStringSubmatch(string(b))
		if len(hrefMatch) != 2 {
			return "", "", errors.New("D:href not found")
		}
		displayNameMatch := displayNameRe.FindStringSubmatch(string(b))
		if len(displayNameMatch) != 2 {
			return "", "", errors.New("D:displayname not found")
		}

		return hrefMatch[1], displayNameMatch[1], nil
	}

	testCases := []struct {
		name, wantHref, wantDisplayName string
	}{{
		name:            `/foo%bar`,
		wantHref:        `/foo%25bar`,
		wantDisplayName: `foo%bar`,
	}, {
		name:            `/こんにちわ世界`,
		wantHref:        `/%E3%81%93%E3%82%93%E3%81%AB%E3%81%A1%E3%82%8F%E4%B8%96%E7%95%8C`,
		wantDisplayName: `こんにちわ世界`,
	}, {
		name:            `/Program Files/`,
		wantHref:        `/Program%20Files`,
		wantDisplayName: `Program Files`,
	}, {
		name:            `/go+lang`,
		wantHref:        `/go+lang`,
		wantDisplayName: `go+lang`,
	}, {
		name:            `/go&lang`,
		wantHref:        `/go&amp;lang`,
		wantDisplayName: `go&amp;lang`,
	}, {
		name:            `/go<lang`,
		wantHref:        `/go%3Clang`,
		wantDisplayName: `go&lt;lang`,
	}}
	ctx := context.Background()
	fs := NewMemFS()
	for _, tc := range testCases {
		if strings.HasSuffix(tc.name, "/") {
			if err := fs.Mkdir(ctx, tc.name, 0755); err != nil {
				t.Fatalf("name=%q: Mkdir: %v", tc.name, err)
			}
		} else {
			f, err := fs.OpenFile(ctx, tc.name, os.O_CREATE, 0644)
			if err != nil {
				t.Fatalf("name=%q: OpenFile: %v", tc.name, err)
			}
			f.Close()
		}
	}

	srv := httptest.NewServer(&Handler{
		FileSystem: fs,
		LockSystem: NewMemLS(),
	})
	defer srv.Close()

	u, err := url.Parse(srv.URL)
	if err != nil {
		t.Fatal(err)
	}

	for _, tc := range testCases {
		u.Path = tc.name
		gotHref, gotDisplayName, err := do("PROPFIND", u.String())
		if err != nil {
			t.Errorf("name=%q: PROPFIND: %v", tc.name, err)
			continue
		}
		if gotHref != tc.wantHref {
			t.Errorf("name=%q: got href %q, want %q", tc.name, gotHref, tc.wantHref)
		}
		if gotDisplayName != tc.wantDisplayName {
			t.Errorf("name=%q: got dispayname %q, want %q", tc.name, gotDisplayName, tc.wantDisplayName)
		}
	}
}
