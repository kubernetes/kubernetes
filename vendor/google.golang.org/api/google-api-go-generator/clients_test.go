// Copyright 2017 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"strings"
	"testing"
	"testing/iotest"

	"golang.org/x/net/context"

	// If you add a client, add a matching go:generate line below.
	dfa "google.golang.org/api/dfareporting/v2.7"
	mon "google.golang.org/api/monitoring/v3"
	storage "google.golang.org/api/storage/v1"
)

//go:generate -command api go run gen.go docurls.go -install -api

//go:generate api dfareporting:v2.7
//go:generate api monitoring:v3
//go:generate api storage:v1

type myHandler struct {
	location string
	r        *http.Request
	body     []byte
	reqURIs  []string
	err      error
}

func (h *myHandler) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	h.r = r
	v, err := url.ParseRequestURI(r.URL.RequestURI())
	if err != nil {
		h.err = err
		return
	}
	h.reqURIs = append(h.reqURIs, v.String())
	if h.location != "" {
		w.Header().Set("Location", h.location)
	}
	h.body, h.err = ioutil.ReadAll(r.Body)
	fmt.Fprintf(w, "{}")
}

func TestMedia(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL

	const body = "fake media data"
	f := strings.NewReader(body)
	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Media(f).Do()
	if err != nil {
		t.Fatalf("unable to insert object: %v", err)
	}
	g := handler.r
	if w := "POST"; g.Method != w {
		t.Errorf("Method = %q; want %q", g.Method, w)
	}
	if w := "HTTP/1.1"; g.Proto != w {
		t.Errorf("Proto = %q; want %q", g.Proto, w)
	}
	if w := 1; g.ProtoMajor != w {
		t.Errorf("ProtoMajor = %v; want %v", g.ProtoMajor, w)
	}
	if w := 1; g.ProtoMinor != w {
		t.Errorf("ProtoMinor = %v; want %v", g.ProtoMinor, w)
	}
	if w, k := "google-api-go-client/0.5", "User-Agent"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w, k := "multipart/related; boundary=", "Content-Type"; len(g.Header[k]) != 1 || !strings.HasPrefix(g.Header[k][0], w) {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w, k := "gzip", "Accept-Encoding"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w := int64(-1); g.ContentLength != w {
		t.Errorf("ContentLength = %v; want %v", g.ContentLength, w)
	}
	if w := "chunked"; len(g.TransferEncoding) != 1 || g.TransferEncoding[0] != w {
		t.Errorf("TransferEncoding = %#v; want %q", g.TransferEncoding, w)
	}
	if w := strings.TrimPrefix(s.BasePath, "http://"); g.Host != w {
		t.Errorf("Host = %q; want %q", g.Host, w)
	}
	if g.Form != nil {
		t.Errorf("Form = %#v; want nil", g.Form)
	}
	if g.PostForm != nil {
		t.Errorf("PostForm = %#v; want nil", g.PostForm)
	}
	if g.MultipartForm != nil {
		t.Errorf("MultipartForm = %#v; want nil", g.MultipartForm)
	}
	if w := "/b/mybucket/o?alt=json&uploadType=multipart"; g.RequestURI != w {
		t.Errorf("RequestURI = %q; want %q", g.RequestURI, w)
	}
	if w := "\r\n\r\n" + body + "\r\n"; !strings.Contains(string(handler.body), w) {
		t.Errorf("Body = %q, want substring %q", handler.body, w)
	}
	if handler.err != nil {
		t.Errorf("handler err = %v, want nil", handler.err)
	}
}

func TestResumableMedia(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	handler.location = server.URL
	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL

	const data = "fake resumable media data"
	mediaSize := len(data)
	f := strings.NewReader(data)
	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Name("filename").ResumableMedia(context.Background(), f, int64(len(data)), "text/plain").Do()
	if err != nil {
		t.Fatalf("unable to insert object: %v", err)
	}
	g := handler.r
	if w := "POST"; g.Method != w {
		t.Errorf("Method = %q; want %q", g.Method, w)
	}
	if w := "HTTP/1.1"; g.Proto != w {
		t.Errorf("Proto = %q; want %q", g.Proto, w)
	}
	if w := 1; g.ProtoMajor != w {
		t.Errorf("ProtoMajor = %v; want %v", g.ProtoMajor, w)
	}
	if w := 1; g.ProtoMinor != w {
		t.Errorf("ProtoMinor = %v; want %v", g.ProtoMinor, w)
	}
	if w, k := "google-api-go-client/0.5", "User-Agent"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if want, got := []string{"text/plain"}, g.Header["Content-Type"]; !reflect.DeepEqual(got, want) {
		t.Errorf("header Content-Type got: %#v; want: %#v", got, want)
	}
	if w, k := "gzip", "Accept-Encoding"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w := int64(mediaSize); g.ContentLength != w {
		t.Errorf("ContentLength = %v; want %v", g.ContentLength, w)
	}
	if len(g.TransferEncoding) != 0 {
		t.Errorf("TransferEncoding = %#v; want nil", g.TransferEncoding)
	}
	if g.Form != nil {
		t.Errorf("Form = %#v; want nil", g.Form)
	}
	if g.PostForm != nil {
		t.Errorf("PostForm = %#v; want nil", g.PostForm)
	}
	if g.MultipartForm != nil {
		t.Errorf("MultipartForm = %#v; want nil", g.MultipartForm)
	}
	if handler.err != nil {
		t.Errorf("handler err = %v, want nil", handler.err)
	}
}

func TestNoMedia(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL

	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Do()
	if err != nil {
		t.Fatalf("unable to insert object: %v", err)
	}
	g := handler.r
	if w := "POST"; g.Method != w {
		t.Errorf("Method = %q; want %q", g.Method, w)
	}
	if w := "HTTP/1.1"; g.Proto != w {
		t.Errorf("Proto = %q; want %q", g.Proto, w)
	}
	if w := 1; g.ProtoMajor != w {
		t.Errorf("ProtoMajor = %v; want %v", g.ProtoMajor, w)
	}
	if w := 1; g.ProtoMinor != w {
		t.Errorf("ProtoMinor = %v; want %v", g.ProtoMinor, w)
	}
	if w, k := "google-api-go-client/0.5", "User-Agent"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w, k := "application/json", "Content-Type"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w, k := "gzip", "Accept-Encoding"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
	if w := int64(116); g.ContentLength != w {
		t.Errorf("ContentLength = %v; want %v", g.ContentLength, w)
	}
	if len(g.TransferEncoding) != 0 {
		t.Errorf("TransferEncoding = %#v; want []string{}", g.TransferEncoding)
	}
	if w := strings.TrimPrefix(s.BasePath, "http://"); g.Host != w {
		t.Errorf("Host = %q; want %q", g.Host, w)
	}
	if g.Form != nil {
		t.Errorf("Form = %#v; want nil", g.Form)
	}
	if g.PostForm != nil {
		t.Errorf("PostForm = %#v; want nil", g.PostForm)
	}
	if g.MultipartForm != nil {
		t.Errorf("MultipartForm = %#v; want nil", g.MultipartForm)
	}
	if w := "/b/mybucket/o?alt=json"; g.RequestURI != w {
		t.Errorf("RequestURI = %q; want %q", g.RequestURI, w)
	}
	if w := `{"bucket":"mybucket","contentEncoding":"utf-8","contentLanguage":"en","contentType":"plain/text","name":"filename"}` + "\n"; string(handler.body) != w {
		t.Errorf("Body = %q, want %q", handler.body, w)
	}
	if handler.err != nil {
		t.Errorf("handler err = %v, want nil", handler.err)
	}
}

func TestMediaErrHandling(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL

	const body = "fake media data"
	f := strings.NewReader(body)
	// The combination of TimeoutReader and OneByteReader causes the first byte to
	// be successfully delivered, but then a timeout error is reported.
	r := iotest.TimeoutReader(iotest.OneByteReader(f))
	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Media(r).Do()
	if err == nil || !strings.Contains(err.Error(), "timeout") {
		t.Errorf("expected timeout error, got %v", err)
	}
	if handler.err != nil {
		t.Errorf("handler err = %v, want nil", handler.err)
	}
}

func TestUserAgent(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL
	s.UserAgent = "myagent/1.0"

	f := strings.NewReader("fake media data")
	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Media(f).Do()
	if err != nil {
		t.Fatalf("unable to insert object: %v", err)
	}
	g := handler.r
	if w, k := "google-api-go-client/0.5 myagent/1.0", "User-Agent"; len(g.Header[k]) != 1 || g.Header[k][0] != w {
		t.Errorf("header %q = %#v; want %q", k, g.Header[k], w)
	}
}

func myProgressUpdater(current, total int64) {}

func TestParams(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	handler.location = server.URL + "/uploadURL"
	client := &http.Client{}
	s, err := storage.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL
	s.UserAgent = "myagent/1.0"

	const data = "fake media data"
	f := strings.NewReader(data)
	o := &storage.Object{
		Bucket:          "mybucket",
		Name:            "filename",
		ContentType:     "plain/text",
		ContentEncoding: "utf-8",
		ContentLanguage: "en",
	}
	_, err = s.Objects.Insert("mybucket", o).Name(o.Name).IfGenerationMatch(42).ResumableMedia(context.Background(), f, int64(len(data)), "plain/text").ProgressUpdater(myProgressUpdater).Projection("full").Do()
	if err != nil {
		t.Fatalf("unable to insert object: %v", err)
	}
	if g, w := len(handler.reqURIs), 2; g != w {
		t.Fatalf("len(reqURIs) = %v, want %v", g, w)
	}
	want := []string{
		"/b/mybucket/o?alt=json&ifGenerationMatch=42&name=filename&projection=full&uploadType=resumable",
		"/uploadURL",
	}
	if !reflect.DeepEqual(handler.reqURIs, want) {
		t.Errorf("reqURIs = %#v, want = %#v", handler.reqURIs, want)
	}
}

func TestRepeatedParams(t *testing.T) {
	handler := &myHandler{}
	server := httptest.NewServer(handler)
	defer server.Close()

	client := &http.Client{}
	s, err := dfa.New(client)
	if err != nil {
		t.Fatalf("unable to create service: %v", err)
	}
	s.BasePath = server.URL
	s.UserAgent = "myagent/1.0"
	cs := dfa.NewCreativesService(s)

	_, err = cs.List(10).Active(true).MaxResults(1).Ids(2, 3).PageToken("abc").SizeIds(4, 5).Types("def", "ghi").IfNoneMatch("not-a-param").Do()
	if err != nil {
		t.Fatalf("dfa.List: %v", err)
	}
	want := []string{
		"/userprofiles/10/creatives?active=true&alt=json&ids=2&ids=3&maxResults=1&pageToken=abc&sizeIds=4&sizeIds=5&types=def&types=ghi",
	}
	if !reflect.DeepEqual(handler.reqURIs, want) {
		t.Errorf("reqURIs = %#v, want = %#v", handler.reqURIs, want)
	}
}

// This test verifies that the unmarshal code generated for float64s
// (in this case, the one inside mon.TypedValue) compiles and
// behaves correctly.
func TestUnmarshalSpecialFloats(t *testing.T) {
	for _, test := range []struct {
		in   string
		want float64
	}{
		{`{"doubleValue": 3}`, 3},
		{`{"doubleValue": "Infinity"}`, math.Inf(1)},
		{`{"doubleValue": "-Infinity"}`, math.Inf(-1)},
		{`{"doubleValue": "NaN"}`, math.NaN()},
	} {
		var got mon.TypedValue
		if err := json.Unmarshal([]byte(test.in), &got); err != nil {
			t.Fatal(err)
		}
		if !fleq(*got.DoubleValue, test.want) {
			t.Errorf("got\n%+v\nwant\n%+v", *got.DoubleValue, test.want)
		}
	}
}

func fleq(f1, f2 float64) bool {
	return f1 == f2 || (math.IsNaN(f1) && math.IsNaN(f2))
}
