// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"fmt"
	"io/ioutil"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"

	. "github.com/magiconair/properties/_third_party/gopkg.in/check.v1"
)

type LoadSuite struct {
	tempFiles []string
}

var _ = Suite(&LoadSuite{})

func (s *LoadSuite) TestLoadFailsWithNotExistingFile(c *C) {
	_, err := LoadFile("doesnotexist.properties", ISO_8859_1)
	c.Assert(err, NotNil)
	c.Assert(err, ErrorMatches, "open.*no such file or directory")
}

func (s *LoadSuite) TestLoadFilesFailsOnNotExistingFile(c *C) {
	_, err := LoadFiles([]string{"doesnotexist.properties"}, ISO_8859_1, false)
	c.Assert(err, NotNil)
	c.Assert(err, ErrorMatches, "open.*no such file or directory")
}

func (s *LoadSuite) TestLoadFilesDoesNotFailOnNotExistingFileAndIgnoreMissing(c *C) {
	p, err := LoadFiles([]string{"doesnotexist.properties"}, ISO_8859_1, true)
	c.Assert(err, IsNil)
	c.Assert(p.Len(), Equals, 0)
}

func (s *LoadSuite) TestLoadString(c *C) {
	x := "key=äüö"
	p1 := MustLoadString(x)
	p2 := must(Load([]byte(x), UTF8))
	c.Assert(p1, DeepEquals, p2)
}

func (s *LoadSuite) TestLoadFile(c *C) {
	filename := s.makeFile(c, "key=value")
	p := MustLoadFile(filename, ISO_8859_1)

	c.Assert(p.Len(), Equals, 1)
	assertKeyValues(c, "", p, "key", "value")
}

func (s *LoadSuite) TestLoadFiles(c *C) {
	filename := s.makeFile(c, "key=value")
	filename2 := s.makeFile(c, "key2=value2")
	p := MustLoadFiles([]string{filename, filename2}, ISO_8859_1, false)
	assertKeyValues(c, "", p, "key", "value", "key2", "value2")
}

func (s *LoadSuite) TestLoadExpandedFile(c *C) {
	os.Setenv("_VARX", "some-value")
	filename := s.makeFilePrefix(c, os.Getenv("_VARX"), "key=value")
	filename = strings.Replace(filename, os.Getenv("_VARX"), "${_VARX}", -1)
	p := MustLoadFile(filename, ISO_8859_1)
	assertKeyValues(c, "", p, "key", "value")
}

func (s *LoadSuite) TestLoadFilesAndIgnoreMissing(c *C) {
	filename := s.makeFile(c, "key=value")
	filename2 := s.makeFile(c, "key2=value2")
	p := MustLoadFiles([]string{filename, filename + "foo", filename2, filename2 + "foo"}, ISO_8859_1, true)
	assertKeyValues(c, "", p, "key", "value", "key2", "value2")
}

func (s *LoadSuite) TestLoadURL(c *C) {
	srv := testServer()
	defer srv.Close()
	p := MustLoadURL(srv.URL + "/a")
	assertKeyValues(c, "", p, "key", "value")
}

func (s *LoadSuite) TestLoadURLs(c *C) {
	srv := testServer()
	defer srv.Close()
	p := MustLoadURLs([]string{srv.URL + "/a", srv.URL + "/b"}, false)
	assertKeyValues(c, "", p, "key", "value", "key2", "value2")
}

func (s *LoadSuite) TestLoadURLsAndFailMissing(c *C) {
	srv := testServer()
	defer srv.Close()
	p, err := LoadURLs([]string{srv.URL + "/a", srv.URL + "/c"}, false)
	c.Assert(p, IsNil)
	c.Assert(err, ErrorMatches, ".*returned 404.*")
}

func (s *LoadSuite) TestLoadURLsAndIgnoreMissing(c *C) {
	srv := testServer()
	defer srv.Close()
	p := MustLoadURLs([]string{srv.URL + "/a", srv.URL + "/b", srv.URL + "/c"}, true)
	assertKeyValues(c, "", p, "key", "value", "key2", "value2")
}

func (s *LoadSuite) TestLoadURLEncoding(c *C) {
	srv := testServer()
	defer srv.Close()

	uris := []string{"/none", "/utf8", "/plain", "/latin1", "/iso88591"}
	for i, uri := range uris {
		p := MustLoadURL(srv.URL + uri)
		c.Assert(p.GetString("key", ""), Equals, "äöü", Commentf("%d", i))
	}
}

func (s *LoadSuite) TestLoadURLFailInvalidEncoding(c *C) {
	srv := testServer()
	defer srv.Close()

	p, err := LoadURL(srv.URL + "/json")
	c.Assert(p, IsNil)
	c.Assert(err, ErrorMatches, ".*invalid content type.*")
}

func (s *LoadSuite) TestLoadAll(c *C) {
	filename := s.makeFile(c, "key=value")
	filename2 := s.makeFile(c, "key2=value3")
	filename3 := s.makeFile(c, "key=value4")
	srv := testServer()
	defer srv.Close()
	p := MustLoadAll([]string{filename, filename2, srv.URL + "/a", srv.URL + "/b", filename3}, UTF8, false)
	assertKeyValues(c, "", p, "key", "value4", "key2", "value2")
}

func (s *LoadSuite) SetUpSuite(c *C) {
	s.tempFiles = make([]string, 0)
}

func (s *LoadSuite) TearDownSuite(c *C) {
	for _, path := range s.tempFiles {
		err := os.Remove(path)
		if err != nil {
			fmt.Printf("os.Remove: %v", err)
		}
	}
}

func (s *LoadSuite) makeFile(c *C, data string) string {
	return s.makeFilePrefix(c, "properties", data)
}

func (s *LoadSuite) makeFilePrefix(c *C, prefix, data string) string {
	f, err := ioutil.TempFile("", prefix)
	if err != nil {
		fmt.Printf("ioutil.TempFile: %v", err)
		c.FailNow()
	}

	// remember the temp file so that we can remove it later
	s.tempFiles = append(s.tempFiles, f.Name())

	n, err := fmt.Fprint(f, data)
	if err != nil {
		fmt.Printf("fmt.Fprintln: %v", err)
		c.FailNow()
	}
	if n != len(data) {
		fmt.Printf("Data size mismatch. expected=%d wrote=%d\n", len(data), n)
		c.FailNow()
	}

	err = f.Close()
	if err != nil {
		fmt.Printf("f.Close: %v", err)
		c.FailNow()
	}

	return f.Name()
}

func testServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		send := func(data []byte, contentType string) {
			w.Header().Set("Content-Type", contentType)
			w.Write(data)
		}

		utf8 := []byte("key=äöü")
		iso88591 := []byte{0x6b, 0x65, 0x79, 0x3d, 0xe4, 0xf6, 0xfc} // key=äöü

		switch r.RequestURI {
		case "/a":
			send([]byte("key=value"), "")
		case "/b":
			send([]byte("key2=value2"), "")
		case "/none":
			send(utf8, "")
		case "/utf8":
			send(utf8, "text/plain; charset=utf-8")
		case "/json":
			send(utf8, "application/json; charset=utf-8")
		case "/plain":
			send(iso88591, "text/plain")
		case "/latin1":
			send(iso88591, "text/plain; charset=latin1")
		case "/iso88591":
			send(iso88591, "text/plain; charset=iso-8859-1")
		default:
			w.WriteHeader(404)
		}
	}))
}
