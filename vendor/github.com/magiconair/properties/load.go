// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"net/http"
	"os"
	"strings"
)

// Encoding specifies encoding of the input data.
type Encoding uint

const (
	// UTF8 interprets the input data as UTF-8.
	UTF8 Encoding = 1 << iota

	// ISO_8859_1 interprets the input data as ISO-8859-1.
	ISO_8859_1
)

// Load reads a buffer into a Properties struct.
func Load(buf []byte, enc Encoding) (*Properties, error) {
	return loadBuf(buf, enc)
}

// LoadString reads an UTF8 string into a properties struct.
func LoadString(s string) (*Properties, error) {
	return loadBuf([]byte(s), UTF8)
}

// LoadFile reads a file into a Properties struct.
func LoadFile(filename string, enc Encoding) (*Properties, error) {
	return loadFiles([]string{filename}, enc, false)
}

// LoadFiles reads multiple files in the given order into
// a Properties struct. If 'ignoreMissing' is true then
// non-existent files will not be reported as error.
func LoadFiles(filenames []string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	return loadFiles(filenames, enc, ignoreMissing)
}

// LoadURL reads the content of the URL into a Properties struct.
//
// The encoding is determined via the Content-Type header which
// should be set to 'text/plain'. If the 'charset' parameter is
// missing, 'iso-8859-1' or 'latin1' the encoding is set to
// ISO-8859-1. If the 'charset' parameter is set to 'utf-8' the
// encoding is set to UTF-8. A missing content type header is
// interpreted as 'text/plain; charset=utf-8'.
func LoadURL(url string) (*Properties, error) {
	return loadURLs([]string{url}, false)
}

// LoadURLs reads the content of multiple URLs in the given order into a
// Properties struct. If 'ignoreMissing' is true then a 404 status code will
// not be reported as error. See LoadURL for the Content-Type header
// and the encoding.
func LoadURLs(urls []string, ignoreMissing bool) (*Properties, error) {
	return loadURLs(urls, ignoreMissing)
}

// MustLoadString reads an UTF8 string into a Properties struct and
// panics on error.
func MustLoadString(s string) *Properties {
	return must(LoadString(s))
}

// MustLoadFile reads a file into a Properties struct and
// panics on error.
func MustLoadFile(filename string, enc Encoding) *Properties {
	return must(LoadFile(filename, enc))
}

// MustLoadFiles reads multiple files in the given order into
// a Properties struct and panics on error. If 'ignoreMissing'
// is true then non-existent files will not be reported as error.
func MustLoadFiles(filenames []string, enc Encoding, ignoreMissing bool) *Properties {
	return must(LoadFiles(filenames, enc, ignoreMissing))
}

// MustLoadURL reads the content of a URL into a Properties struct and
// panics on error.
func MustLoadURL(url string) *Properties {
	return must(LoadURL(url))
}

// MustLoadFiles reads the content of multiple URLs in the given order into a
// Properties struct and panics on error. If 'ignoreMissing' is true then a 404
// status code will not be reported as error.
func MustLoadURLs(urls []string, ignoreMissing bool) *Properties {
	return must(LoadURLs(urls, ignoreMissing))
}

func loadBuf(buf []byte, enc Encoding) (*Properties, error) {
	p, err := parse(convert(buf, enc))
	if err != nil {
		return nil, err
	}
	return p, p.check()
}

func loadFiles(filenames []string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	var buf bytes.Buffer
	for _, filename := range filenames {
		f, err := expandFilename(filename)
		if err != nil {
			return nil, err
		}

		data, err := ioutil.ReadFile(f)
		if err != nil {
			if ignoreMissing && os.IsNotExist(err) {
				LogPrintf("properties: %s not found. skipping", filename)
				continue
			}
			return nil, err
		}

		// concatenate the buffers and add a new line in case
		// the previous file didn't end with a new line
		buf.Write(data)
		buf.WriteRune('\n')
	}
	return loadBuf(buf.Bytes(), enc)
}

func loadURLs(urls []string, ignoreMissing bool) (*Properties, error) {
	var buf bytes.Buffer
	for _, u := range urls {
		resp, err := http.Get(u)
		if err != nil {
			return nil, fmt.Errorf("properties: error fetching %q. %s", u, err)
		}
		if resp.StatusCode == 404 && ignoreMissing {
			LogPrintf("properties: %s returned %d. skipping", u, resp.StatusCode)
			continue
		}
		if resp.StatusCode != 200 {
			return nil, fmt.Errorf("properties: %s returned %d", u, resp.StatusCode)
		}
		body, err := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		if err != nil {
			return nil, fmt.Errorf("properties: %s error reading response. %s", u, err)
		}

		ct := resp.Header.Get("Content-Type")
		var enc Encoding
		switch strings.ToLower(ct) {
		case "text/plain", "text/plain; charset=iso-8859-1", "text/plain; charset=latin1":
			enc = ISO_8859_1
		case "", "text/plain; charset=utf-8":
			enc = UTF8
		default:
			return nil, fmt.Errorf("properties: invalid content type %s", ct)
		}

		buf.WriteString(convert(body, enc))
		buf.WriteRune('\n')
	}
	return loadBuf(buf.Bytes(), UTF8)
}

func must(p *Properties, err error) *Properties {
	if err != nil {
		ErrorHandler(err)
	}
	return p
}

// expandFilename expands ${ENV_VAR} expressions in a filename.
// If the environment variable does not exist then it will be replaced
// with an empty string. Malformed expressions like "${ENV_VAR" will
// be reported as error.
func expandFilename(filename string) (string, error) {
	return expand(filename, make(map[string]bool), "${", "}", make(map[string]string))
}

// Interprets a byte buffer either as an ISO-8859-1 or UTF-8 encoded string.
// For ISO-8859-1 we can convert each byte straight into a rune since the
// first 256 unicode code points cover ISO-8859-1.
func convert(buf []byte, enc Encoding) string {
	switch enc {
	case UTF8:
		return string(buf)
	case ISO_8859_1:
		runes := make([]rune, len(buf))
		for i, b := range buf {
			runes[i] = rune(b)
		}
		return string(runes)
	default:
		ErrorHandler(fmt.Errorf("unsupported encoding %v", enc))
	}
	panic("ErrorHandler should exit")
}
