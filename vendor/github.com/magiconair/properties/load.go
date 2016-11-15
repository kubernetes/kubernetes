// Copyright 2016 Frank Schroeder. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package properties

import (
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
	return loadAll([]string{filename}, enc, false)
}

// LoadFiles reads multiple files in the given order into
// a Properties struct. If 'ignoreMissing' is true then
// non-existent files will not be reported as error.
func LoadFiles(filenames []string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	return loadAll(filenames, enc, ignoreMissing)
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
	return loadAll([]string{url}, UTF8, false)
}

// LoadURLs reads the content of multiple URLs in the given order into a
// Properties struct. If 'ignoreMissing' is true then a 404 status code will
// not be reported as error. See LoadURL for the Content-Type header
// and the encoding.
func LoadURLs(urls []string, ignoreMissing bool) (*Properties, error) {
	return loadAll(urls, UTF8, ignoreMissing)
}

// LoadAll reads the content of multiple URLs or files in the given order into a
// Properties struct. If 'ignoreMissing' is true then a 404 status code or missing file will
// not be reported as error. Encoding sets the encoding for files. For the URLs please see
// LoadURL for the Content-Type header and the encoding.
func LoadAll(names []string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	return loadAll(names, enc, ignoreMissing)
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

// MustLoadAll reads the content of multiple URLs or files in the given order into a
// Properties struct. If 'ignoreMissing' is true then a 404 status code or missing file will
// not be reported as error. Encoding sets the encoding for files. For the URLs please see
// LoadURL for the Content-Type header and the encoding. It panics on error.
func MustLoadAll(names []string, enc Encoding, ignoreMissing bool) *Properties {
	return must(LoadAll(names, enc, ignoreMissing))
}

func loadBuf(buf []byte, enc Encoding) (*Properties, error) {
	p, err := parse(convert(buf, enc))
	if err != nil {
		return nil, err
	}
	return p, p.check()
}

func loadAll(names []string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	result := NewProperties()
	for _, name := range names {
		n, err := expandName(name)
		if err != nil {
			return nil, err
		}
		var p *Properties
		if strings.HasPrefix(n, "http://") || strings.HasPrefix(n, "https://") {
			p, err = loadURL(n, ignoreMissing)
		} else {
			p, err = loadFile(n, enc, ignoreMissing)
		}
		if err != nil {
			return nil, err
		}
		result.Merge(p)

	}
	return result, result.check()
}

func loadFile(filename string, enc Encoding, ignoreMissing bool) (*Properties, error) {
	data, err := ioutil.ReadFile(filename)
	if err != nil {
		if ignoreMissing && os.IsNotExist(err) {
			LogPrintf("properties: %s not found. skipping", filename)
			return NewProperties(), nil
		}
		return nil, err
	}
	p, err := parse(convert(data, enc))
	if err != nil {
		return nil, err
	}
	return p, nil
}

func loadURL(url string, ignoreMissing bool) (*Properties, error) {
	resp, err := http.Get(url)
	if err != nil {
		return nil, fmt.Errorf("properties: error fetching %q. %s", url, err)
	}
	if resp.StatusCode == 404 && ignoreMissing {
		LogPrintf("properties: %s returned %d. skipping", url, resp.StatusCode)
		return NewProperties(), nil
	}
	if resp.StatusCode != 200 {
		return nil, fmt.Errorf("properties: %s returned %d", url, resp.StatusCode)
	}
	body, err := ioutil.ReadAll(resp.Body)
	resp.Body.Close()
	if err != nil {
		return nil, fmt.Errorf("properties: %s error reading response. %s", url, err)
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

	p, err := parse(convert(body, enc))
	if err != nil {
		return nil, err
	}
	return p, nil
}

func must(p *Properties, err error) *Properties {
	if err != nil {
		ErrorHandler(err)
	}
	return p
}

// expandName expands ${ENV_VAR} expressions in a name.
// If the environment variable does not exist then it will be replaced
// with an empty string. Malformed expressions like "${ENV_VAR" will
// be reported as error.
func expandName(name string) (string, error) {
	return expand(name, make(map[string]bool), "${", "}", make(map[string]string))
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
