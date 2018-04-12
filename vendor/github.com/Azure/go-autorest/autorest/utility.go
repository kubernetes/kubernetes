package autorest

// Copyright 2017 Microsoft Corporation
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.

import (
	"bytes"
	"encoding/json"
	"encoding/xml"
	"fmt"
	"io"
	"net/url"
	"reflect"
	"sort"
	"strings"
)

// EncodedAs is a series of constants specifying various data encodings
type EncodedAs string

const (
	// EncodedAsJSON states that data is encoded as JSON
	EncodedAsJSON EncodedAs = "JSON"

	// EncodedAsXML states that data is encoded as Xml
	EncodedAsXML EncodedAs = "XML"
)

// Decoder defines the decoding method json.Decoder and xml.Decoder share
type Decoder interface {
	Decode(v interface{}) error
}

// NewDecoder creates a new decoder appropriate to the passed encoding.
// encodedAs specifies the type of encoding and r supplies the io.Reader containing the
// encoded data.
func NewDecoder(encodedAs EncodedAs, r io.Reader) Decoder {
	if encodedAs == EncodedAsJSON {
		return json.NewDecoder(r)
	} else if encodedAs == EncodedAsXML {
		return xml.NewDecoder(r)
	}
	return nil
}

// CopyAndDecode decodes the data from the passed io.Reader while making a copy. Having a copy
// is especially useful if there is a chance the data will fail to decode.
// encodedAs specifies the expected encoding, r provides the io.Reader to the data, and v
// is the decoding destination.
func CopyAndDecode(encodedAs EncodedAs, r io.Reader, v interface{}) (bytes.Buffer, error) {
	b := bytes.Buffer{}
	return b, NewDecoder(encodedAs, io.TeeReader(r, &b)).Decode(v)
}

// TeeReadCloser returns a ReadCloser that writes to w what it reads from rc.
// It utilizes io.TeeReader to copy the data read and has the same behavior when reading.
// Further, when it is closed, it ensures that rc is closed as well.
func TeeReadCloser(rc io.ReadCloser, w io.Writer) io.ReadCloser {
	return &teeReadCloser{rc, io.TeeReader(rc, w)}
}

type teeReadCloser struct {
	rc io.ReadCloser
	r  io.Reader
}

func (t *teeReadCloser) Read(p []byte) (int, error) {
	return t.r.Read(p)
}

func (t *teeReadCloser) Close() error {
	return t.rc.Close()
}

func containsInt(ints []int, n int) bool {
	for _, i := range ints {
		if i == n {
			return true
		}
	}
	return false
}

func escapeValueStrings(m map[string]string) map[string]string {
	for key, value := range m {
		m[key] = url.QueryEscape(value)
	}
	return m
}

func ensureValueStrings(mapOfInterface map[string]interface{}) map[string]string {
	mapOfStrings := make(map[string]string)
	for key, value := range mapOfInterface {
		mapOfStrings[key] = ensureValueString(value)
	}
	return mapOfStrings
}

func ensureValueString(value interface{}) string {
	if value == nil {
		return ""
	}
	switch v := value.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		return fmt.Sprintf("%v", v)
	}
}

// MapToValues method converts map[string]interface{} to url.Values.
func MapToValues(m map[string]interface{}) url.Values {
	v := url.Values{}
	for key, value := range m {
		x := reflect.ValueOf(value)
		if x.Kind() == reflect.Array || x.Kind() == reflect.Slice {
			for i := 0; i < x.Len(); i++ {
				v.Add(key, ensureValueString(x.Index(i)))
			}
		} else {
			v.Add(key, ensureValueString(value))
		}
	}
	return v
}

// String method converts interface v to string. If interface is a list, it
// joins list elements using separator.
func String(v interface{}, sep ...string) string {
	if len(sep) > 0 {
		return ensureValueString(strings.Join(v.([]string), sep[0]))
	}
	return ensureValueString(v)
}

// Encode method encodes url path and query parameters.
func Encode(location string, v interface{}, sep ...string) string {
	s := String(v, sep...)
	switch strings.ToLower(location) {
	case "path":
		return pathEscape(s)
	case "query":
		return queryEscape(s)
	default:
		return s
	}
}

func pathEscape(s string) string {
	return strings.Replace(url.QueryEscape(s), "+", "%20", -1)
}

func queryEscape(s string) string {
	return url.QueryEscape(s)
}

// This method is same as Encode() method of "net/url" go package,
// except it does not encode the query parameters because they
// already come encoded. It formats values map in query format (bar=foo&a=b).
func createQuery(v url.Values) string {
	var buf bytes.Buffer
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		vs := v[k]
		prefix := url.QueryEscape(k) + "="
		for _, v := range vs {
			if buf.Len() > 0 {
				buf.WriteByte('&')
			}
			buf.WriteString(prefix)
			buf.WriteString(v)
		}
	}
	return buf.String()
}
