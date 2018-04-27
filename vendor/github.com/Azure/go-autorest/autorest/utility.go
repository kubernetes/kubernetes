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
	"net/http"
	"net/url"
	"reflect"
	"strings"

	"github.com/Azure/go-autorest/autorest/adal"
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

// AsStringSlice method converts interface{} to []string. This expects a
//that the parameter passed to be a slice or array of a type that has the underlying
//type a string.
func AsStringSlice(s interface{}) ([]string, error) {
	v := reflect.ValueOf(s)
	if v.Kind() != reflect.Slice && v.Kind() != reflect.Array {
		return nil, NewError("autorest", "AsStringSlice", "the value's type is not an array.")
	}
	stringSlice := make([]string, 0, v.Len())

	for i := 0; i < v.Len(); i++ {
		stringSlice = append(stringSlice, v.Index(i).String())
	}
	return stringSlice, nil
}

// String method converts interface v to string. If interface is a list, it
// joins list elements using the seperator. Note that only sep[0] will be used for
// joining if any separator is specified.
func String(v interface{}, sep ...string) string {
	if len(sep) == 0 {
		return ensureValueString(v)
	}
	stringSlice, ok := v.([]string)
	if ok == false {
		var err error
		stringSlice, err = AsStringSlice(v)
		if err != nil {
			panic(fmt.Sprintf("autorest: Couldn't convert value to a string %s.", err))
		}
	}
	return ensureValueString(strings.Join(stringSlice, sep[0]))
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

// ChangeToGet turns the specified http.Request into a GET (it assumes it wasn't).
// This is mainly useful for long-running operations that use the Azure-AsyncOperation
// header, so we change the initial PUT into a GET to retrieve the final result.
func ChangeToGet(req *http.Request) *http.Request {
	req.Method = "GET"
	req.Body = nil
	req.ContentLength = 0
	req.Header.Del("Content-Length")
	return req
}

// IsTokenRefreshError returns true if the specified error implements the TokenRefreshError
// interface.  If err is a DetailedError it will walk the chain of Original errors.
func IsTokenRefreshError(err error) bool {
	if _, ok := err.(adal.TokenRefreshError); ok {
		return true
	}
	if de, ok := err.(DetailedError); ok {
		return IsTokenRefreshError(de.Original)
	}
	return false
}
