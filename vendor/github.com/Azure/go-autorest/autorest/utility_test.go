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
	"net/http"
	"net/url"
	"reflect"
	"sort"
	"strings"
	"testing"

	"github.com/Azure/go-autorest/autorest/mocks"
)

const (
	jsonT = `
    {
      "name":"Rob Pike",
      "age":42
    }`
	xmlT = `<?xml version="1.0" encoding="UTF-8"?>
	<Person>
		<Name>Rob Pike</Name>
		<Age>42</Age>
	</Person>`
)

func TestNewDecoderCreatesJSONDecoder(t *testing.T) {
	d := NewDecoder(EncodedAsJSON, strings.NewReader(jsonT))
	_, ok := d.(*json.Decoder)
	if d == nil || !ok {
		t.Fatal("autorest: NewDecoder failed to create a JSON decoder when requested")
	}
}

func TestNewDecoderCreatesXMLDecoder(t *testing.T) {
	d := NewDecoder(EncodedAsXML, strings.NewReader(xmlT))
	_, ok := d.(*xml.Decoder)
	if d == nil || !ok {
		t.Fatal("autorest: NewDecoder failed to create an XML decoder when requested")
	}
}

func TestNewDecoderReturnsNilForUnknownEncoding(t *testing.T) {
	d := NewDecoder("unknown", strings.NewReader(xmlT))
	if d != nil {
		t.Fatal("autorest: NewDecoder created a decoder for an unknown encoding")
	}
}

func TestCopyAndDecodeDecodesJSON(t *testing.T) {
	_, err := CopyAndDecode(EncodedAsJSON, strings.NewReader(jsonT), &mocks.T{})
	if err != nil {
		t.Fatalf("autorest: CopyAndDecode returned an error with valid JSON - %v", err)
	}
}

func TestCopyAndDecodeDecodesXML(t *testing.T) {
	_, err := CopyAndDecode(EncodedAsXML, strings.NewReader(xmlT), &mocks.T{})
	if err != nil {
		t.Fatalf("autorest: CopyAndDecode returned an error with valid XML - %v", err)
	}
}

func TestCopyAndDecodeReturnsJSONDecodingErrors(t *testing.T) {
	_, err := CopyAndDecode(EncodedAsJSON, strings.NewReader(jsonT[0:len(jsonT)-2]), &mocks.T{})
	if err == nil {
		t.Fatalf("autorest: CopyAndDecode failed to return an error with invalid JSON")
	}
}

func TestCopyAndDecodeReturnsXMLDecodingErrors(t *testing.T) {
	_, err := CopyAndDecode(EncodedAsXML, strings.NewReader(xmlT[0:len(xmlT)-2]), &mocks.T{})
	if err == nil {
		t.Fatalf("autorest: CopyAndDecode failed to return an error with invalid XML")
	}
}

func TestCopyAndDecodeAlwaysReturnsACopy(t *testing.T) {
	b, _ := CopyAndDecode(EncodedAsJSON, strings.NewReader(jsonT), &mocks.T{})
	if b.String() != jsonT {
		t.Fatalf("autorest: CopyAndDecode failed to return a valid copy of the data - %v", b.String())
	}
}

func TestTeeReadCloser_Copies(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)
	b := &bytes.Buffer{}

	r.Body = TeeReadCloser(r.Body, b)

	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: TeeReadCloser returned an unexpected error -- %v", err)
	}
	if b.String() != jsonT {
		t.Fatalf("autorest: TeeReadCloser failed to copy the bytes read")
	}
}

func TestTeeReadCloser_PassesReadErrors(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)

	r.Body.(*mocks.Body).Close()
	r.Body = TeeReadCloser(r.Body, &bytes.Buffer{})

	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err == nil {
		t.Fatalf("autorest: TeeReadCloser failed to return the expected error")
	}
}

func TestTeeReadCloser_ClosesWrappedReader(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)

	b := r.Body.(*mocks.Body)
	r.Body = TeeReadCloser(r.Body, &bytes.Buffer{})
	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: TeeReadCloser returned an unexpected error -- %v", err)
	}
	if b.IsOpen() {
		t.Fatalf("autorest: TeeReadCloser failed to close the nested io.ReadCloser")
	}
}

func TestContainsIntFindsValue(t *testing.T) {
	ints := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	v := 5
	if !containsInt(ints, v) {
		t.Fatalf("autorest: containsInt failed to find %v in %v", v, ints)
	}
}

func TestContainsIntDoesNotFindValue(t *testing.T) {
	ints := []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	v := 42
	if containsInt(ints, v) {
		t.Fatalf("autorest: containsInt unexpectedly found %v in %v", v, ints)
	}
}

func TestContainsIntAcceptsEmptyList(t *testing.T) {
	ints := make([]int, 10)
	if containsInt(ints, 42) {
		t.Fatalf("autorest: containsInt failed to handle an empty list")
	}
}

func TestContainsIntAcceptsNilList(t *testing.T) {
	var ints []int
	if containsInt(ints, 42) {
		t.Fatalf("autorest: containsInt failed to handle an nil list")
	}
}

func TestEscapeStrings(t *testing.T) {
	m := map[string]string{
		"string": "a long string with = odd characters",
		"int":    "42",
		"nil":    "",
	}
	r := map[string]string{
		"string": "a+long+string+with+%3D+odd+characters",
		"int":    "42",
		"nil":    "",
	}
	v := escapeValueStrings(m)
	if !reflect.DeepEqual(v, r) {
		t.Fatalf("autorest: ensureValueStrings returned %v\n", v)
	}
}

func TestEnsureStrings(t *testing.T) {
	m := map[string]interface{}{
		"string": "string",
		"int":    42,
		"nil":    nil,
		"bytes":  []byte{255, 254, 253},
	}
	r := map[string]string{
		"string": "string",
		"int":    "42",
		"nil":    "",
		"bytes":  string([]byte{255, 254, 253}),
	}
	v := ensureValueStrings(m)
	if !reflect.DeepEqual(v, r) {
		t.Fatalf("autorest: ensureValueStrings returned %v\n", v)
	}
}

func ExampleString() {
	m := []string{
		"string1",
		"string2",
		"string3",
	}

	fmt.Println(String(m, ","))
	// Output: string1,string2,string3
}

func TestStringWithValidString(t *testing.T) {
	i := 123
	if String(i) != "123" {
		t.Fatal("autorest: String method failed to convert integer 123 to string")
	}
}

func TestEncodeWithValidPath(t *testing.T) {
	s := Encode("Path", "Hello Gopher")
	if s != "Hello%20Gopher" {
		t.Fatalf("autorest: Encode method failed for valid path encoding. Got: %v; Want: %v", s, "Hello%20Gopher")
	}
}

func TestEncodeWithValidQuery(t *testing.T) {
	s := Encode("Query", "Hello Gopher")
	if s != "Hello+Gopher" {
		t.Fatalf("autorest: Encode method failed for valid query encoding. Got: '%v'; Want: 'Hello+Gopher'", s)
	}
}

func TestEncodeWithValidNotPathQuery(t *testing.T) {
	s := Encode("Host", "Hello Gopher")
	if s != "Hello Gopher" {
		t.Fatalf("autorest: Encode method failed for parameter not query or path. Got: '%v'; Want: 'Hello Gopher'", s)
	}
}

func TestMapToValues(t *testing.T) {
	m := map[string]interface{}{
		"a": "a",
		"b": 2,
	}
	v := url.Values{}
	v.Add("a", "a")
	v.Add("b", "2")
	if !isEqual(v, MapToValues(m)) {
		t.Fatalf("autorest: MapToValues method failed to return correct values - expected(%v) got(%v)", v, MapToValues(m))
	}
}

func TestMapToValuesWithArrayValues(t *testing.T) {
	m := map[string]interface{}{
		"a": []string{"a", "b"},
		"b": 2,
		"c": []int{3, 4},
	}
	v := url.Values{}
	v.Add("a", "a")
	v.Add("a", "b")
	v.Add("b", "2")
	v.Add("c", "3")
	v.Add("c", "4")

	if !isEqual(v, MapToValues(m)) {
		t.Fatalf("autorest: MapToValues method failed to return correct values - expected(%v) got(%v)", v, MapToValues(m))
	}
}

func isEqual(v, u url.Values) bool {
	for key, value := range v {
		if len(u[key]) == 0 {
			return false
		}
		sort.Strings(value)
		sort.Strings(u[key])
		for i := range value {
			if value[i] != u[key][i] {
				return false
			}
		}
		u.Del(key)
	}
	if len(u) > 0 {
		return false
	}
	return true
}

func doEnsureBodyClosed(t *testing.T) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if resp != nil && resp.Body != nil && resp.Body.(*mocks.Body).IsOpen() {
				t.Fatal("autorest: Expected Body to be closed -- it was left open")
			}
			return resp, err
		})
	}
}

type mockAuthorizer struct{}

func (ma mockAuthorizer) WithAuthorization() PrepareDecorator {
	return WithHeader(headerAuthorization, mocks.TestAuthorizationHeader)
}

type mockFailingAuthorizer struct{}

func (mfa mockFailingAuthorizer) WithAuthorization() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			return r, fmt.Errorf("ERROR: mockFailingAuthorizer returned expected error")
		})
	}
}

type mockInspector struct {
	wasInvoked bool
}

func (mi *mockInspector) WithInspection() PrepareDecorator {
	return func(p Preparer) Preparer {
		return PreparerFunc(func(r *http.Request) (*http.Request, error) {
			mi.wasInvoked = true
			return p.Prepare(r)
		})
	}
}

func (mi *mockInspector) ByInspecting() RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			mi.wasInvoked = true
			return r.Respond(resp)
		})
	}
}

func withMessage(output *string, msg string) SendDecorator {
	return func(s Sender) Sender {
		return SenderFunc(func(r *http.Request) (*http.Response, error) {
			resp, err := s.Do(r)
			if err == nil {
				*output += msg
			}
			return resp, err
		})
	}
}

func withErrorRespondDecorator(e *error) RespondDecorator {
	return func(r Responder) Responder {
		return ResponderFunc(func(resp *http.Response) error {
			err := r.Respond(resp)
			if err != nil {
				return err
			}
			*e = fmt.Errorf("autorest: Faux Respond Error")
			return *e
		})
	}
}
