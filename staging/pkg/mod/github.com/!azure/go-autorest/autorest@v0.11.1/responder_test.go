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
	"fmt"
	"io/ioutil"
	"net/http"
	"reflect"
	"strings"
	"testing"

	"github.com/Azure/go-autorest/autorest/mocks"
)

func ExampleWithErrorUnlessOK() {
	r := mocks.NewResponse()
	r.Request = mocks.NewRequest()

	// Respond and leave the response body open (for a subsequent responder to close)
	err := Respond(r,
		WithErrorUnlessOK(),
		ByDiscardingBody(),
		ByClosingIfError())

	if err == nil {
		fmt.Printf("%s of %s returned HTTP 200", r.Request.Method, r.Request.URL)

		// Complete handling the response and close the body
		Respond(r,
			ByDiscardingBody(),
			ByClosing())
	}
	// Output: GET of https://microsoft.com/a/b/c/ returned HTTP 200
}

func TestByUnmarshallingBytes(t *testing.T) {
	expected := []byte("Lorem Ipsum Dolor")

	// we'll create a fixed-sized array here, since that's the expectation
	bytes := make([]byte, len(expected))

	Respond(mocks.NewResponseWithBytes(expected),
		ByUnmarshallingBytes(&bytes),
		ByClosing())

	if len(bytes) != len(expected) {
		t.Fatalf("Expected Response to be %d bytes but got %d bytes", len(expected), len(bytes))
	}

	if !reflect.DeepEqual(expected, bytes) {
		t.Fatalf("Expected Response to be %s but got %s", expected, bytes)
	}
}

func ExampleByUnmarshallingJSON() {
	c := `
	{
		"name" : "Rob Pike",
		"age"  : 42
	}
	`

	type V struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}

	v := &V{}

	Respond(mocks.NewResponseWithContent(c),
		ByUnmarshallingJSON(v),
		ByClosing())

	fmt.Printf("%s is %d years old\n", v.Name, v.Age)
	// Output: Rob Pike is 42 years old
}

func ExampleByUnmarshallingXML() {
	c := `<?xml version="1.0" encoding="UTF-8"?>
	<Person>
	  <Name>Rob Pike</Name>
	  <Age>42</Age>
	</Person>`

	type V struct {
		Name string `xml:"Name"`
		Age  int    `xml:"Age"`
	}

	v := &V{}

	Respond(mocks.NewResponseWithContent(c),
		ByUnmarshallingXML(v),
		ByClosing())

	fmt.Printf("%s is %d years old\n", v.Name, v.Age)
	// Output: Rob Pike is 42 years old
}

func TestCreateResponderDoesNotModify(t *testing.T) {
	r1 := mocks.NewResponse()
	r2 := mocks.NewResponse()
	p := CreateResponder()
	err := p.Respond(r1)
	if err != nil {
		t.Fatalf("autorest: CreateResponder failed (%v)", err)
	}
	if !reflect.DeepEqual(r1, r2) {
		t.Fatalf("autorest: CreateResponder without decorators modified the response")
	}
}

func TestCreateResponderRunsDecoratorsInOrder(t *testing.T) {
	s := ""

	d := func(n int) RespondDecorator {
		return func(r Responder) Responder {
			return ResponderFunc(func(resp *http.Response) error {
				err := r.Respond(resp)
				if err == nil {
					s += fmt.Sprintf("%d", n)
				}
				return err
			})
		}
	}

	p := CreateResponder(d(1), d(2), d(3))
	err := p.Respond(&http.Response{})
	if err != nil {
		t.Fatalf("autorest: Respond failed (%v)", err)
	}

	if s != "123" {
		t.Fatalf("autorest: CreateResponder invoked decorators in an incorrect order; expected '123', received '%s'", s)
	}
}

func TestByIgnoring(t *testing.T) {
	r := mocks.NewResponse()

	Respond(r,
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(r2 *http.Response) error {
					r1 := mocks.NewResponse()
					if !reflect.DeepEqual(r1, r2) {
						t.Fatalf("autorest: ByIgnoring modified the HTTP Response -- received %v, expected %v", r2, r1)
					}
					return nil
				})
			}
		})(),
		ByIgnoring(),
		ByClosing())
}

func TestByCopying_Copies(t *testing.T) {
	r := mocks.NewResponseWithContent(jsonT)
	b := &bytes.Buffer{}

	err := Respond(r,
		ByCopying(b),
		ByUnmarshallingJSON(&mocks.T{}),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: ByCopying returned an unexpected error -- %v", err)
	}
	if b.String() != jsonT {
		t.Fatalf("autorest: ByCopying failed to copy the bytes read")
	}
}

func TestByCopying_ReturnsNestedErrors(t *testing.T) {
	r := mocks.NewResponseWithContent(jsonT)

	r.Body.Close()
	err := Respond(r,
		ByCopying(&bytes.Buffer{}),
		ByUnmarshallingJSON(&mocks.T{}),
		ByClosing())
	if err == nil {
		t.Fatalf("autorest: ByCopying failed to return the expected error")
	}
}

func TestByCopying_AcceptsNilReponse(t *testing.T) {
	r := mocks.NewResponse()

	Respond(r,
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					r.Respond(nil)
					return nil
				})
			}
		})(),
		ByCopying(&bytes.Buffer{}))
}

func TestByCopying_AcceptsNilBody(t *testing.T) {
	r := mocks.NewResponse()

	Respond(r,
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					resp.Body = nil
					r.Respond(resp)
					return nil
				})
			}
		})(),
		ByCopying(&bytes.Buffer{}))
}

func TestByClosing(t *testing.T) {
	r := mocks.NewResponse()
	err := Respond(r, ByClosing())
	if err != nil {
		t.Fatalf("autorest: ByClosing failed (%v)", err)
	}
	if r.Body.(*mocks.Body).IsOpen() {
		t.Fatalf("autorest: ByClosing did not close the response body")
	}
}

func TestByClosingAcceptsNilResponse(t *testing.T) {
	r := mocks.NewResponse()

	Respond(r,
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					r.Respond(nil)
					return nil
				})
			}
		})(),
		ByClosing())
}

func TestByClosingAcceptsNilBody(t *testing.T) {
	r := mocks.NewResponse()

	Respond(r,
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					resp.Body = nil
					r.Respond(resp)
					return nil
				})
			}
		})(),
		ByClosing())
}

func TestByClosingClosesEvenAfterErrors(t *testing.T) {
	var e error

	r := mocks.NewResponse()
	Respond(r,
		withErrorRespondDecorator(&e),
		ByClosing())

	if r.Body.(*mocks.Body).IsOpen() {
		t.Fatalf("autorest: ByClosing did not close the response body after an error occurred")
	}
}

func TestByClosingClosesReturnsNestedErrors(t *testing.T) {
	var e error

	r := mocks.NewResponse()
	err := Respond(r,
		withErrorRespondDecorator(&e),
		ByClosing())

	if err == nil || !reflect.DeepEqual(e, err) {
		t.Fatalf("autorest: ByClosing failed to return a nested error")
	}
}

func TestByClosingIfErrorAcceptsNilResponse(t *testing.T) {
	var e error

	r := mocks.NewResponse()

	Respond(r,
		withErrorRespondDecorator(&e),
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					r.Respond(nil)
					return nil
				})
			}
		})(),
		ByClosingIfError())
}

func TestByClosingIfErrorAcceptsNilBody(t *testing.T) {
	var e error

	r := mocks.NewResponse()

	Respond(r,
		withErrorRespondDecorator(&e),
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					resp.Body = nil
					r.Respond(resp)
					return nil
				})
			}
		})(),
		ByClosingIfError())
}

func TestByClosingIfErrorClosesIfAnErrorOccurs(t *testing.T) {
	var e error

	r := mocks.NewResponse()
	Respond(r,
		withErrorRespondDecorator(&e),
		ByClosingIfError())

	if r.Body.(*mocks.Body).IsOpen() {
		t.Fatalf("autorest: ByClosingIfError did not close the response body after an error occurred")
	}
}

func TestByClosingIfErrorDoesNotClosesIfNoErrorOccurs(t *testing.T) {
	r := mocks.NewResponse()
	Respond(r,
		ByClosingIfError())

	if !r.Body.(*mocks.Body).IsOpen() {
		t.Fatalf("autorest: ByClosingIfError closed the response body even though no error occurred")
	}
}

func TestByDiscardingBody(t *testing.T) {
	r := mocks.NewResponse()
	err := Respond(r,
		ByDiscardingBody())
	if err != nil {
		t.Fatalf("autorest: ByDiscardingBody failed (%v)", err)
	}
	buf, err := ioutil.ReadAll(r.Body)
	if err != nil {
		t.Fatalf("autorest: Reading result of ByDiscardingBody failed (%v)", err)
	}

	if len(buf) != 0 {
		t.Logf("autorest: Body was not empty after calling ByDiscardingBody.")
		t.Fail()
	}
}

func TestByDiscardingBodyAcceptsNilResponse(t *testing.T) {
	var e error

	r := mocks.NewResponse()

	Respond(r,
		withErrorRespondDecorator(&e),
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					r.Respond(nil)
					return nil
				})
			}
		})(),
		ByDiscardingBody())
}

func TestByDiscardingBodyAcceptsNilBody(t *testing.T) {
	var e error

	r := mocks.NewResponse()

	Respond(r,
		withErrorRespondDecorator(&e),
		(func() RespondDecorator {
			return func(r Responder) Responder {
				return ResponderFunc(func(resp *http.Response) error {
					resp.Body.Close()
					resp.Body = nil
					r.Respond(resp)
					return nil
				})
			}
		})(),
		ByDiscardingBody())
}

func TestByUnmarshallingJSON(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)
	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: ByUnmarshallingJSON failed (%v)", err)
	}
	if v.Name != "Rob Pike" || v.Age != 42 {
		t.Fatalf("autorest: ByUnmarshallingJSON failed to properly unmarshal")
	}
}

func TestByUnmarshallingJSON_HandlesReadErrors(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)
	r.Body.(*mocks.Body).Close()

	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err == nil {
		t.Fatalf("autorest: ByUnmarshallingJSON failed to receive / respond to read error")
	}
}

func TestByUnmarshallingJSONIncludesJSONInErrors(t *testing.T) {
	v := &mocks.T{}
	j := jsonT[0 : len(jsonT)-2]
	r := mocks.NewResponseWithContent(j)
	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err == nil || !strings.Contains(err.Error(), j) {
		t.Fatalf("autorest: ByUnmarshallingJSON failed to return JSON in error (%v)", err)
	}
}

func TestByUnmarshallingJSONEmptyInput(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(``)
	err := Respond(r,
		ByUnmarshallingJSON(v),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: ByUnmarshallingJSON failed to return nil in case of empty JSON (%v)", err)
	}
}

func TestByUnmarshallingXML(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(xmlT)
	err := Respond(r,
		ByUnmarshallingXML(v),
		ByClosing())
	if err != nil {
		t.Fatalf("autorest: ByUnmarshallingXML failed (%v)", err)
	}
	if v.Name != "Rob Pike" || v.Age != 42 {
		t.Fatalf("autorest: ByUnmarshallingXML failed to properly unmarshal")
	}
}

func TestByUnmarshallingXML_HandlesReadErrors(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(xmlT)
	r.Body.(*mocks.Body).Close()

	err := Respond(r,
		ByUnmarshallingXML(v),
		ByClosing())
	if err == nil {
		t.Fatalf("autorest: ByUnmarshallingXML failed to receive / respond to read error")
	}
}

func TestByUnmarshallingXMLIncludesXMLInErrors(t *testing.T) {
	v := &mocks.T{}
	x := xmlT[0 : len(xmlT)-2]
	r := mocks.NewResponseWithContent(x)
	err := Respond(r,
		ByUnmarshallingXML(v),
		ByClosing())
	if err == nil || !strings.Contains(err.Error(), x) {
		t.Fatalf("autorest: ByUnmarshallingXML failed to return XML in error (%v)", err)
	}
}

func TestRespondAcceptsNullResponse(t *testing.T) {
	err := Respond(nil)
	if err != nil {
		t.Fatalf("autorest: Respond returned an unexpected error when given a null Response (%v)", err)
	}
}

func TestWithErrorUnlessStatusCodeOKResponse(t *testing.T) {
	v := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)
	err := Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		ByUnmarshallingJSON(v),
		ByClosing())

	if err != nil {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK) failed on okay response. (%v)", err)
	}

	if v.Name != "Rob Pike" || v.Age != 42 {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK) corrupted the response body of okay response.")
	}
}

func TesWithErrorUnlessStatusCodeErrorResponse(t *testing.T) {
	v := &mocks.T{}
	e := &mocks.T{}
	r := mocks.NewResponseWithContent(jsonT)
	r.Status = "400 BadRequest"
	r.StatusCode = http.StatusBadRequest

	err := Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK),
		ByUnmarshallingJSON(v),
		ByClosing())

	if err == nil {
		t.Fatal("autorest: WithErrorUnlessStatusCode(http.StatusOK) did not return error, on a response to a bad request.")
	}

	var errorRespBody []byte
	if derr, ok := err.(DetailedError); !ok {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK) got wrong error type : %T, expected: DetailedError, on a response to a bad request.", err)
	} else {
		errorRespBody = derr.ServiceError
	}

	if errorRespBody == nil {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK) ServiceError not returned in DetailedError on a response to a bad request.")
	}

	err = json.Unmarshal(errorRespBody, e)
	if err != nil {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK) cannot parse error returned in ServiceError into json. %v", err)
	}

	expected := &mocks.T{Name: "Rob Pike", Age: 42}
	if e != expected {
		t.Fatalf("autorest: WithErrorUnlessStatusCode(http.StatusOK wrong value from parsed ServiceError: got=%#v expected=%#v", e, expected)
	}
}

func TestWithErrorUnlessStatusCode(t *testing.T) {
	r := mocks.NewResponse()
	r.Request = mocks.NewRequest()
	r.Status = "400 BadRequest"
	r.StatusCode = http.StatusBadRequest

	err := Respond(r,
		WithErrorUnlessStatusCode(http.StatusBadRequest, http.StatusUnauthorized, http.StatusInternalServerError),
		ByClosingIfError())

	if err != nil {
		t.Fatalf("autorest: WithErrorUnlessStatusCode returned an error (%v) for an acceptable status code (%s)", err, r.Status)
	}
}

func TestWithErrorUnlessStatusCodeEmitsErrorForUnacceptableStatusCode(t *testing.T) {
	r := mocks.NewResponse()
	r.Request = mocks.NewRequest()
	r.Status = "400 BadRequest"
	r.StatusCode = http.StatusBadRequest

	err := Respond(r,
		WithErrorUnlessStatusCode(http.StatusOK, http.StatusUnauthorized, http.StatusInternalServerError),
		ByClosingIfError())

	if err == nil {
		t.Fatalf("autorest: WithErrorUnlessStatusCode failed to return an error for an unacceptable status code (%s)", r.Status)
	}
}

func TestWithErrorUnlessOK(t *testing.T) {
	r := mocks.NewResponse()
	r.Request = mocks.NewRequest()

	err := Respond(r,
		WithErrorUnlessOK(),
		ByClosingIfError())

	if err != nil {
		t.Fatalf("autorest: WithErrorUnlessOK returned an error for OK status code (%v)", err)
	}
}

func TestWithErrorUnlessOKEmitsErrorIfNotOK(t *testing.T) {
	r := mocks.NewResponse()
	r.Request = mocks.NewRequest()
	r.Status = "400 BadRequest"
	r.StatusCode = http.StatusBadRequest

	err := Respond(r,
		WithErrorUnlessOK(),
		ByClosingIfError())

	if err == nil {
		t.Fatalf("autorest: WithErrorUnlessOK failed to return an error for a non-OK status code (%v)", err)
	}
}

func TestExtractHeader(t *testing.T) {
	r := mocks.NewResponse()
	v := []string{"v1", "v2", "v3"}
	mocks.SetResponseHeaderValues(r, mocks.TestHeader, v)

	if !reflect.DeepEqual(ExtractHeader(mocks.TestHeader, r), v) {
		t.Fatalf("autorest: ExtractHeader failed to retrieve the expected header -- expected [%s]%v, received [%s]%v",
			mocks.TestHeader, v, mocks.TestHeader, ExtractHeader(mocks.TestHeader, r))
	}
}

func TestExtractHeaderHandlesMissingHeader(t *testing.T) {
	var v []string
	r := mocks.NewResponse()

	if !reflect.DeepEqual(ExtractHeader(mocks.TestHeader, r), v) {
		t.Fatalf("autorest: ExtractHeader failed to handle a missing header -- expected %v, received %v",
			v, ExtractHeader(mocks.TestHeader, r))
	}
}

func TestExtractHeaderValue(t *testing.T) {
	r := mocks.NewResponse()
	v := "v1"
	mocks.SetResponseHeader(r, mocks.TestHeader, v)

	if ExtractHeaderValue(mocks.TestHeader, r) != v {
		t.Fatalf("autorest: ExtractHeader failed to retrieve the expected header -- expected [%s]%v, received [%s]%v",
			mocks.TestHeader, v, mocks.TestHeader, ExtractHeaderValue(mocks.TestHeader, r))
	}
}

func TestExtractHeaderValueHandlesMissingHeader(t *testing.T) {
	r := mocks.NewResponse()
	v := ""

	if ExtractHeaderValue(mocks.TestHeader, r) != v {
		t.Fatalf("autorest: ExtractHeader failed to retrieve the expected header -- expected [%s]%v, received [%s]%v",
			mocks.TestHeader, v, mocks.TestHeader, ExtractHeaderValue(mocks.TestHeader, r))
	}
}

func TestExtractHeaderValueRetrievesFirstValue(t *testing.T) {
	r := mocks.NewResponse()
	v := []string{"v1", "v2", "v3"}
	mocks.SetResponseHeaderValues(r, mocks.TestHeader, v)

	if ExtractHeaderValue(mocks.TestHeader, r) != v[0] {
		t.Fatalf("autorest: ExtractHeader failed to retrieve the expected header -- expected [%s]%v, received [%s]%v",
			mocks.TestHeader, v[0], mocks.TestHeader, ExtractHeaderValue(mocks.TestHeader, r))
	}
}
