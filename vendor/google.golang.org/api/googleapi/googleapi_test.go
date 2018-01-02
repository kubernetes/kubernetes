// Copyright 2011 Google Inc. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package googleapi

import (
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/url"
	"reflect"
	"strings"
	"testing"
)

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
		"http://www.golang.org/%7Bbucket/get",
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
		got := u.EscapedPath()
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

func TestRoundChunkSize(t *testing.T) {
	type testCase struct {
		in   int
		want int
	}
	for _, tc := range []testCase{
		{0, 0},
		{256*1024 - 1, 256 * 1024},
		{256 * 1024, 256 * 1024},
		{256*1024 + 1, 2 * 256 * 1024},
	} {
		mo := &MediaOptions{}
		ChunkSize(tc.in).setOptions(mo)
		if got := mo.ChunkSize; got != tc.want {
			t.Errorf("rounding chunk size: got: %v; want %v", got, tc.want)
		}
	}
}
