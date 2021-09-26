package handler

import (
	"io/ioutil"
	"math"
	"net/http"
	"net/http/httptest"
	"reflect"
	"sort"
	"testing"

	"github.com/davecgh/go-spew/spew"
	"github.com/go-openapi/spec"
	json "github.com/json-iterator/go"
	yaml "gopkg.in/yaml.v2"
)

var returnedSwagger = []byte(`{
  "swagger": "2.0",
  "info": {
   "title": "Kubernetes",
   "version": "v1.11.0"
  }}`)

func TestRegisterOpenAPIVersionedService(t *testing.T) {
	var s spec.Swagger
	err := s.UnmarshalJSON(returnedSwagger)
	if err != nil {
		t.Errorf("Unexpected error in unmarshalling SwaggerJSON: %v", err)
	}

	returnedJSON, err := json.Marshal(s)
	if err != nil {
		t.Errorf("Unexpected error in preparing returnedJSON: %v", err)
	}
	var decodedJSON map[string]interface{}
	if err := json.Unmarshal(returnedJSON, &decodedJSON); err != nil {
		t.Fatal(err)
	}
	returnedPb, err := ToProtoBinary(decodedJSON)
	if err != nil {
		t.Errorf("Unexpected error in preparing returnedPb: %v", err)
	}

	mux := http.NewServeMux()
	o, err := NewOpenAPIService(&s)
	if err != nil {
		t.Fatal(err)
	}
	if err = o.RegisterOpenAPIVersionedService("/openapi/v2", mux); err != nil {
		t.Errorf("Unexpected error in register OpenAPI versioned service: %v", err)
	}
	server := httptest.NewServer(mux)
	defer server.Close()
	client := server.Client()

	tcs := []struct {
		acceptHeader string
		respStatus   int
		respBody     []byte
	}{
		{"", 200, returnedJSON},
		{"*/*", 200, returnedJSON},
		{"application/*", 200, returnedJSON},
		{"application/json", 200, returnedJSON},
		{"test/test", 406, []byte{}},
		{"application/test", 406, []byte{}},
		{"application/test, */*", 200, returnedJSON},
		{"application/test, application/json", 200, returnedJSON},
		{"application/com.github.proto-openapi.spec.v2@v1.0+protobuf", 200, returnedPb},
		{"application/json, application/com.github.proto-openapi.spec.v2@v1.0+protobuf", 200, returnedJSON},
		{"application/com.github.proto-openapi.spec.v2@v1.0+protobuf, application/json", 200, returnedPb},
		{"application/com.github.proto-openapi.spec.v2@v1.0+protobuf; q=0.5, application/json", 200, returnedJSON},
	}

	for _, tc := range tcs {
		req, err := http.NewRequest("GET", server.URL+"/openapi/v2", nil)
		if err != nil {
			t.Errorf("Accept: %v: Unexpected error in creating new request: %v", tc.acceptHeader, err)
		}

		req.Header.Add("Accept", tc.acceptHeader)
		resp, err := client.Do(req)
		if err != nil {
			t.Errorf("Accept: %v: Unexpected error in serving HTTP request: %v", tc.acceptHeader, err)
		}

		if resp.StatusCode != tc.respStatus {
			t.Errorf("Accept: %v: Unexpected response status code, want: %v, got: %v", tc.acceptHeader, tc.respStatus, resp.StatusCode)
		}
		defer resp.Body.Close()
		body, err := ioutil.ReadAll(resp.Body)
		if err != nil {
			t.Errorf("Accept: %v: Unexpected error in reading response body: %v", tc.acceptHeader, err)
		}
		if !reflect.DeepEqual(body, tc.respBody) {
			t.Errorf("Accept: %v: Response body mismatches, \nwant: %s, \ngot:  %s", tc.acceptHeader, string(tc.respBody), string(body))
		}
	}
}

func TestJsonToYAML(t *testing.T) {
	intOrInt64 := func(i64 int64) interface{} {
		if i := int(i64); i64 == int64(i) {
			return i
		}
		return i64
	}

	tests := []struct {
		name     string
		input    map[string]interface{}
		expected yaml.MapSlice
	}{
		{"nil", nil, nil},
		{"empty", map[string]interface{}{}, yaml.MapSlice{}},
		{
			"values",
			map[string]interface{}{
				"bool":         true,
				"float64":      float64(42.1),
				"fractionless": float64(42),
				"int":          int(42),
				"int64":        int64(42),
				"int64 big":    float64(math.Pow(2, 62)),
				"map":          map[string]interface{}{"foo": "bar"},
				"slice":        []interface{}{"foo", "bar"},
				"string":       string("foo"),
				"uint64 big":   float64(math.Pow(2, 63)),
			},
			yaml.MapSlice{
				{"bool", true},
				{"float64", float64(42.1)},
				{"fractionless", int(42)},
				{"int", int(42)},
				{"int64", int(42)},
				{"int64 big", intOrInt64(int64(1) << 62)},
				{"map", yaml.MapSlice{{"foo", "bar"}}},
				{"slice", []interface{}{"foo", "bar"}},
				{"string", string("foo")},
				{"uint64 big", uint64(1) << 63},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := jsonToYAML(tt.input)
			sortMapSlicesInPlace(tt.expected)
			sortMapSlicesInPlace(got)
			if !reflect.DeepEqual(got, tt.expected) {
				t.Errorf("jsonToYAML() = %v, want %v", spew.Sdump(got), spew.Sdump(tt.expected))
			}
		})
	}
}

func sortMapSlicesInPlace(x interface{}) {
	switch x := x.(type) {
	case []interface{}:
		for i := range x {
			sortMapSlicesInPlace(x[i])
		}
	case yaml.MapSlice:
		sort.Slice(x, func(a, b int) bool {
			return x[a].Key.(string) < x[b].Key.(string)
		})
	}
}

func TestToProtoBinary(t *testing.T) {
	bs, err := ioutil.ReadFile("../../test/integration/testdata/aggregator/openapi.json")
	if err != nil {
		t.Fatal(err)
	}
	var j map[string]interface{}
	if err := json.Unmarshal(bs, &j); err != nil {
		t.Fatal(err)
	}
	if _, err := ToProtoBinary(j); err != nil {
		t.Fatal()
	}
	// TODO: add some kind of roundtrip test here
}
