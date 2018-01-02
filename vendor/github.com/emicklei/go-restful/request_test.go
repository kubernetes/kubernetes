package restful

import (
	"encoding/json"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"testing"
)

func TestQueryParameter(t *testing.T) {
	hreq := http.Request{Method: "GET"}
	hreq.URL, _ = url.Parse("http://www.google.com/search?q=foo&q=bar")
	rreq := Request{Request: &hreq}
	if rreq.QueryParameter("q") != "foo" {
		t.Errorf("q!=foo %#v", rreq)
	}
}

type Anything map[string]interface{}

type Number struct {
	ValueFloat float64
	ValueInt   int64
}

type Sample struct {
	Value string
}

func TestReadEntityJson(t *testing.T) {
	bodyReader := strings.NewReader(`{"Value" : "42"}`)
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/json")
	request := &Request{Request: httpRequest}
	sam := new(Sample)
	request.ReadEntity(sam)
	if sam.Value != "42" {
		t.Fatal("read failed")
	}
}

func TestReadEntityJsonCharset(t *testing.T) {
	bodyReader := strings.NewReader(`{"Value" : "42"}`)
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/json; charset=UTF-8")
	request := NewRequest(httpRequest)
	sam := new(Sample)
	request.ReadEntity(sam)
	if sam.Value != "42" {
		t.Fatal("read failed")
	}
}

func TestReadEntityJsonNumber(t *testing.T) {
	bodyReader := strings.NewReader(`{"Value" : 4899710515899924123}`)
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/json")
	request := &Request{Request: httpRequest}
	any := make(Anything)
	request.ReadEntity(&any)
	number, ok := any["Value"].(json.Number)
	if !ok {
		t.Fatal("read failed")
	}
	vint, err := number.Int64()
	if err != nil {
		t.Fatal("convert failed")
	}
	if vint != 4899710515899924123 {
		t.Fatal("read failed")
	}
	vfloat, err := number.Float64()
	if err != nil {
		t.Fatal("convert failed")
	}
	// match the default behaviour
	vstring := strconv.FormatFloat(vfloat, 'e', 15, 64)
	if vstring != "4.899710515899924e+18" {
		t.Fatal("convert float64 failed")
	}
}

func TestReadEntityJsonLong(t *testing.T) {
	bodyReader := strings.NewReader(`{"ValueFloat" : 4899710515899924123, "ValueInt": 4899710515899924123}`)
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/json")
	request := &Request{Request: httpRequest}
	number := new(Number)
	request.ReadEntity(&number)
	if number.ValueInt != 4899710515899924123 {
		t.Fatal("read failed")
	}
	// match the default behaviour
	vstring := strconv.FormatFloat(number.ValueFloat, 'e', 15, 64)
	if vstring != "4.899710515899924e+18" {
		t.Fatal("convert float64 failed")
	}
}

func TestBodyParameter(t *testing.T) {
	bodyReader := strings.NewReader(`value1=42&value2=43`)
	httpRequest, _ := http.NewRequest("POST", "/test?value1=44", bodyReader) // POST and PUT body parameters take precedence over URL query string
	httpRequest.Header.Set("Content-Type", "application/x-www-form-urlencoded; charset=UTF-8")
	request := NewRequest(httpRequest)
	v1, err := request.BodyParameter("value1")
	if err != nil {
		t.Error(err)
	}
	v2, err := request.BodyParameter("value2")
	if err != nil {
		t.Error(err)
	}
	if v1 != "42" || v2 != "43" {
		t.Fatal("read failed")
	}
}

func TestReadEntityUnkown(t *testing.T) {
	bodyReader := strings.NewReader("?")
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	httpRequest.Header.Set("Content-Type", "application/rubbish")
	request := NewRequest(httpRequest)
	sam := new(Sample)
	err := request.ReadEntity(sam)
	if err == nil {
		t.Fatal("read should be in error")
	}
}

func TestSetAttribute(t *testing.T) {
	bodyReader := strings.NewReader("?")
	httpRequest, _ := http.NewRequest("GET", "/test", bodyReader)
	request := NewRequest(httpRequest)
	request.SetAttribute("go", "there")
	there := request.Attribute("go")
	if there != "there" {
		t.Fatalf("missing request attribute:%v", there)
	}
}
