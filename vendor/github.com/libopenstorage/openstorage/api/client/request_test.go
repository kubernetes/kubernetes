package client

import (
	"encoding/json"
	"net/url"
	"testing"
)

func TestQueryOptionLabel(t *testing.T) {
	expected := "http://localhost:8000/v1/resource/instance?"
	u, _ := url.Parse("http://localhost:8000")
	r := NewRequest(nil, u, "GET", "v1", "", "")
	l := map[string]string{"foo": "bar"}
	b, _ := json.Marshal(l)
	q := url.Values{}
	q.Set("key", string(b))
	expected += q.Encode()
	actual := r.Resource("resource").Instance("instance").QueryOptionLabel("key", l).URL().String()
	if actual != expected {
		t.Fatalf("\nExpected %#v\nbut got  %#v", expected, actual)
	}
}

func TestQueryOption(t *testing.T) {
	expected := "http://localhost:8000/v1/resource/instance?key=val"
	url, _ := url.Parse("http://localhost:8000")
	r := NewRequest(nil, url, "GET", "v1", "", "")
	actual := r.Resource("resource").Instance("instance").QueryOption("key", "val").URL().String()
	if actual != expected {
		t.Fatalf("\nExpected %#v\nbut got  %#v", expected, actual)
	}
}

func TestBasic(t *testing.T) {
	expected := "http://localhost:8000/v1/resource/instance"
	url, _ := url.Parse("http://localhost:8000")
	r := NewRequest(nil, url, "GET", "v1", "", "")
	actual := r.Resource("resource").Instance("instance").URL().String()
	if actual != expected {
		t.Fatalf("\nExpected %#v\nbut got  %#v", expected, actual)
	}
}

func init() {
}
