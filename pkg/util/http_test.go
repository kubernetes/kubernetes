package util

import (
	"net/http"
	"net/http/httptest"
	"net/url"
	"reflect"
	"testing"
)

func TestRawURL(t *testing.T) {
	testCases := map[string]string{
		"":      "/",
		"/":     "/",
		"/bar":  "/bar",
		"/bar/": "/bar/",

		"/bar?test=1":          "/bar",
		"/bar%3Ftest=1":        "/bar%3Ftest=1",
		"/bar?test=1#fragment": "/bar",
		"/bar#fragment":        "/bar",

		"/%2F":      "/%2F",
		"/%2F/a%2F": "/%2F/a%2F",
	}
	ch := make(chan string, 1)
	server := httptest.NewServer(RawURL(http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		ch <- req.URL.Path
	})))
	for source, expected := range testCases {
		target, _ := url.Parse(server.URL)
		target.Opaque = source
		req := &http.Request{Method: "GET", URL: target}
		client := http.Client{}
		_, err := client.Do(req)
		if err != nil {
			t.Fatalf("unexpected error %v", err)
		}
		actual := <-ch
		if expected != actual {
			t.Errorf("expected %s to become %s, got %s", source, expected, actual)
		}
	}
	close(ch)
	server.Close()
}

func TestSplitRawPath(t *testing.T) {
	testCases := map[string][]string{
		"":           []string{},
		"/":          []string{},
		"//":         []string{},
		"/bar":       []string{"bar"},
		"/%2F":       []string{"/"},
		"/%2F/a%2F":  []string{"/", "a/"},
		"/%2F/a%2F/": []string{"/", "a/"},
	}
	for source, expected := range testCases {
		actual, err := SplitRawPath(source)
		if err != nil {
			t.Errorf("unexpected error %v", err)
			continue
		}
		if !reflect.DeepEqual(expected, actual) {
			t.Errorf("expected %s to become %v, got %v", source, expected, actual)
		}
	}

	failureCases := []string{
		"/%2", // single escape char
		"/%Z", // out of range
	}
	for _, source := range failureCases {
		_, err := SplitRawPath(source)
		if err == nil {
			t.Errorf("unexpected non-error for %s", source)
		}
	}
}
