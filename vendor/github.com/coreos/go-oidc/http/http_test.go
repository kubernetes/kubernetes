package http

import (
	"net/http"
	"net/url"
	"reflect"
	"testing"
	"time"
)

func TestCacheControlMaxAgeSuccess(t *testing.T) {
	tests := []struct {
		hdr     string
		wantAge time.Duration
		wantOK  bool
	}{
		{"max-age=12", 12 * time.Second, true},
		{"max-age=-12", 0, false},
		{"max-age=0", 0, false},
		{"public, max-age=12", 12 * time.Second, true},
		{"public, max-age=40192, must-revalidate", 40192 * time.Second, true},
		{"public, not-max-age=12, must-revalidate", time.Duration(0), false},
	}

	for i, tt := range tests {
		maxAge, ok, err := cacheControlMaxAge(tt.hdr)
		if err != nil {
			t.Errorf("case %d: err=%v", i, err)
		}
		if tt.wantAge != maxAge {
			t.Errorf("case %d: want=%d got=%d", i, tt.wantAge, maxAge)
		}
		if tt.wantOK != ok {
			t.Errorf("case %d: incorrect ok value: want=%t got=%t", i, tt.wantOK, ok)
		}
	}
}

func TestCacheControlMaxAgeFail(t *testing.T) {
	tests := []string{
		"max-age=aasdf",
		"max-age=",
		"max-age",
	}

	for i, tt := range tests {
		_, ok, err := cacheControlMaxAge(tt)
		if ok {
			t.Errorf("case %d: want ok=false, got true", i)
		}
		if err == nil {
			t.Errorf("case %d: want non-nil err", i)
		}
	}
}

func TestMergeQuery(t *testing.T) {
	tests := []struct {
		u string
		q url.Values
		w string
	}{
		// No values
		{
			u: "http://example.com",
			q: nil,
			w: "http://example.com",
		},
		// No additional values
		{
			u: "http://example.com?foo=bar",
			q: nil,
			w: "http://example.com?foo=bar",
		},
		// Simple addition
		{
			u: "http://example.com",
			q: url.Values{
				"foo": []string{"bar"},
			},
			w: "http://example.com?foo=bar",
		},
		// Addition with existing values
		{
			u: "http://example.com?dog=boo",
			q: url.Values{
				"foo": []string{"bar"},
			},
			w: "http://example.com?dog=boo&foo=bar",
		},
		// Merge
		{
			u: "http://example.com?dog=boo",
			q: url.Values{
				"dog": []string{"elroy"},
			},
			w: "http://example.com?dog=boo&dog=elroy",
		},
		// Add and merge
		{
			u: "http://example.com?dog=boo",
			q: url.Values{
				"dog": []string{"elroy"},
				"foo": []string{"bar"},
			},
			w: "http://example.com?dog=boo&dog=elroy&foo=bar",
		},
		// Multivalue merge
		{
			u: "http://example.com?dog=boo",
			q: url.Values{
				"dog": []string{"elroy", "penny"},
			},
			w: "http://example.com?dog=boo&dog=elroy&dog=penny",
		},
	}

	for i, tt := range tests {
		ur, err := url.Parse(tt.u)
		if err != nil {
			t.Errorf("case %d: failed parsing test url: %v, error: %v", i, tt.u, err)
		}

		got := MergeQuery(*ur, tt.q)
		want, err := url.Parse(tt.w)
		if err != nil {
			t.Errorf("case %d: failed parsing want url: %v, error: %v", i, tt.w, err)
		}

		if !reflect.DeepEqual(*want, got) {
			t.Errorf("case %d: want: %v, got: %v", i, *want, got)
		}
	}
}

func TestExpiresPass(t *testing.T) {
	tests := []struct {
		date    string
		exp     string
		wantTTL time.Duration
		wantOK  bool
	}{
		// Expires and Date properly set
		{
			date:    "Thu, 01 Dec 1983 22:00:00 GMT",
			exp:     "Fri, 02 Dec 1983 01:00:00 GMT",
			wantTTL: 10800 * time.Second,
			wantOK:  true,
		},
		// empty headers
		{
			date:   "",
			exp:    "",
			wantOK: false,
		},
		// lack of Expirs short-ciruits Date parsing
		{
			date:   "foo",
			exp:    "",
			wantOK: false,
		},
		// lack of Date short-ciruits Expires parsing
		{
			date:   "",
			exp:    "foo",
			wantOK: false,
		},
		// no Date
		{
			exp:     "Thu, 01 Dec 1983 22:00:00 GMT",
			wantTTL: 0,
			wantOK:  false,
		},
		// no Expires
		{
			date:    "Thu, 01 Dec 1983 22:00:00 GMT",
			wantTTL: 0,
			wantOK:  false,
		},
		// Expires set to false
		{
			date:    "Thu, 01 Dec 1983 22:00:00 GMT",
			exp:     "0",
			wantTTL: 0,
			wantOK:  false,
		},
		// Expires < Date
		{
			date:    "Fri, 02 Dec 1983 01:00:00 GMT",
			exp:     "Thu, 01 Dec 1983 22:00:00 GMT",
			wantTTL: 0,
			wantOK:  false,
		},
	}

	for i, tt := range tests {
		ttl, ok, err := expires(tt.date, tt.exp)
		if err != nil {
			t.Errorf("case %d: err=%v", i, err)
		}
		if tt.wantTTL != ttl {
			t.Errorf("case %d: want=%d got=%d", i, tt.wantTTL, ttl)
		}
		if tt.wantOK != ok {
			t.Errorf("case %d: incorrect ok value: want=%t got=%t", i, tt.wantOK, ok)
		}
	}
}

func TestExpiresFail(t *testing.T) {
	tests := []struct {
		date string
		exp  string
	}{
		// malformed Date header
		{
			date: "foo",
			exp:  "Fri, 02 Dec 1983 01:00:00 GMT",
		},
		// malformed exp header
		{
			date: "Fri, 02 Dec 1983 01:00:00 GMT",
			exp:  "bar",
		},
	}

	for i, tt := range tests {
		_, _, err := expires(tt.date, tt.exp)
		if err == nil {
			t.Errorf("case %d: expected non-nil error", i)
		}
	}
}

func TestCacheablePass(t *testing.T) {
	tests := []struct {
		headers http.Header
		wantTTL time.Duration
		wantOK  bool
	}{
		// valid Cache-Control
		{
			headers: http.Header{
				"Cache-Control": []string{"max-age=100"},
			},
			wantTTL: 100 * time.Second,
			wantOK:  true,
		},
		// valid Date/Expires
		{
			headers: http.Header{
				"Date":    []string{"Thu, 01 Dec 1983 22:00:00 GMT"},
				"Expires": []string{"Fri, 02 Dec 1983 01:00:00 GMT"},
			},
			wantTTL: 10800 * time.Second,
			wantOK:  true,
		},
		// Cache-Control supersedes Date/Expires
		{
			headers: http.Header{
				"Cache-Control": []string{"max-age=100"},
				"Date":          []string{"Thu, 01 Dec 1983 22:00:00 GMT"},
				"Expires":       []string{"Fri, 02 Dec 1983 01:00:00 GMT"},
			},
			wantTTL: 100 * time.Second,
			wantOK:  true,
		},
		// no caching headers
		{
			headers: http.Header{},
			wantOK:  false,
		},
	}

	for i, tt := range tests {
		ttl, ok, err := Cacheable(tt.headers)
		if err != nil {
			t.Errorf("case %d: err=%v", i, err)
			continue
		}
		if tt.wantTTL != ttl {
			t.Errorf("case %d: want=%d got=%d", i, tt.wantTTL, ttl)
		}
		if tt.wantOK != ok {
			t.Errorf("case %d: incorrect ok value: want=%t got=%t", i, tt.wantOK, ok)
		}
	}
}

func TestCacheableFail(t *testing.T) {
	tests := []http.Header{
		// invalid Cache-Control short-circuits
		http.Header{
			"Cache-Control": []string{"max-age"},
			"Date":          []string{"Thu, 01 Dec 1983 22:00:00 GMT"},
			"Expires":       []string{"Fri, 02 Dec 1983 01:00:00 GMT"},
		},
		// no Cache-Control, invalid Expires
		http.Header{
			"Date":    []string{"Thu, 01 Dec 1983 22:00:00 GMT"},
			"Expires": []string{"boo"},
		},
	}

	for i, tt := range tests {
		_, _, err := Cacheable(tt)
		if err == nil {
			t.Errorf("case %d: want non-nil err", i)
		}
	}
}

func TestNewResourceLocation(t *testing.T) {
	tests := []struct {
		ru   *url.URL
		id   string
		want string
	}{
		{
			ru: &url.URL{
				Scheme: "http",
				Host:   "example.com",
			},
			id:   "foo",
			want: "http://example.com/foo",
		},
		// https
		{
			ru: &url.URL{
				Scheme: "https",
				Host:   "example.com",
			},
			id:   "foo",
			want: "https://example.com/foo",
		},
		// with path
		{
			ru: &url.URL{
				Scheme: "http",
				Host:   "example.com",
				Path:   "one/two/three",
			},
			id:   "foo",
			want: "http://example.com/one/two/three/foo",
		},
		// with fragment
		{
			ru: &url.URL{
				Scheme:   "http",
				Host:     "example.com",
				Fragment: "frag",
			},
			id:   "foo",
			want: "http://example.com/foo",
		},
		// with query
		{
			ru: &url.URL{
				Scheme:   "http",
				Host:     "example.com",
				RawQuery: "dog=elroy",
			},
			id:   "foo",
			want: "http://example.com/foo",
		},
	}

	for i, tt := range tests {
		got := NewResourceLocation(tt.ru, tt.id)
		if tt.want != got {
			t.Errorf("case %d: want=%s, got=%s", i, tt.want, got)
		}
	}
}
