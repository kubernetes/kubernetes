// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package uritemplates

import (
	"fmt"
	"log"
	"net/url"
	"testing"
)

func ExampleExpand() {
	values := map[string]string{
		"user": "golang",
		"repo": "go",
	}
	expanded, _, err := Expand("https://api.github.com/repos{/user,repo}", values)
	if err != nil {
		log.Fatalf("Error expanding template: %v", err)
	}
	fmt.Println(expanded)
	// Output:
	// https://api.github.com/repos/golang/go
}

func TestExpand(t *testing.T) {
	testCases := []struct {
		tmpl   string
		values map[string]string
		want   string
	}{
		// These examples come from the RFC:
		// http://tools.ietf.org/html/rfc6570
		{
			tmpl:   "http://www.example.com/foo{?query,number}",
			values: map[string]string{"query": "mycelium", "number": "100"},
			want:   "http://www.example.com/foo?query=mycelium&number=100",
		},
		{
			tmpl:   "http://www.example.com/foo{?query,number}",
			values: map[string]string{"query": "mycelium"},
			want:   "http://www.example.com/foo?query=mycelium",
		},
		{
			tmpl:   "http://www.example.com/foo{?query,number}",
			values: map[string]string{},
			want:   "http://www.example.com/foo",
		},
	}

	for _, tt := range testCases {
		exp, _, err := Expand(tt.tmpl, tt.values)
		if err != nil {
			t.Errorf("Expand(%q, %v) error: %v", tt.tmpl, tt.values, err)
			continue
		}
		if exp != tt.want {
			t.Errorf("Expand(%q, %v)\ngot  %q\nwant %q", tt.tmpl, tt.values, exp, tt.want)
		}
	}
}

func TestExpandRFCLevels(t *testing.T) {
	values := map[string]string{
		"dub":   "me/too",
		"hello": "Hello World!",
		"half":  "50%",
		"var":   "value",
		"who":   "fred",
		"base":  "http://example.com/home/",
		"path":  "/foo/bar",
		"semi":  ";",
		"v":     "6",
		"x":     "1024",
		"y":     "768",
		"empty": "",
		// undef not mapped.
	}
	testCases := []struct {
		tmpl, want string
	}{
		// These examples come from the RFC levels specification.
		// http://tools.ietf.org/html/rfc6570
		// Level 1 examples.
		{tmpl: "{var}", want: "value"},
		{tmpl: "{hello}", want: "Hello%20World%21"},

		// Level 2 examples.
		{tmpl: "{+var}", want: "value"},
		{tmpl: "{+hello}", want: "Hello%20World!"},
		{tmpl: "{+path}/here", want: "/foo/bar/here"},
		{tmpl: "here?ref={+path}", want: "here?ref=/foo/bar"},
		{tmpl: "X{#var}", want: "X#value"},
		{tmpl: "X{#hello}", want: "X#Hello%20World!"},

		// Level 3 examples.
		{tmpl: "map?{x,y}", want: "map?1024,768"},
		{tmpl: "{x,hello,y}", want: "1024,Hello%20World%21,768"},
		{tmpl: "{+x,hello,y}", want: "1024,Hello%20World!,768"},
		{tmpl: "{+path,x}/here", want: "/foo/bar,1024/here"},
		{tmpl: "{#x,hello,y}", want: "#1024,Hello%20World!,768"},
		{tmpl: "{#path,x}/here", want: "#/foo/bar,1024/here"},
		{tmpl: "X{.var}", want: "X.value"},
		{tmpl: "X{.x,y}", want: "X.1024.768"},
		{tmpl: "{/var}", want: "/value"},
		{tmpl: "{/var,x}/here", want: "/value/1024/here"},
		{tmpl: "{;x,y}", want: ";x=1024;y=768"},
		{tmpl: "{;x,y,empty}", want: ";x=1024;y=768;empty"},
		{tmpl: "{?x,y}", want: "?x=1024&y=768"},
		{tmpl: "{?x,y,empty}", want: "?x=1024&y=768&empty="},
		{tmpl: "?fixed=yes{&x}", want: "?fixed=yes&x=1024"},
		{tmpl: "{&x,y,empty}", want: "&x=1024&y=768&empty="},

		{tmpl: "{var:3}", want: "val"},
		{tmpl: "{var:30}", want: "value"},
		{tmpl: "{+path:6}/here", want: "/foo/b/here"},
		{tmpl: "{#path:6}/here", want: "#/foo/b/here"},
		{tmpl: "X{.var:3}", want: "X.val"},
		{tmpl: "{/var:1,var}", want: "/v/value"},
		{tmpl: "{;hello:5}", want: ";hello=Hello"},
		{tmpl: "{?var:3}", want: "?var=val"},
		{tmpl: "{&var:3}", want: "&var=val"},

		// 2.4.1 Prefix values.
		{tmpl: "{var}", want: "value"},
		{tmpl: "{var:20}", want: "value"},
		{tmpl: "{var:3}", want: "val"},
		{tmpl: "{semi}", want: "%3B"},
		{tmpl: "{semi:2}", want: "%3B"},
		// 3.2.2.  Simple String Expansion: {var}
		{tmpl: "{var}", want: "value"},
		{tmpl: "{hello}", want: "Hello%20World%21"},
		{tmpl: "{half}", want: "50%25"},
		{tmpl: "O{empty}X", want: "OX"},
		{tmpl: "O{undef}X", want: "OX"},
		{tmpl: "{x,y}", want: "1024,768"},
		{tmpl: "{x,hello,y}", want: "1024,Hello%20World%21,768"},
		{tmpl: "?{x,empty}", want: "?1024,"},
		{tmpl: "?{x,undef}", want: "?1024"},
		{tmpl: "?{undef,y}", want: "?768"},
		{tmpl: "{var:3}", want: "val"},
		{tmpl: "{var:30}", want: "value"},
		// 3.2.3.  Reserved Expansion: {+var}
		{tmpl: "{+var}", want: "value"},
		{tmpl: "{+hello}", want: "Hello%20World!"},
		{tmpl: "{+half}", want: "50%25"},
		{tmpl: "{base}index", want: "http%3A%2F%2Fexample.com%2Fhome%2Findex"},
		{tmpl: "{+base}index", want: "http://example.com/home/index"},
		{tmpl: "O{+empty}X", want: "OX"},
		{tmpl: "O{+undef}X", want: "OX"},
		{tmpl: "{+path}/here", want: "/foo/bar/here"},
		{tmpl: "here?ref={+path}", want: "here?ref=/foo/bar"},
		{tmpl: "up{+path}{var}/here", want: "up/foo/barvalue/here"},
		{tmpl: "{+x,hello,y}", want: "1024,Hello%20World!,768"},
		{tmpl: "{+path,x}/here", want: "/foo/bar,1024/here"},
		{tmpl: "{+path:6}/here", want: "/foo/b/here"},
		// 3.2.4.  Fragment Expansion: {#var}
		{tmpl: "{#var}", want: "#value"},
		{tmpl: "{#hello}", want: "#Hello%20World!"},
		{tmpl: "{#half}", want: "#50%25"},
		{tmpl: "foo{#empty}", want: "foo#"},
		{tmpl: "foo{#undef}", want: "foo"},
		{tmpl: "{#x,hello,y}", want: "#1024,Hello%20World!,768"},
		{tmpl: "{#path,x}/here", want: "#/foo/bar,1024/here"},
		{tmpl: "{#path:6}/here", want: "#/foo/b/here"},
		// 3.2.5.  Label Expansion with Dot-Prefix: {.var}
		{tmpl: "{.who}", want: ".fred"},
		{tmpl: "{.who,who}", want: ".fred.fred"},
		{tmpl: "{.half,who}", want: ".50%25.fred"},
		{tmpl: "X{.var}", want: "X.value"},
		{tmpl: "X{.empty}", want: "X."},
		{tmpl: "X{.undef}", want: "X"},
		{tmpl: "X{.var:3}", want: "X.val"},
		// 3.2.6.  Path Segment Expansion: {/var}
		{tmpl: "{/who}", want: "/fred"},
		{tmpl: "{/who,who}", want: "/fred/fred"},
		{tmpl: "{/half,who}", want: "/50%25/fred"},
		{tmpl: "{/who,dub}", want: "/fred/me%2Ftoo"},
		{tmpl: "{/var}", want: "/value"},
		{tmpl: "{/var,empty}", want: "/value/"},
		{tmpl: "{/var,undef}", want: "/value"},
		{tmpl: "{/var,x}/here", want: "/value/1024/here"},
		{tmpl: "{/var:1,var}", want: "/v/value"},
		// 3.2.7.  Path-Style Parameter Expansion: {;var}
		{tmpl: "{;who}", want: ";who=fred"},
		{tmpl: "{;half}", want: ";half=50%25"},
		{tmpl: "{;empty}", want: ";empty"},
		{tmpl: "{;v,empty,who}", want: ";v=6;empty;who=fred"},
		{tmpl: "{;v,bar,who}", want: ";v=6;who=fred"},
		{tmpl: "{;x,y}", want: ";x=1024;y=768"},
		{tmpl: "{;x,y,empty}", want: ";x=1024;y=768;empty"},
		{tmpl: "{;x,y,undef}", want: ";x=1024;y=768"},
		{tmpl: "{;hello:5}", want: ";hello=Hello"},
		// 3.2.8.  Form-Style Query Expansion: {?var}
		{tmpl: "{?who}", want: "?who=fred"},
		{tmpl: "{?half}", want: "?half=50%25"},
		{tmpl: "{?x,y}", want: "?x=1024&y=768"},
		{tmpl: "{?x,y,empty}", want: "?x=1024&y=768&empty="},
		{tmpl: "{?x,y,undef}", want: "?x=1024&y=768"},
		{tmpl: "{?var:3}", want: "?var=val"},
		// 3.2.9.  Form-Style Query Continuation: {&var}
		{tmpl: "{&who}", want: "&who=fred"},
		{tmpl: "{&half}", want: "&half=50%25"},
		{tmpl: "?fixed=yes{&x}", want: "?fixed=yes&x=1024"},
		{tmpl: "{&x,y,empty}", want: "&x=1024&y=768&empty="},
		{tmpl: "{&x,y,undef}", want: "&x=1024&y=768"},
		{tmpl: "{&var:3}", want: "&var=val"},
	}
	for _, tt := range testCases {
		esc, unesc, err := Expand(tt.tmpl, values)
		if err != nil {
			t.Errorf("Expand(%q) error: %v", tt.tmpl, err)
			continue
		}
		if esc != tt.want {
			t.Errorf("Expand(%q)\ngot  %q\nwant %q", tt.tmpl, esc, tt.want)
		}
		// Check that the escaped form is equivalent to unescaped.
		urlUnesc, err := url.QueryUnescape(esc)
		if err != nil {
			t.Errorf("Expand(%q) gave invalid escaping %q: %v", tt.tmpl, esc, err)
			continue
		}
		if urlUnesc != unesc {
			t.Errorf("Expand(%q) gave inconsistent escaped/unescaped\nunescaped %q\nescaped   %q\nwhich is  %q", tt.tmpl, unesc, esc, urlUnesc)
		}
	}
}

func TestExpandUnescaped(t *testing.T) {
	testCases := []struct {
		tmpl, wantEsc, wantUnesc string
		values                   map[string]string
	}{
		{
			tmpl: "/foo/{bucket}/bar",
			values: map[string]string{
				"bucket": "simple",
			},
			wantEsc:   "/foo/simple/bar",
			wantUnesc: "/foo/simple/bar",
		},
		{
			tmpl: "/foo/{bucket}/bar",
			values: map[string]string{
				"bucket": "path/with/slash",
			},
			wantEsc:   "/foo/path%2Fwith%2Fslash/bar",
			wantUnesc: "/foo/path/with/slash/bar",
		},
		{
			tmpl: "/foo/{+bucket}/bar",
			values: map[string]string{
				"bucket": "path/with/slash",
			},
			wantEsc:   "/foo/path/with/slash/bar",
			wantUnesc: "/foo/path/with/slash/bar",
		},
		{
			tmpl: "/foo/{bucket}/bar",
			values: map[string]string{
				"bucket": "double%2Fescaped",
			},
			wantEsc:   "/foo/double%252Fescaped/bar",
			wantUnesc: "/foo/double%2Fescaped/bar",
		},
	}
	for _, tt := range testCases {
		esc, unesc, err := Expand(tt.tmpl, tt.values)
		if err != nil {
			t.Errorf("Expand(%q) error: %v", tt.tmpl, err)
			continue
		}
		if esc != tt.wantEsc || unesc != tt.wantUnesc {
			t.Errorf("Expand(%q)\ngot  esc=%q, unesc=%q\nwant esc=%q, unesc=%q", tt.tmpl, esc, unesc, tt.wantEsc, tt.wantUnesc)
		}
	}
}
