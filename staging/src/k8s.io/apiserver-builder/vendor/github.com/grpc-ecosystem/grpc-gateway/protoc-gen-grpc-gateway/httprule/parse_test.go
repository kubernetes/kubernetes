package httprule

import (
	"flag"
	"fmt"
	"reflect"
	"testing"

	"github.com/golang/glog"
)

func TestTokenize(t *testing.T) {
	for _, spec := range []struct {
		src    string
		tokens []string
	}{
		{
			src:    "",
			tokens: []string{eof},
		},
		{
			src:    "v1",
			tokens: []string{"v1", eof},
		},
		{
			src:    "v1/b",
			tokens: []string{"v1", "/", "b", eof},
		},
		{
			src:    "v1/endpoint/*",
			tokens: []string{"v1", "/", "endpoint", "/", "*", eof},
		},
		{
			src:    "v1/endpoint/**",
			tokens: []string{"v1", "/", "endpoint", "/", "**", eof},
		},
		{
			src: "v1/b/{bucket_name=*}",
			tokens: []string{
				"v1", "/",
				"b", "/",
				"{", "bucket_name", "=", "*", "}",
				eof,
			},
		},
		{
			src: "v1/b/{bucket_name=buckets/*}",
			tokens: []string{
				"v1", "/",
				"b", "/",
				"{", "bucket_name", "=", "buckets", "/", "*", "}",
				eof,
			},
		},
		{
			src: "v1/b/{bucket_name=buckets/*}/o",
			tokens: []string{
				"v1", "/",
				"b", "/",
				"{", "bucket_name", "=", "buckets", "/", "*", "}", "/",
				"o",
				eof,
			},
		},
		{
			src: "v1/b/{bucket_name=buckets/*}/o/{name}",
			tokens: []string{
				"v1", "/",
				"b", "/",
				"{", "bucket_name", "=", "buckets", "/", "*", "}", "/",
				"o", "/", "{", "name", "}",
				eof,
			},
		},
		{
			src: "v1/a=b&c=d;e=f:g/endpoint.rdf",
			tokens: []string{
				"v1", "/",
				"a=b&c=d;e=f:g", "/",
				"endpoint.rdf",
				eof,
			},
		},
	} {
		tokens, verb := tokenize(spec.src)
		if got, want := tokens, spec.tokens; !reflect.DeepEqual(got, want) {
			t.Errorf("tokenize(%q) = %q, _; want %q, _", spec.src, got, want)
		}
		if got, want := verb, ""; got != want {
			t.Errorf("tokenize(%q) = _, %q; want _, %q", spec.src, got, want)
		}

		src := fmt.Sprintf("%s:%s", spec.src, "LOCK")
		tokens, verb = tokenize(src)
		if got, want := tokens, spec.tokens; !reflect.DeepEqual(got, want) {
			t.Errorf("tokenize(%q) = %q, _; want %q, _", src, got, want)
		}
		if got, want := verb, "LOCK"; got != want {
			t.Errorf("tokenize(%q) = _, %q; want _, %q", src, got, want)
		}
	}
}

func TestParseSegments(t *testing.T) {
	flag.Set("v", "3")
	for _, spec := range []struct {
		tokens []string
		want   []segment
	}{
		{
			tokens: []string{"v1", eof},
			want: []segment{
				literal("v1"),
			},
		},
		{
			tokens: []string{"-._~!$&'()*+,;=:@", eof},
			want: []segment{
				literal("-._~!$&'()*+,;=:@"),
			},
		},
		{
			tokens: []string{"%e7%ac%ac%e4%b8%80%e7%89%88", eof},
			want: []segment{
				literal("%e7%ac%ac%e4%b8%80%e7%89%88"),
			},
		},
		{
			tokens: []string{"v1", "/", "*", eof},
			want: []segment{
				literal("v1"),
				wildcard{},
			},
		},
		{
			tokens: []string{"v1", "/", "**", eof},
			want: []segment{
				literal("v1"),
				deepWildcard{},
			},
		},
		{
			tokens: []string{"{", "name", "}", eof},
			want: []segment{
				variable{
					path: "name",
					segments: []segment{
						wildcard{},
					},
				},
			},
		},
		{
			tokens: []string{"{", "name", "=", "*", "}", eof},
			want: []segment{
				variable{
					path: "name",
					segments: []segment{
						wildcard{},
					},
				},
			},
		},
		{
			tokens: []string{"{", "field", ".", "nested", ".", "nested2", "=", "*", "}", eof},
			want: []segment{
				variable{
					path: "field.nested.nested2",
					segments: []segment{
						wildcard{},
					},
				},
			},
		},
		{
			tokens: []string{"{", "name", "=", "a", "/", "b", "/", "*", "}", eof},
			want: []segment{
				variable{
					path: "name",
					segments: []segment{
						literal("a"),
						literal("b"),
						wildcard{},
					},
				},
			},
		},
		{
			tokens: []string{
				"v1", "/",
				"{",
				"name", ".", "nested", ".", "nested2",
				"=",
				"a", "/", "b", "/", "*",
				"}", "/",
				"o", "/",
				"{",
				"another_name",
				"=",
				"a", "/", "b", "/", "*", "/", "c",
				"}", "/",
				"**",
				eof},
			want: []segment{
				literal("v1"),
				variable{
					path: "name.nested.nested2",
					segments: []segment{
						literal("a"),
						literal("b"),
						wildcard{},
					},
				},
				literal("o"),
				variable{
					path: "another_name",
					segments: []segment{
						literal("a"),
						literal("b"),
						wildcard{},
						literal("c"),
					},
				},
				deepWildcard{},
			},
		},
	} {
		p := parser{tokens: spec.tokens}
		segs, err := p.topLevelSegments()
		if err != nil {
			t.Errorf("parser{%q}.segments() failed with %v; want success", spec.tokens, err)
			continue
		}
		if got, want := segs, spec.want; !reflect.DeepEqual(got, want) {
			t.Errorf("parser{%q}.segments() = %#v; want %#v", spec.tokens, got, want)
		}
		if got := p.tokens; len(got) > 0 {
			t.Errorf("p.tokens = %q; want []; spec.tokens=%q", got, spec.tokens)
		}
	}
}

func TestParseSegmentsWithErrors(t *testing.T) {
	flag.Set("v", "3")
	for _, spec := range []struct {
		tokens []string
	}{
		{
			// double slash
			tokens: []string{"/", eof},
		},
		{
			// invalid literal
			tokens: []string{"a?b", eof},
		},
		{
			// invalid percent-encoding
			tokens: []string{"%", eof},
		},
		{
			// invalid percent-encoding
			tokens: []string{"%2", eof},
		},
		{
			// invalid percent-encoding
			tokens: []string{"a%2z", eof},
		},
		{
			// empty segments
			tokens: []string{eof},
		},
		{
			// unterminated variable
			tokens: []string{"{", "name", eof},
		},
		{
			// unterminated variable
			tokens: []string{"{", "name", "=", eof},
		},
		{
			// unterminated variable
			tokens: []string{"{", "name", "=", "*", eof},
		},
		{
			// empty component in field path
			tokens: []string{"{", "name", ".", "}", eof},
		},
		{
			// empty component in field path
			tokens: []string{"{", "name", ".", ".", "nested", "}", eof},
		},
		{
			// invalid character in identifier
			tokens: []string{"{", "field-name", "}", eof},
		},
		{
			// no slash between segments
			tokens: []string{"v1", "endpoint", eof},
		},
		{
			// no slash between segments
			tokens: []string{"v1", "{", "name", "}", eof},
		},
	} {
		p := parser{tokens: spec.tokens}
		segs, err := p.topLevelSegments()
		if err == nil {
			t.Errorf("parser{%q}.segments() succeeded; want InvalidTemplateError; accepted %#v", spec.tokens, segs)
			continue
		}
		glog.V(1).Info(err)
	}
}
