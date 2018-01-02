package httprule

import (
	"fmt"
	"testing"
)

func TestTemplateStringer(t *testing.T) {
	for _, spec := range []struct {
		segs []segment
		want string
	}{
		{
			segs: []segment{
				literal("v1"),
			},
			want: "/v1",
		},
		{
			segs: []segment{
				wildcard{},
			},
			want: "/*",
		},
		{
			segs: []segment{
				deepWildcard{},
			},
			want: "/**",
		},
		{
			segs: []segment{
				variable{
					path: "name",
					segments: []segment{
						literal("a"),
					},
				},
			},
			want: "/{name=a}",
		},
		{
			segs: []segment{
				variable{
					path: "name",
					segments: []segment{
						literal("a"),
						wildcard{},
						literal("b"),
					},
				},
			},
			want: "/{name=a/*/b}",
		},
		{
			segs: []segment{
				literal("v1"),
				variable{
					path: "name",
					segments: []segment{
						literal("a"),
						wildcard{},
						literal("b"),
					},
				},
				literal("c"),
				variable{
					path: "field.nested",
					segments: []segment{
						wildcard{},
						literal("d"),
					},
				},
				wildcard{},
				literal("e"),
				deepWildcard{},
			},
			want: "/v1/{name=a/*/b}/c/{field.nested=*/d}/*/e/**",
		},
	} {
		tmpl := template{segments: spec.segs}
		if got, want := tmpl.String(), spec.want; got != want {
			t.Errorf("%#v.String() = %q; want %q", tmpl, got, want)
		}

		tmpl.verb = "LOCK"
		if got, want := tmpl.String(), fmt.Sprintf("%s:LOCK", spec.want); got != want {
			t.Errorf("%#v.String() = %q; want %q", tmpl, got, want)
		}
	}
}
