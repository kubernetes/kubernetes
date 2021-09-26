package yaml_test

import (
	"strings"
	"testing"

	. "gopkg.in/check.v1"
	"gopkg.in/yaml.v2"
)

var limitTests = []struct {
	name  string
	data  []byte
	error string
}{
	{
		name:  "1000kb of maps with 100 aliases",
		data:  []byte(`{a: &a [{a}` + strings.Repeat(`,{a}`, 1000*1024/4-100) + `], b: &b [*a` + strings.Repeat(`,*a`, 99) + `]}`),
		error: "yaml: document contains excessive aliasing",
	}, {
		name:  "1000kb of deeply nested slices",
		data:  []byte(strings.Repeat(`[`, 1000*1024)),
		error: "yaml: exceeded max depth of 10000",
	}, {
		name:  "1000kb of deeply nested maps",
		data:  []byte("x: " + strings.Repeat(`{`, 1000*1024)),
		error: "yaml: exceeded max depth of 10000",
	}, {
		name:  "1000kb of deeply nested indents",
		data:  []byte(strings.Repeat(`- `, 1000*1024)),
		error: "yaml: exceeded max depth of 10000",
	}, {
		name: "1000kb of 1000-indent lines",
		data: []byte(strings.Repeat(strings.Repeat(`- `, 1000)+"\n", 1024/2)),
	},
	{name: "1kb of maps", data: []byte(`a: &a [{a}` + strings.Repeat(`,{a}`, 1*1024/4-1) + `]`)},
	{name: "10kb of maps", data: []byte(`a: &a [{a}` + strings.Repeat(`,{a}`, 10*1024/4-1) + `]`)},
	{name: "100kb of maps", data: []byte(`a: &a [{a}` + strings.Repeat(`,{a}`, 100*1024/4-1) + `]`)},
	{name: "1000kb of maps", data: []byte(`a: &a [{a}` + strings.Repeat(`,{a}`, 1000*1024/4-1) + `]`)},
	{name: "1000kb slice nested at max-depth", data: []byte(strings.Repeat(`[`, 10000) + `1` + strings.Repeat(`,1`, 1000*1024/2-20000-1) + strings.Repeat(`]`, 10000))},
	{name: "1000kb slice nested in maps at max-depth", data: []byte("{a,b:\n" + strings.Repeat(" {a,b:", 10000-2) + ` [1` + strings.Repeat(",1", 1000*1024/2-6*10000-1) + `]` + strings.Repeat(`}`, 10000-1))},
	{name: "1000kb of 10000-nested lines", data: []byte(strings.Repeat(`- `+strings.Repeat(`[`, 10000)+strings.Repeat(`]`, 10000)+"\n", 1000*1024/20000))},
}

func (s *S) TestLimits(c *C) {
	if testing.Short() {
		return
	}
	for _, tc := range limitTests {
		var v interface{}
		err := yaml.Unmarshal(tc.data, &v)
		if len(tc.error) > 0 {
			c.Assert(err, ErrorMatches, tc.error, Commentf("testcase: %s", tc.name))
		} else {
			c.Assert(err, IsNil, Commentf("testcase: %s", tc.name))
		}
	}
}

func Benchmark1000KB100Aliases(b *testing.B) {
	benchmark(b, "1000kb of maps with 100 aliases")
}
func Benchmark1000KBDeeplyNestedSlices(b *testing.B) {
	benchmark(b, "1000kb of deeply nested slices")
}
func Benchmark1000KBDeeplyNestedMaps(b *testing.B) {
	benchmark(b, "1000kb of deeply nested maps")
}
func Benchmark1000KBDeeplyNestedIndents(b *testing.B) {
	benchmark(b, "1000kb of deeply nested indents")
}
func Benchmark1000KB1000IndentLines(b *testing.B) {
	benchmark(b, "1000kb of 1000-indent lines")
}
func Benchmark1KBMaps(b *testing.B) {
	benchmark(b, "1kb of maps")
}
func Benchmark10KBMaps(b *testing.B) {
	benchmark(b, "10kb of maps")
}
func Benchmark100KBMaps(b *testing.B) {
	benchmark(b, "100kb of maps")
}
func Benchmark1000KBMaps(b *testing.B) {
	benchmark(b, "1000kb of maps")
}

func BenchmarkDeepSlice(b *testing.B) {
	benchmark(b, "1000kb slice nested at max-depth")
}

func BenchmarkDeepFlow(b *testing.B) {
	benchmark(b, "1000kb slice nested in maps at max-depth")
}

func Benchmark1000KBMaxDepthNested(b *testing.B) {
	benchmark(b, "1000kb of 10000-nested lines")
}

func benchmark(b *testing.B, name string) {
	for _, t := range limitTests {
		if t.name != name {
			continue
		}

		b.ResetTimer()

		for i := 0; i < b.N; i++ {
			var v interface{}
			err := yaml.Unmarshal(t.data, &v)
			if len(t.error) > 0 {
				if err == nil {
					b.Errorf("expected error, got none")
				} else if err.Error() != t.error {
					b.Errorf("expected error '%s', got '%s'", t.error, err.Error())
				}
			} else {
				if err != nil {
					b.Errorf("unexpected error: %v", err)
				}
			}
		}

		return
	}

	b.Errorf("testcase %q not found", name)
}
