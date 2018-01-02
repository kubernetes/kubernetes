package runtime

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/grpc-ecosystem/grpc-gateway/utilities"
)

const (
	validVersion = 1
	anything     = 0
)

func TestNewPattern(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string
		verb string

		stackSizeWant, tailLenWant int
	}{
		{},
		{
			ops:           []int{int(utilities.OpNop), anything},
			stackSizeWant: 0,
			tailLenWant:   0,
		},
		{
			ops:           []int{int(utilities.OpPush), anything},
			stackSizeWant: 1,
			tailLenWant:   0,
		},
		{
			ops:           []int{int(utilities.OpLitPush), 0},
			pool:          []string{"abc"},
			stackSizeWant: 1,
			tailLenWant:   0,
		},
		{
			ops:           []int{int(utilities.OpPushM), anything},
			stackSizeWant: 1,
			tailLenWant:   0,
		},
		{
			ops: []int{
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
			},
			stackSizeWant: 1,
			tailLenWant:   0,
		},
		{
			ops: []int{
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 0,
			},
			pool:          []string{"abc"},
			stackSizeWant: 1,
			tailLenWant:   0,
		},
		{
			ops: []int{
				int(utilities.OpPush), anything,
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPushM), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
			},
			pool:          []string{"lit1", "lit2", "var1"},
			stackSizeWant: 4,
			tailLenWant:   0,
		},
		{
			ops: []int{
				int(utilities.OpPushM), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 2,
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
			},
			pool:          []string{"lit1", "lit2", "var1"},
			stackSizeWant: 2,
			tailLenWant:   2,
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPushM), anything,
				int(utilities.OpLitPush), 2,
				int(utilities.OpConcatN), 3,
				int(utilities.OpLitPush), 3,
				int(utilities.OpCapture), 4,
			},
			pool:          []string{"lit1", "lit2", "lit3", "lit4", "var1"},
			stackSizeWant: 4,
			tailLenWant:   2,
		},
		{
			ops:           []int{int(utilities.OpLitPush), 0},
			pool:          []string{"abc"},
			verb:          "LOCK",
			stackSizeWant: 1,
			tailLenWant:   0,
		},
	} {
		pat, err := NewPattern(validVersion, spec.ops, spec.pool, spec.verb)
		if err != nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want success", validVersion, spec.ops, spec.pool, spec.verb, err)
			continue
		}
		if got, want := pat.stacksize, spec.stackSizeWant; got != want {
			t.Errorf("pat.stacksize = %d; want %d", got, want)
		}
		if got, want := pat.tailLen, spec.tailLenWant; got != want {
			t.Errorf("pat.stacksize = %d; want %d", got, want)
		}
	}
}

func TestNewPatternWithWrongOp(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string
		verb string
	}{
		{
			// op code out of bound
			ops: []int{-1, anything},
		},
		{
			// op code out of bound
			ops: []int{int(utilities.OpEnd), 0},
		},
		{
			// odd number of items
			ops: []int{int(utilities.OpPush)},
		},
		{
			// negative index
			ops:  []int{int(utilities.OpLitPush), -1},
			pool: []string{"abc"},
		},
		{
			// index out of bound
			ops:  []int{int(utilities.OpLitPush), 1},
			pool: []string{"abc"},
		},
		{
			// negative # of segments
			ops:  []int{int(utilities.OpConcatN), -1},
			pool: []string{"abc"},
		},
		{
			// negative index
			ops:  []int{int(utilities.OpCapture), -1},
			pool: []string{"abc"},
		},
		{
			// index out of bound
			ops:  []int{int(utilities.OpCapture), 1},
			pool: []string{"abc"},
		},
		{
			// pushM appears twice
			ops: []int{
				int(utilities.OpPushM), anything,
				int(utilities.OpLitPush), 0,
				int(utilities.OpPushM), anything,
			},
			pool: []string{"abc"},
		},
	} {
		_, err := NewPattern(validVersion, spec.ops, spec.pool, spec.verb)
		if err == nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) succeeded; want failure with %v", validVersion, spec.ops, spec.pool, spec.verb, ErrInvalidPattern)
			continue
		}
		if err != ErrInvalidPattern {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want failure with %v", validVersion, spec.ops, spec.pool, spec.verb, err, ErrInvalidPattern)
			continue
		}
	}
}

func TestNewPatternWithStackUnderflow(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string
		verb string
	}{
		{
			ops: []int{int(utilities.OpConcatN), 1},
		},
		{
			ops:  []int{int(utilities.OpCapture), 0},
			pool: []string{"abc"},
		},
	} {
		_, err := NewPattern(validVersion, spec.ops, spec.pool, spec.verb)
		if err == nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) succeeded; want failure with %v", validVersion, spec.ops, spec.pool, spec.verb, ErrInvalidPattern)
			continue
		}
		if err != ErrInvalidPattern {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want failure with %v", validVersion, spec.ops, spec.pool, spec.verb, err, ErrInvalidPattern)
			continue
		}
	}
}

func TestMatch(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string
		verb string

		match    []string
		notMatch []string
	}{
		{
			match:    []string{""},
			notMatch: []string{"example"},
		},
		{
			ops:      []int{int(utilities.OpNop), anything},
			match:    []string{""},
			notMatch: []string{"example", "path/to/example"},
		},
		{
			ops:      []int{int(utilities.OpPush), anything},
			match:    []string{"abc", "def"},
			notMatch: []string{"", "abc/def"},
		},
		{
			ops:      []int{int(utilities.OpLitPush), 0},
			pool:     []string{"v1"},
			match:    []string{"v1"},
			notMatch: []string{"", "v2"},
		},
		{
			ops:   []int{int(utilities.OpPushM), anything},
			match: []string{"", "abc", "abc/def", "abc/def/ghi"},
		},
		{
			ops: []int{
				int(utilities.OpPushM), anything,
				int(utilities.OpLitPush), 0,
			},
			pool:  []string{"tail"},
			match: []string{"tail", "abc/tail", "abc/def/tail"},
			notMatch: []string{
				"", "abc", "abc/def",
				"tail/extra", "abc/tail/extra", "abc/def/tail/extra",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 2,
			},
			pool:  []string{"v1", "bucket", "name"},
			match: []string{"v1/bucket/my-bucket", "v1/bucket/our-bucket"},
			notMatch: []string{
				"",
				"v1",
				"v1/bucket",
				"v2/bucket/my-bucket",
				"v1/pubsub/my-topic",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPushM), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
			},
			pool: []string{"v1", "o", "name"},
			match: []string{
				"v1/o",
				"v1/o/my-bucket",
				"v1/o/our-bucket",
				"v1/o/my-bucket/dir",
				"v1/o/my-bucket/dir/dir2",
				"v1/o/my-bucket/dir/dir2/obj",
			},
			notMatch: []string{
				"",
				"v1",
				"v2/o/my-bucket",
				"v1/b/my-bucket",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
				int(utilities.OpLitPush), 3,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 4,
			},
			pool: []string{"v2", "b", "name", "o", "oname"},
			match: []string{
				"v2/b/my-bucket/o/obj",
				"v2/b/our-bucket/o/obj",
				"v2/b/my-bucket/o/dir",
			},
			notMatch: []string{
				"",
				"v2",
				"v2/b",
				"v2/b/my-bucket",
				"v2/b/my-bucket/o",
			},
		},
		{
			ops:      []int{int(utilities.OpLitPush), 0},
			pool:     []string{"v1"},
			verb:     "LOCK",
			match:    []string{"v1:LOCK"},
			notMatch: []string{"v1", "LOCK"},
		},
	} {
		pat, err := NewPattern(validVersion, spec.ops, spec.pool, spec.verb)
		if err != nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want success", validVersion, spec.ops, spec.pool, spec.verb, err)
			continue
		}

		for _, path := range spec.match {
			_, err = pat.Match(segments(path))
			if err != nil {
				t.Errorf("pat.Match(%q) failed with %v; want success; pattern = (%v, %q)", path, err, spec.ops, spec.pool)
			}
		}

		for _, path := range spec.notMatch {
			_, err = pat.Match(segments(path))
			if err == nil {
				t.Errorf("pat.Match(%q) succeeded; want failure with %v; pattern = (%v, %q)", path, ErrNotMatch, spec.ops, spec.pool)
				continue
			}
			if err != ErrNotMatch {
				t.Errorf("pat.Match(%q) failed with %v; want failure with %v; pattern = (%v, %q)", spec.notMatch, err, ErrNotMatch, spec.ops, spec.pool)
			}
		}
	}
}

func TestMatchWithBinding(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string
		path string
		verb string

		want map[string]string
	}{
		{
			want: make(map[string]string),
		},
		{
			ops:  []int{int(utilities.OpNop), anything},
			want: make(map[string]string),
		},
		{
			ops:  []int{int(utilities.OpPush), anything},
			path: "abc",
			want: make(map[string]string),
		},
		{
			ops:  []int{int(utilities.OpPush), anything},
			verb: "LOCK",
			path: "abc:LOCK",
			want: make(map[string]string),
		},
		{
			ops:  []int{int(utilities.OpLitPush), 0},
			pool: []string{"endpoint"},
			path: "endpoint",
			want: make(map[string]string),
		},
		{
			ops:  []int{int(utilities.OpPushM), anything},
			path: "abc/def/ghi",
			want: make(map[string]string),
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 2,
			},
			pool: []string{"v1", "bucket", "name"},
			path: "v1/bucket/my-bucket",
			want: map[string]string{
				"name": "my-bucket",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 2,
			},
			pool: []string{"v1", "bucket", "name"},
			verb: "LOCK",
			path: "v1/bucket/my-bucket:LOCK",
			want: map[string]string{
				"name": "my-bucket",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPushM), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
			},
			pool: []string{"v1", "o", "name"},
			path: "v1/o/my-bucket/dir/dir2/obj",
			want: map[string]string{
				"name": "o/my-bucket/dir/dir2/obj",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPushM), anything,
				int(utilities.OpLitPush), 2,
				int(utilities.OpConcatN), 3,
				int(utilities.OpCapture), 4,
				int(utilities.OpLitPush), 3,
			},
			pool: []string{"v1", "o", ".ext", "tail", "name"},
			path: "v1/o/my-bucket/dir/dir2/obj/.ext/tail",
			want: map[string]string{
				"name": "o/my-bucket/dir/dir2/obj/.ext",
			},
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
				int(utilities.OpLitPush), 3,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 4,
			},
			pool: []string{"v2", "b", "name", "o", "oname"},
			path: "v2/b/my-bucket/o/obj",
			want: map[string]string{
				"name":  "b/my-bucket",
				"oname": "obj",
			},
		},
	} {
		pat, err := NewPattern(validVersion, spec.ops, spec.pool, spec.verb)
		if err != nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want success", validVersion, spec.ops, spec.pool, spec.verb, err)
			continue
		}

		got, err := pat.Match(segments(spec.path))
		if err != nil {
			t.Errorf("pat.Match(%q) failed with %v; want success; pattern = (%v, %q)", spec.path, err, spec.ops, spec.pool)
		}
		if !reflect.DeepEqual(got, spec.want) {
			t.Errorf("pat.Match(%q) = %q; want %q; pattern = (%v, %q)", spec.path, got, spec.want, spec.ops, spec.pool)
		}
	}
}

func segments(path string) (components []string, verb string) {
	if path == "" {
		return nil, ""
	}
	components = strings.Split(path, "/")
	l := len(components)
	c := components[l-1]
	if idx := strings.LastIndex(c, ":"); idx >= 0 {
		components[l-1], verb = c[:idx], c[idx+1:]
	}
	return components, verb
}

func TestPatternString(t *testing.T) {
	for _, spec := range []struct {
		ops  []int
		pool []string

		want string
	}{
		{
			want: "/",
		},
		{
			ops:  []int{int(utilities.OpNop), anything},
			want: "/",
		},
		{
			ops:  []int{int(utilities.OpPush), anything},
			want: "/*",
		},
		{
			ops:  []int{int(utilities.OpLitPush), 0},
			pool: []string{"endpoint"},
			want: "/endpoint",
		},
		{
			ops:  []int{int(utilities.OpPushM), anything},
			want: "/**",
		},
		{
			ops: []int{
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
			},
			want: "/*",
		},
		{
			ops: []int{
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 1,
				int(utilities.OpCapture), 0,
			},
			pool: []string{"name"},
			want: "/{name=*}",
		},
		{
			ops: []int{
				int(utilities.OpLitPush), 0,
				int(utilities.OpLitPush), 1,
				int(utilities.OpPush), anything,
				int(utilities.OpConcatN), 2,
				int(utilities.OpCapture), 2,
				int(utilities.OpLitPush), 3,
				int(utilities.OpPushM), anything,
				int(utilities.OpLitPush), 4,
				int(utilities.OpConcatN), 3,
				int(utilities.OpCapture), 6,
				int(utilities.OpLitPush), 5,
			},
			pool: []string{"v1", "buckets", "bucket_name", "objects", ".ext", "tail", "name"},
			want: "/v1/{bucket_name=buckets/*}/{name=objects/**/.ext}/tail",
		},
	} {
		p, err := NewPattern(validVersion, spec.ops, spec.pool, "")
		if err != nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want success", validVersion, spec.ops, spec.pool, "", err)
			continue
		}
		if got, want := p.String(), spec.want; got != want {
			t.Errorf("%#v.String() = %q; want %q", p, got, want)
		}

		verb := "LOCK"
		p, err = NewPattern(validVersion, spec.ops, spec.pool, verb)
		if err != nil {
			t.Errorf("NewPattern(%d, %v, %q, %q) failed with %v; want success", validVersion, spec.ops, spec.pool, verb, err)
			continue
		}
		if got, want := p.String(), fmt.Sprintf("%s:%s", spec.want, verb); got != want {
			t.Errorf("%#v.String() = %q; want %q", p, got, want)
		}
	}
}
