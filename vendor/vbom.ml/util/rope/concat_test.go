package rope

import (
	"bytes"
	"errors"
	"fmt"
	"testing"

	"github.com/bruth/assert"
)

var (
	lhs  = leaf("123")
	rhs  = leaf("456")
	tree node
)
var substrings []struct {
	orig, want node
	start, end int64
}

func init() {
	defer disableCoalesce()()

	tree = conc(lhs, rhs, 0, 0)

	substrings = []struct {
		orig, want node
		start, end int64
	}{
		{
			orig:  tree,
			want:  tree,
			start: 0,
			end:   tree.length(),
		},
		{
			orig:  tree,
			want:  tree,
			start: -100,
			end:   100,
		},
		{
			orig:  tree,
			want:  leaf("1"),
			start: 0,
			end:   1,
		},
		{
			orig:  tree,
			want:  conc(lhs, leaf("4"), 0, 0),
			start: 0,
			end:   4,
		},
		{
			orig:  tree,
			want:  leaf("1"),
			start: -100,
			end:   1,
		},
		{
			orig:  tree,
			want:  conc(leaf("3"), rhs, 0, 0),
			start: 2,
			end:   tree.length(),
		},
		{
			orig:  tree,
			want:  rhs,
			start: 3,
			end:   tree.length(),
		},
		{
			orig:  tree,
			want:  leaf(""),
			start: 3,
			end:   2,
		},
		{
			orig:  tree,
			want:  lhs,
			start: 0,
			end:   3,
		},
		{
			orig:  tree,
			want:  tree,
			start: 0,
			end:   100,
		},
		{
			orig:  tree,
			want:  leaf(""),
			start: 0,
			end:   0,
		},
		{
			orig:  tree,
			want:  leaf(""),
			start: -200,
			end:   -100,
		},
		{
			orig:  tree,
			want:  conc(leaf("23"), leaf("4"), 0, 0),
			start: 1,
			end:   4,
		},
	}
}

func TestConcatLength(t *testing.T) {
	assert.Equal(t, depthT(1), tree.depth())
	assert.Equal(t, int64(6), tree.length())
}

func TestConcatSubstr(t *testing.T) {
	defer disableCoalesce()()

	for _, ss := range substrings {
		got := ss.orig.slice(ss.start, ss.end)
		msg := fmt.Sprintf("%q[%v:%v] != %q", Rope{ss.orig}, ss.start, ss.end, Rope{got})
		assert.Equal(t, ss.want, got, msg)
	}
}

func TestConcatDropPrefix(t *testing.T) {
	defer disableCoalesce()()

	for _, ss := range substrings {
		if ss.end < ss.orig.length() {
			// Ignore non-suffix substrings
			continue
		}
		got := ss.orig.dropPrefix(ss.start)
		msg := fmt.Sprintf("%q[%v:] != %q", Rope{ss.orig}, ss.start, Rope{got})
		assert.Equal(t, ss.want, got, msg)
	}
}

func TestConcatDropPostfix(t *testing.T) {
	defer disableCoalesce()()

	for _, ss := range substrings {
		if ss.start > 0 {
			// Ignore non-prefix substrings
			continue
		}
		got := ss.orig.dropPostfix(ss.end)
		msg := fmt.Sprintf("%q[:%v] != %q", Rope{ss.orig}, ss.end, Rope{got})
		assert.Equal(t, ss.want, got, msg)
	}
}

func TestConcatWriteTo(t *testing.T) {
	buf := bytes.NewBuffer(nil)
	_, _ = tree.WriteTo(buf)
	assert.Equal(t, string(lhs+rhs), buf.String())
}

func TestConcatWalkLeaves(t *testing.T) {
	counter := 0
	err := tree.walkLeaves(func(l string) error {
		switch counter {
		case 0:
			assert.Equal(t, string(lhs), l)
		case 1:
			assert.Equal(t, string(rhs), l)
		case 2:
			t.Errorf("leaf.walkLeaves: function called too many times")
		default:
			// Ignore more than two calls, error has already been produced.
		}
		counter++

		return nil
	})
	assert.Nil(t, err)

	counter = 0
	err = tree.walkLeaves(func(l string) error {
		switch counter {
		case 0:
			assert.Equal(t, string(lhs), l)
			return errors.New("stop now")
		case 1:
			t.Errorf("leaf.walkLeaves: did not stop after returning true")
		default:
			// Ignore more than two calls, error has already been produced.
		}
		counter++

		return nil
	})
	assert.Equal(t, errors.New("stop now"), err)
}
