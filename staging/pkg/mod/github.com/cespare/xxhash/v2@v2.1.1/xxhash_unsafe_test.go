// +build !appengine

package xxhash

import (
	"strings"
	"testing"
)

func TestStringAllocs(t *testing.T) {
	longStr := strings.Repeat("a", 1000)
	t.Run("Sum64String", func(t *testing.T) {
		testAllocs(t, func() {
			sink = Sum64String(longStr)
		})
	})
	t.Run("Digest.WriteString", func(t *testing.T) {
		testAllocs(t, func() {
			d := New()
			d.WriteString(longStr)
			sink = d.Sum64()
		})
	})
}
