package rope

import (
	"bytes"
	"errors"
	"testing"

	"github.com/bruth/assert"
)

func TestLeaf(t *testing.T) {
	v := leaf("foo")

	assert.Equal(t, depthT(0), v.depth())
	assert.Equal(t, int64(3), v.length())

	assert.Equal(t, v, v.slice(0, v.length()))
	assert.Equal(t, v, v.slice(-100, 100))
	assert.Equal(t, leaf("f"), v.slice(0, 1))
	assert.Equal(t, leaf("f"), v.slice(-100, 1))
	assert.Equal(t, leaf("o"), v.slice(2, v.length()))
	assert.Equal(t, leaf(""), v.slice(2, 1))

	assert.Equal(t, v, v.dropPrefix(0))
	assert.Equal(t, v, v.dropPrefix(-100))
	assert.Equal(t, leaf("oo"), v.dropPrefix(1))
	assert.Equal(t, leaf(""), v.dropPrefix(3))
	assert.Equal(t, leaf(""), v.dropPrefix(100))

	assert.Equal(t, v, v.dropPostfix(3))
	assert.Equal(t, v, v.dropPostfix(100))
	assert.Equal(t, leaf("fo"), v.dropPostfix(2))
	assert.Equal(t, leaf(""), v.dropPostfix(0))
	assert.Equal(t, leaf(""), v.dropPostfix(-100))

	buf := bytes.NewBuffer(nil)
	_, _ = v.WriteTo(buf)
	assert.Equal(t, "foo", buf.String())

	counter := 0
	err := v.walkLeaves(func(l string) error {
		if counter > 0 {
			t.Errorf("leaf.walkLeaves: function called too many times")
			return errors.New("called a lot")
		}
		counter++

		assert.Equal(t, string(v), l)

		return nil
	})
	assert.Nil(t, err)
}
