package swag

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSplitHostPort(t *testing.T) {
	data := []struct {
		Input string
		Host  string
		Port  int
		Err   bool
	}{
		{"localhost:3933", "localhost", 3933, false},
		{"localhost:yellow", "", -1, true},
		{"localhost", "", -1, true},
		{"localhost:", "", -1, true},
		{"localhost:3933", "localhost", 3933, false},
	}

	for _, e := range data {
		h, p, err := SplitHostPort(e.Input)
		if (!e.Err && assert.NoError(t, err)) || (e.Err && assert.Error(t, err)) {
			assert.Equal(t, e.Host, h)
			assert.Equal(t, e.Port, p)
		}
	}
}
