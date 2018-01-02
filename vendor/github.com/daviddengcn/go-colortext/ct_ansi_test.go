// +build !windows

package ct

import (
	"testing"

	"github.com/golangplus/testing/assert"
)

func TestAnsiText(t *testing.T) {
	assert.Equal(t, "ansiText", ansiText(None, false, None, false), "")
	assert.Equal(t, "ansiText", ansiText(Red, false, None, false), "\x1b[0;31m")
	assert.Equal(t, "ansiText", ansiText(Red, true, None, false), "\x1b[0;31;1m")
	assert.Equal(t, "ansiText", ansiText(None, false, Green, false), "\x1b[0;42m")
	assert.Equal(t, "ansiText", ansiText(Red, false, Green, false), "\x1b[0;31;42m")
	assert.Equal(t, "ansiText", ansiText(Red, true, Green, false), "\x1b[0;31;1;42m")
}
