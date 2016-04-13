package longpath

import (
	"strings"
	"testing"
)

func TestStandardLongPath(t *testing.T) {
	c := `C:\simple\path`
	longC := AddPrefix(c)
	if !strings.EqualFold(longC, `\\?\C:\simple\path`) {
		t.Errorf("Wrong long path returned. Original = %s ; Long = %s", c, longC)
	}
}

func TestUNCLongPath(t *testing.T) {
	c := `\\server\share\path`
	longC := AddPrefix(c)
	if !strings.EqualFold(longC, `\\?\UNC\server\share\path`) {
		t.Errorf("Wrong UNC long path returned. Original = %s ; Long = %s", c, longC)
	}
}
