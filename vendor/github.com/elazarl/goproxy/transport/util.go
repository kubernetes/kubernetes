package transport

import (
	"fmt"
	"strings"
)

type badStringError struct {
	what string
	str  string
}

func (e *badStringError) Error() string { return fmt.Sprintf("%s %q", e.what, e.str) }

func hasPort(s string) bool { return strings.LastIndex(s, ":") > strings.LastIndex(s, "]") }
