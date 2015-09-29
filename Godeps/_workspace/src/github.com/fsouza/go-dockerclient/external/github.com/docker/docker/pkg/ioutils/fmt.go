package ioutils

import (
	"fmt"
	"io"
)

// FprintfIfNotEmpty prints the string value if it's not empty
func FprintfIfNotEmpty(w io.Writer, format, value string) (int, error) {
	if value != "" {
		return fmt.Fprintf(w, format, value)
	}
	return 0, nil
}
