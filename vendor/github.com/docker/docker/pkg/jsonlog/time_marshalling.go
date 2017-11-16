// Package jsonlog provides helper functions to parse and print time (time.Time) as JSON.
package jsonlog

import (
	"errors"
	"time"
)

const (
	// RFC3339NanoFixed is our own version of RFC339Nano because we want one
	// that pads the nano seconds part with zeros to ensure
	// the timestamps are aligned in the logs.
	RFC3339NanoFixed = "2006-01-02T15:04:05.000000000Z07:00"
	// JSONFormat is the format used by FastMarshalJSON
	JSONFormat = `"` + time.RFC3339Nano + `"`
)

// FastTimeMarshalJSON avoids one of the extra allocations that
// time.MarshalJSON is making.
func FastTimeMarshalJSON(t time.Time) (string, error) {
	if y := t.Year(); y < 0 || y >= 10000 {
		// RFC 3339 is clear that years are 4 digits exactly.
		// See golang.org/issue/4556#c15 for more discussion.
		return "", errors.New("time.MarshalJSON: year outside of range [0,9999]")
	}
	return t.Format(JSONFormat), nil
}
