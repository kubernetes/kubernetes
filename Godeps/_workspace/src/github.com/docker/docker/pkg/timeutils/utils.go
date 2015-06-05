package timeutils

import (
	"strconv"
	"strings"
	"time"
)

// GetTimestamp tries to parse given string as RFC3339 time
// or Unix timestamp (with seconds precision), if successful
//returns a Unix timestamp as string otherwise returns value back.
func GetTimestamp(value string) string {
	var format string
	if strings.Contains(value, ".") {
		format = time.RFC3339Nano
	} else {
		format = time.RFC3339
	}

	loc := time.FixedZone(time.Now().Zone())
	if len(value) < len(format) {
		format = format[:len(value)]
	}
	t, err := time.ParseInLocation(format, value, loc)
	if err != nil {
		return value
	}
	return strconv.FormatInt(t.Unix(), 10)
}
