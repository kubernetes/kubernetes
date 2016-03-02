package timeutils

import (
	"strconv"
	"strings"
	"time"
)

// GetTimestamp tries to parse given string as golang duration,
// then RFC3339 time and finally as a Unix timestamp. If
// any of these were successful, it returns a Unix timestamp
// as string otherwise returns the given value back.
// In case of duration input, the returned timestamp is computed
// as the given reference time minus the amount of the duration.
func GetTimestamp(value string, reference time.Time) string {
	if d, err := time.ParseDuration(value); value != "0" && err == nil {
		return strconv.FormatInt(reference.Add(-d).Unix(), 10)
	}

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
