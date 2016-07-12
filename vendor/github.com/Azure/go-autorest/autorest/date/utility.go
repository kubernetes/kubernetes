package date

import (
	"strings"
	"time"
)

// ParseTime to parse Time string to specified format.
func ParseTime(format string, t string) (d time.Time, err error) {
	return time.Parse(format, strings.ToUpper(t))
}
