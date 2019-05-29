package protocol

import (
	"strconv"
	"time"
)

// Names of time formats supported by the SDK
const (
	RFC822TimeFormatName  = "rfc822"
	ISO8601TimeFormatName = "iso8601"
	UnixTimeFormatName    = "unixTimestamp"
)

// Time formats supported by the SDK
const (
	// RFC 7231#section-7.1.1.1 timetamp format. e.g Tue, 29 Apr 2014 18:30:38 GMT
	RFC822TimeFormat = "Mon, 2 Jan 2006 15:04:05 GMT"

	// RFC3339 a subset of the ISO8601 timestamp format. e.g 2014-04-29T18:30:38Z
	ISO8601TimeFormat = "2006-01-02T15:04:05Z"
)

// IsKnownTimestampFormat returns if the timestamp format name
// is know to the SDK's protocols.
func IsKnownTimestampFormat(name string) bool {
	switch name {
	case RFC822TimeFormatName:
		fallthrough
	case ISO8601TimeFormatName:
		fallthrough
	case UnixTimeFormatName:
		return true
	default:
		return false
	}
}

// FormatTime returns a string value of the time.
func FormatTime(name string, t time.Time) string {
	t = t.UTC()

	switch name {
	case RFC822TimeFormatName:
		return t.Format(RFC822TimeFormat)
	case ISO8601TimeFormatName:
		return t.Format(ISO8601TimeFormat)
	case UnixTimeFormatName:
		return strconv.FormatInt(t.Unix(), 10)
	default:
		panic("unknown timestamp format name, " + name)
	}
}

// ParseTime attempts to parse the time given the format. Returns
// the time if it was able to be parsed, and fails otherwise.
func ParseTime(formatName, value string) (time.Time, error) {
	switch formatName {
	case RFC822TimeFormatName:
		return time.Parse(RFC822TimeFormat, value)
	case ISO8601TimeFormatName:
		return time.Parse(ISO8601TimeFormat, value)
	case UnixTimeFormatName:
		v, err := strconv.ParseFloat(value, 64)
		if err != nil {
			return time.Time{}, err
		}
		return time.Unix(int64(v), 0), nil
	default:
		panic("unknown timestamp format name, " + formatName)
	}
}
