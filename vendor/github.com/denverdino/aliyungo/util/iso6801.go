package util

import (
	"fmt"
	"strconv"
	"time"
)

// GetISO8601TimeStamp gets timestamp string in ISO8601 format
func GetISO8601TimeStamp(ts time.Time) string {
	t := ts.UTC()
	return fmt.Sprintf("%04d-%02d-%02dT%02d:%02d:%02dZ", t.Year(), t.Month(), t.Day(), t.Hour(), t.Minute(), t.Second())
}

const formatISO8601 = "2006-01-02T15:04:05Z"
const jsonFormatISO8601 = `"` + formatISO8601 + `"`
const formatISO8601withoutSeconds = "2006-01-02T15:04Z"
const jsonFormatISO8601withoutSeconds = `"` + formatISO8601withoutSeconds + `"`

// A ISO6801Time represents a time in ISO8601 format
type ISO6801Time time.Time

// New constructs a new iso8601.Time instance from an existing
// time.Time instance.  This causes the nanosecond field to be set to
// 0, and its time zone set to a fixed zone with no offset from UTC
// (but it is *not* UTC itself).
func NewISO6801Time(t time.Time) ISO6801Time {
	return ISO6801Time(time.Date(
		t.Year(),
		t.Month(),
		t.Day(),
		t.Hour(),
		t.Minute(),
		t.Second(),
		0,
		time.UTC,
	))
}

// IsDefault checks if the time is default
func (it *ISO6801Time) IsDefault() bool {
	return *it == ISO6801Time{}
}

// MarshalJSON serializes the ISO6801Time into JSON string
func (it ISO6801Time) MarshalJSON() ([]byte, error) {
	return []byte(time.Time(it).Format(jsonFormatISO8601)), nil
}

// UnmarshalJSON deserializes the ISO6801Time from JSON string
func (it *ISO6801Time) UnmarshalJSON(data []byte) error {
	str := string(data)

	if str == "\"\"" || len(data) == 0 {
		return nil
	}
	var t time.Time
	var err error
	if str[0] == '"' {
		t, err = time.ParseInLocation(jsonFormatISO8601, str, time.UTC)
		if err != nil {
			t, err = time.ParseInLocation(jsonFormatISO8601withoutSeconds, str, time.UTC)
		}
	} else {
		var i int64
		i, err = strconv.ParseInt(str, 10, 64)
		if err == nil {
			t = time.Unix(i/1000, i%1000)
		}
	}
	if err == nil {
		*it = ISO6801Time(t)
	}
	return err
}

// String returns the time in ISO6801Time format
func (it ISO6801Time) String() string {
	return time.Time(it).Format(formatISO8601)
}
