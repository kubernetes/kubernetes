package date

import (
	"regexp"
	"time"
)

// Azure reports time in UTC but it doesn't include the 'Z' time zone suffix in some cases.
const (
	azureUtcFormatJSON = `"2006-01-02T15:04:05.999999999"`
	azureUtcFormat     = "2006-01-02T15:04:05.999999999"
	rfc3339JSON        = `"` + time.RFC3339Nano + `"`
	rfc3339            = time.RFC3339Nano
	tzOffsetRegex      = `(Z|z|\+|-)(\d+:\d+)*"*$`
)

// Time defines a type similar to time.Time but assumes a layout of RFC3339 date-time (i.e.,
// 2006-01-02T15:04:05Z).
type Time struct {
	time.Time
}

// MarshalBinary preserves the Time as a byte array conforming to RFC3339 date-time (i.e.,
// 2006-01-02T15:04:05Z).
func (t Time) MarshalBinary() ([]byte, error) {
	return t.Time.MarshalText()
}

// UnmarshalBinary reconstitutes a Time saved as a byte array conforming to RFC3339 date-time
// (i.e., 2006-01-02T15:04:05Z).
func (t *Time) UnmarshalBinary(data []byte) error {
	return t.UnmarshalText(data)
}

// MarshalJSON preserves the Time as a JSON string conforming to RFC3339 date-time (i.e.,
// 2006-01-02T15:04:05Z).
func (t Time) MarshalJSON() (json []byte, err error) {
	return t.Time.MarshalJSON()
}

// UnmarshalJSON reconstitutes the Time from a JSON string conforming to RFC3339 date-time
// (i.e., 2006-01-02T15:04:05Z).
func (t *Time) UnmarshalJSON(data []byte) (err error) {
	timeFormat := azureUtcFormatJSON
	match, err := regexp.Match(tzOffsetRegex, data)
	if err != nil {
		return err
	} else if match {
		timeFormat = rfc3339JSON
	}
	t.Time, err = ParseTime(timeFormat, string(data))
	return err
}

// MarshalText preserves the Time as a byte array conforming to RFC3339 date-time (i.e.,
// 2006-01-02T15:04:05Z).
func (t Time) MarshalText() (text []byte, err error) {
	return t.Time.MarshalText()
}

// UnmarshalText reconstitutes a Time saved as a byte array conforming to RFC3339 date-time
// (i.e., 2006-01-02T15:04:05Z).
func (t *Time) UnmarshalText(data []byte) (err error) {
	timeFormat := azureUtcFormat
	match, err := regexp.Match(tzOffsetRegex, data)
	if err != nil {
		return err
	} else if match {
		timeFormat = rfc3339
	}
	t.Time, err = ParseTime(timeFormat, string(data))
	return err
}

// String returns the Time formatted as an RFC3339 date-time string (i.e.,
// 2006-01-02T15:04:05Z).
func (t Time) String() string {
	// Note: time.Time.String does not return an RFC3339 compliant string, time.Time.MarshalText does.
	b, err := t.MarshalText()
	if err != nil {
		return ""
	}
	return string(b)
}

// ToTime returns a Time as a time.Time
func (t Time) ToTime() time.Time {
	return t.Time
}
