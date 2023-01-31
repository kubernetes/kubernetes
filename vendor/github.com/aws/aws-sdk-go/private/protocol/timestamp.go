package protocol

import (
	"bytes"
	"fmt"
	"math"
	"strconv"
	"time"

	"github.com/aws/aws-sdk-go/internal/sdkmath"
)

// Names of time formats supported by the SDK
const (
	RFC822TimeFormatName  = "rfc822"
	ISO8601TimeFormatName = "iso8601"
	UnixTimeFormatName    = "unixTimestamp"
)

// Time formats supported by the SDK
// Output time is intended to not contain decimals
const (
	// RFC 7231#section-7.1.1.1 timetamp format. e.g Tue, 29 Apr 2014 18:30:38 GMT
	RFC822TimeFormat                           = "Mon, 2 Jan 2006 15:04:05 GMT"
	rfc822TimeFormatSingleDigitDay             = "Mon, _2 Jan 2006 15:04:05 GMT"
	rfc822TimeFormatSingleDigitDayTwoDigitYear = "Mon, _2 Jan 06 15:04:05 GMT"

	// This format is used for output time without seconds precision
	RFC822OutputTimeFormat = "Mon, 02 Jan 2006 15:04:05 GMT"

	// RFC3339 a subset of the ISO8601 timestamp format. e.g 2014-04-29T18:30:38Z
	ISO8601TimeFormat    = "2006-01-02T15:04:05.999999999Z"
	iso8601TimeFormatNoZ = "2006-01-02T15:04:05.999999999"

	// This format is used for output time with fractional second precision up to milliseconds
	ISO8601OutputTimeFormat = "2006-01-02T15:04:05.999999999Z"
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
	t = t.UTC().Truncate(time.Millisecond)

	switch name {
	case RFC822TimeFormatName:
		return t.Format(RFC822OutputTimeFormat)
	case ISO8601TimeFormatName:
		return t.Format(ISO8601OutputTimeFormat)
	case UnixTimeFormatName:
		ms := t.UnixNano() / int64(time.Millisecond)
		return strconv.FormatFloat(float64(ms)/1e3, 'f', -1, 64)
	default:
		panic("unknown timestamp format name, " + name)
	}
}

// ParseTime attempts to parse the time given the format. Returns
// the time if it was able to be parsed, and fails otherwise.
func ParseTime(formatName, value string) (time.Time, error) {
	switch formatName {
	case RFC822TimeFormatName: // Smithy HTTPDate format
		return tryParse(value,
			RFC822TimeFormat,
			rfc822TimeFormatSingleDigitDay,
			rfc822TimeFormatSingleDigitDayTwoDigitYear,
			time.RFC850,
			time.ANSIC,
		)
	case ISO8601TimeFormatName: // Smithy DateTime format
		return tryParse(value,
			ISO8601TimeFormat,
			iso8601TimeFormatNoZ,
			time.RFC3339Nano,
			time.RFC3339,
		)
	case UnixTimeFormatName:
		v, err := strconv.ParseFloat(value, 64)
		_, dec := math.Modf(v)
		dec = sdkmath.Round(dec*1e3) / 1e3 //Rounds 0.1229999 to 0.123
		if err != nil {
			return time.Time{}, err
		}
		return time.Unix(int64(v), int64(dec*(1e9))), nil
	default:
		panic("unknown timestamp format name, " + formatName)
	}
}

func tryParse(v string, formats ...string) (time.Time, error) {
	var errs parseErrors
	for _, f := range formats {
		t, err := time.Parse(f, v)
		if err != nil {
			errs = append(errs, parseError{
				Format: f,
				Err:    err,
			})
			continue
		}
		return t, nil
	}

	return time.Time{}, fmt.Errorf("unable to parse time string, %v", errs)
}

type parseErrors []parseError

func (es parseErrors) Error() string {
	var s bytes.Buffer
	for _, e := range es {
		fmt.Fprintf(&s, "\n * %q: %v", e.Format, e.Err)
	}

	return "parse errors:" + s.String()
}

type parseError struct {
	Format string
	Err    error
}
