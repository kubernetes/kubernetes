package toml // import "github.com/influxdata/influxdb/toml"

import (
	"fmt"
	"strconv"
	"time"
)

// maxInt is the largest integer representable by a word (architecture dependent).
const maxInt = int64(^uint(0) >> 1)

// Duration is a TOML wrapper type for time.Duration.
type Duration time.Duration

func (d Duration) String() string {
	return time.Duration(d).String()
}

// UnmarshalText parses a TOML value into a duration value.
func (d *Duration) UnmarshalText(text []byte) error {
	// Ignore if there is no value set.
	if len(text) == 0 {
		return nil
	}

	// Otherwise parse as a duration formatted string.
	duration, err := time.ParseDuration(string(text))
	if err != nil {
		return err
	}

	// Set duration and return.
	*d = Duration(duration)
	return nil
}

// MarshalText converts a duration to a string for decoding toml
func (d Duration) MarshalText() (text []byte, err error) {
	return []byte(d.String()), nil
}

// Size represents a TOML parseable file size.
// Users can specify size using "m" for megabytes and "g" for gigabytes.
type Size int

// UnmarshalText parses a byte size from text.
func (s *Size) UnmarshalText(text []byte) error {
	// Parse numeric portion of value.
	length := len(string(text))
	size, err := strconv.ParseInt(string(text[:length-1]), 10, 64)
	if err != nil {
		return err
	}

	// Parse unit of measure ("m", "g", etc).
	switch suffix := text[len(text)-1]; suffix {
	case 'm':
		size *= 1 << 20 // MB
	case 'g':
		size *= 1 << 30 // GB
	default:
		return fmt.Errorf("unknown size suffix: %c", suffix)
	}

	// Check for overflow.
	if size > maxInt {
		return fmt.Errorf("size %d cannot be represented by an int", size)
	}

	*s = Size(size)
	return nil
}
