package progress

import (
	"fmt"
	"time"

	units "github.com/docker/go-units"
)

// Bytes converts a regular int64 to human readable type.
type Bytes int64

// String returns the string representation of bytes
func (b Bytes) String() string {
	return units.CustomSize("%02.1f %s", float64(b), 1024.0, []string{"B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB", "YiB"})
}

// BytesPerSecond is the rate in seconds for byte operations
type BytesPerSecond int64

// NewBytesPerSecond returns the rate that n bytes were written in the provided duration
func NewBytesPerSecond(n int64, duration time.Duration) BytesPerSecond {
	return BytesPerSecond(float64(n) / duration.Seconds())
}

// String returns the string representation of the rate
func (bps BytesPerSecond) String() string {
	return fmt.Sprintf("%v/s", Bytes(bps))
}
