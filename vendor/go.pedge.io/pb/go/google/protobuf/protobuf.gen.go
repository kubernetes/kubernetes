package google_protobuf

import (
	"time"
)

var (
	// EmptyInstance is an instance of Empty.
	EmptyInstance = &Empty{}
)

// Now returns the current time as a protobuf Timestamp.
func Now() *Timestamp {
	return TimeToProto(time.Now().UTC())
}

// TimeToProto converts a go Time to a protobuf Timestamp.
func TimeToProto(t time.Time) *Timestamp {
	return &Timestamp{
		Seconds: t.UnixNano() / int64(time.Second),
		Nanos:   int32(t.UnixNano() % int64(time.Second)),
	}
}

// GoTime converts a protobuf Timestamp to a go Time.
func (t *Timestamp) GoTime() time.Time {
	if t == nil {
		return time.Unix(0, 0).UTC()
	}
	return time.Unix(
		t.Seconds,
		int64(t.Nanos),
	).UTC()
}

// Before returns true if t is before j.
func (t *Timestamp) Before(j *Timestamp) bool {
	if j == nil {
		return false
	}
	if t == nil {
		return true
	}
	if t.Seconds < j.Seconds {
		return true
	}
	if t.Seconds > j.Seconds {
		return false
	}
	return t.Nanos < j.Nanos
}

// DurationToProto converts a go Duration to a protobuf Duration.
func DurationToProto(d time.Duration) *Duration {
	return &Duration{
		Seconds: int64(d) / int64(time.Second),
		Nanos:   int32(int64(d) % int64(time.Second)),
	}
}

// GoDuration converts a protobuf Duration to a go Duration.
func (d *Duration) GoDuration() time.Duration {
	if d == nil {
		return 0
	}
	return time.Duration((d.Seconds * int64(time.Second)) + int64(d.Nanos))
}
