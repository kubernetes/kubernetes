package statement

import (
	"time"
)

// ResponseTime is a struct that contains `Value`
// `Time` pairing.
type ResponseTime struct {
	Value int
	Time  time.Time
}

// NewResponseTime returns a new response time
// with value `v` and time `time.Now()`.
func NewResponseTime(v int) ResponseTime {
	r := ResponseTime{Value: v, Time: time.Now()}
	return r
}

// ResponseTimes is a slice of response times
type ResponseTimes []ResponseTime

// Implements the `Len` method for the
// sort.Interface type
func (rs ResponseTimes) Len() int {
	return len(rs)
}

// Implements the `Less` method for the
// sort.Interface type
func (rs ResponseTimes) Less(i, j int) bool {
	return rs[i].Value < rs[j].Value
}

// Implements the `Swap` method for the
// sort.Interface type
func (rs ResponseTimes) Swap(i, j int) {
	rs[i], rs[j] = rs[j], rs[i]
}
