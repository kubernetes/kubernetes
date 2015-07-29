//
// Written by Maxim Khitrov (November 2012)
//

package flowrate

import (
	"math"
	"strconv"
	"time"
)

// clockRate is the resolution and precision of clock().
const clockRate = 20 * time.Millisecond

// czero is the process start time rounded down to the nearest clockRate
// increment.
var czero = time.Duration(time.Now().UnixNano()) / clockRate * clockRate

// clock returns a low resolution timestamp relative to the process start time.
func clock() time.Duration {
	return time.Duration(time.Now().UnixNano())/clockRate*clockRate - czero
}

// clockToTime converts a clock() timestamp to an absolute time.Time value.
func clockToTime(c time.Duration) time.Time {
	return time.Unix(0, int64(czero+c))
}

// clockRound returns d rounded to the nearest clockRate increment.
func clockRound(d time.Duration) time.Duration {
	return (d + clockRate>>1) / clockRate * clockRate
}

// round returns x rounded to the nearest int64 (non-negative values only).
func round(x float64) int64 {
	if _, frac := math.Modf(x); frac >= 0.5 {
		return int64(math.Ceil(x))
	}
	return int64(math.Floor(x))
}

// Percent represents a percentage in increments of 1/1000th of a percent.
type Percent uint32

// percentOf calculates what percent of the total is x.
func percentOf(x, total float64) Percent {
	if x < 0 || total <= 0 {
		return 0
	} else if p := round(x / total * 1e5); p <= math.MaxUint32 {
		return Percent(p)
	}
	return Percent(math.MaxUint32)
}

func (p Percent) Float() float64 {
	return float64(p) * 1e-3
}

func (p Percent) String() string {
	var buf [12]byte
	b := strconv.AppendUint(buf[:0], uint64(p)/1000, 10)
	n := len(b)
	b = strconv.AppendUint(b, 1000+uint64(p)%1000, 10)
	b[n] = '.'
	return string(append(b, '%'))
}
