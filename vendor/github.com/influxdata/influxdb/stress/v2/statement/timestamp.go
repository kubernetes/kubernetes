package statement

import (
	"log"
	"time"
)

// A Timestamp contains all informaiton needed to generate timestamps for points created by InsertStatements
type Timestamp struct {
	Count    int
	Duration time.Duration
	Jitter   bool
}

// Time returns the next timestamp needed by the InsertStatement
func (t *Timestamp) Time(startDate string, series int, precision string) func() int64 {
	var start time.Time
	var err error

	if startDate == "now" {
		start = time.Now()
	} else {
		start, err = time.Parse("2006-01-02", startDate)
	}

	if err != nil {
		log.Fatalf("Error parsing start time from StartDate\n  string: %v\n  error: %v\n", startDate, err)
	}

	return nextTime(start, t.Duration, series, precision)
}

func nextTime(ti time.Time, step time.Duration, series int, precision string) func() int64 {
	t := ti
	count := 0
	return func() int64 {
		count++
		if count > series {
			t = t.Add(step)
			count = 1
		}

		var timestamp int64
		if precision == "s" {
			timestamp = t.Unix()
		} else {
			timestamp = t.UnixNano()
		}
		return timestamp
	}
}
