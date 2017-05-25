// Copyright (c) 2017 Uber Technologies, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package metrics

import (
	"time"
)

// StartStopwatch begins recording the executing time of an event, returning
// a Stopwatch that should be used to stop the recording the time for
// that event.  Multiple events can be occurring simultaneously each
// represented by different active Stopwatches
func StartStopwatch(timer Timer) Stopwatch {
	return Stopwatch{t: timer, start: time.Now()}
}

// A Stopwatch tracks the execution time of a specific event
type Stopwatch struct {
	t     Timer
	start time.Time
}

// Stop stops executing of the stopwatch and records the amount of elapsed time
func (s Stopwatch) Stop() {
	s.t.Record(s.ElapsedTime())
}

// ElapsedTime returns the amount of elapsed time (in time.Duration)
func (s Stopwatch) ElapsedTime() time.Duration {
	return time.Since(s.start)
}
