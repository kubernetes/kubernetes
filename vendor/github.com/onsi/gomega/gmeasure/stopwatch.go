package gmeasure

import "time"

/*
Stopwatch provides a convenient abstraction for recording durations.  There are two ways to make a Stopwatch:

You can make a Stopwatch from an Experiment via experiment.NewStopwatch().  This is how you first get a hold of a Stopwatch.

You can subsequently call stopwatch.NewStopwatch() to get a fresh Stopwatch.
This is only necessary if you need to record durations on a different goroutine as a single Stopwatch is not considered thread-safe.

The Stopwatch starts as soon as it is created.  You can Pause() the stopwatch and Reset() it as needed.

Stopwatches refer back to their parent Experiment.  They use this reference to record any measured durations back with the Experiment.
*/
type Stopwatch struct {
	Experiment    *Experiment
	t             time.Time
	pauseT        time.Time
	pauseDuration time.Duration
	running       bool
}

func newStopwatch(experiment *Experiment) *Stopwatch {
	return &Stopwatch{
		Experiment: experiment,
		t:          time.Now(),
		running:    true,
	}
}

/*
NewStopwatch returns a new Stopwatch pointing to the same Experiment as this Stopwatch
*/
func (s *Stopwatch) NewStopwatch() *Stopwatch {
	return newStopwatch(s.Experiment)
}

/*
Record captures the amount of time that has passed since the Stopwatch was created or most recently Reset().  It records the duration on it's associated Experiment in a Measurement with the passed-in name.

Record takes all the decorators that experiment.RecordDuration takes (e.g. Annotation("...") can be used to annotate this duration)

Note that Record does not Reset the Stopwatch.  It does, however, return the Stopwatch so the following pattern is common:

    stopwatch := experiment.NewStopwatch()
    // first expensive operation
    stopwatch.Record("first operation").Reset() //records the duration of the first operation and resets the stopwatch.
    // second expensive operation
    stopwatch.Record("second operation").Reset() //records the duration of the second operation and resets the stopwatch.

omitting the Reset() after the first operation would cause the duration recorded for the second operation to include the time elapsed by both the first _and_ second operations.

The Stopwatch must be running (i.e. not paused) when Record is called.
*/
func (s *Stopwatch) Record(name string, args ...interface{}) *Stopwatch {
	if !s.running {
		panic("stopwatch is not running - call Resume or Reset before calling Record")
	}
	duration := time.Since(s.t) - s.pauseDuration
	s.Experiment.RecordDuration(name, duration, args...)
	return s
}

/*
Reset resets the Stopwatch.  Subsequent recorded durations will measure the time elapsed from the moment Reset was called.
If the Stopwatch was Paused it is unpaused after calling Reset.
*/
func (s *Stopwatch) Reset() *Stopwatch {
	s.running = true
	s.t = time.Now()
	s.pauseDuration = 0
	return s
}

/*
Pause pauses the Stopwatch.  While pasued the Stopwatch does not accumulate elapsed time.  This is useful for ignoring expensive operations that are incidental to the behavior you are attempting to characterize.
Note: You must call Resume() before you can Record() subsequent measurements.

For example:

    stopwatch := experiment.NewStopwatch()
    // first expensive operation
    stopwatch.Record("first operation").Reset()
    // second expensive operation - part 1
    stopwatch.Pause()
    // something expensive that we don't care about
    stopwatch.Resume()
    // second expensive operation - part 2
    stopwatch.Record("second operation").Reset() // the recorded duration captures the time elapsed during parts 1 and 2 of the second expensive operation, but not the bit in between


The Stopwatch must be running when Pause is called.
*/
func (s *Stopwatch) Pause() *Stopwatch {
	if !s.running {
		panic("stopwatch is not running - call Resume or Reset before calling Pause")
	}
	s.running = false
	s.pauseT = time.Now()
	return s
}

/*
Resume resumes a paused Stopwatch.  Any time that elapses after Resume is called will be accumulated as elapsed time when a subsequent duration is Recorded.

The Stopwatch must be Paused when Resume is called
*/
func (s *Stopwatch) Resume() *Stopwatch {
	if s.running {
		panic("stopwatch is running - call Pause before calling Resume")
	}
	s.running = true
	s.pauseDuration = s.pauseDuration + time.Since(s.pauseT)
	return s
}
