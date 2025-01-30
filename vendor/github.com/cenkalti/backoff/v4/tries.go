package backoff

import "time"

/*
WithMaxRetries creates a wrapper around another BackOff, which will
return Stop if NextBackOff() has been called too many times since
the last time Reset() was called

Note: Implementation is not thread-safe.
*/
func WithMaxRetries(b BackOff, max uint64) BackOff {
	return &backOffTries{delegate: b, maxTries: max}
}

type backOffTries struct {
	delegate BackOff
	maxTries uint64
	numTries uint64
}

func (b *backOffTries) NextBackOff() time.Duration {
	if b.maxTries == 0 {
		return Stop
	}
	if b.maxTries > 0 {
		if b.maxTries <= b.numTries {
			return Stop
		}
		b.numTries++
	}
	return b.delegate.NextBackOff()
}

func (b *backOffTries) Reset() {
	b.numTries = 0
	b.delegate.Reset()
}
