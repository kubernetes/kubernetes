package backoff

import (
	"math/rand/v2"
	"time"
)

/*
ExponentialBackOff is a backoff implementation that increases the backoff
period for each retry attempt using a randomization function that grows exponentially.

NextBackOff() is calculated using the following formula:

	randomized interval =
	    RetryInterval * (random value in range [1 - RandomizationFactor, 1 + RandomizationFactor])

In other words NextBackOff() will range between the randomization factor
percentage below and above the retry interval.

For example, given the following parameters:

	RetryInterval = 2
	RandomizationFactor = 0.5
	Multiplier = 2

the actual backoff period used in the next retry attempt will range between 1 and 3 seconds,
multiplied by the exponential, that is, between 2 and 6 seconds.

Note: MaxInterval caps the RetryInterval and not the randomized interval.

Example: Given the following default arguments, for 9 tries the sequence will be:

	Request #  RetryInterval (seconds)  Randomized Interval (seconds)

	 1          0.5                     [0.25,   0.75]
	 2          0.75                    [0.375,  1.125]
	 3          1.125                   [0.562,  1.687]
	 4          1.687                   [0.8435, 2.53]
	 5          2.53                    [1.265,  3.795]
	 6          3.795                   [1.897,  5.692]
	 7          5.692                   [2.846,  8.538]
	 8          8.538                   [4.269, 12.807]
	 9         12.807                   [6.403, 19.210]

Note: Implementation is not thread-safe.
*/
type ExponentialBackOff struct {
	InitialInterval     time.Duration
	RandomizationFactor float64
	Multiplier          float64
	MaxInterval         time.Duration

	currentInterval time.Duration
}

// Default values for ExponentialBackOff.
const (
	DefaultInitialInterval     = 500 * time.Millisecond
	DefaultRandomizationFactor = 0.5
	DefaultMultiplier          = 1.5
	DefaultMaxInterval         = 60 * time.Second
)

// NewExponentialBackOff creates an instance of ExponentialBackOff using default values.
func NewExponentialBackOff() *ExponentialBackOff {
	return &ExponentialBackOff{
		InitialInterval:     DefaultInitialInterval,
		RandomizationFactor: DefaultRandomizationFactor,
		Multiplier:          DefaultMultiplier,
		MaxInterval:         DefaultMaxInterval,
	}
}

// Reset the interval back to the initial retry interval and restarts the timer.
// Reset must be called before using b.
func (b *ExponentialBackOff) Reset() {
	b.currentInterval = b.InitialInterval
}

// NextBackOff calculates the next backoff interval using the formula:
//
//	Randomized interval = RetryInterval * (1 ± RandomizationFactor)
func (b *ExponentialBackOff) NextBackOff() time.Duration {
	if b.currentInterval == 0 {
		b.currentInterval = b.InitialInterval
	}

	next := getRandomValueFromInterval(b.RandomizationFactor, rand.Float64(), b.currentInterval)
	b.incrementCurrentInterval()
	return next
}

// Increments the current interval by multiplying it with the multiplier.
func (b *ExponentialBackOff) incrementCurrentInterval() {
	// Check for overflow, if overflow is detected set the current interval to the max interval.
	if float64(b.currentInterval) >= float64(b.MaxInterval)/b.Multiplier {
		b.currentInterval = b.MaxInterval
	} else {
		b.currentInterval = time.Duration(float64(b.currentInterval) * b.Multiplier)
	}
}

// Returns a random value from the following interval:
//
//	[currentInterval - randomizationFactor * currentInterval, currentInterval + randomizationFactor * currentInterval].
func getRandomValueFromInterval(randomizationFactor, random float64, currentInterval time.Duration) time.Duration {
	if randomizationFactor == 0 {
		return currentInterval // make sure no randomness is used when randomizationFactor is 0.
	}
	var delta = randomizationFactor * float64(currentInterval)
	var minInterval = float64(currentInterval) - delta
	var maxInterval = float64(currentInterval) + delta

	// Get a random value from the range [minInterval, maxInterval].
	// The formula used below has a +1 because if the minInterval is 1 and the maxInterval is 3 then
	// we want a 33% chance for selecting either 1, 2 or 3.
	return time.Duration(minInterval + (random * (maxInterval - minInterval + 1)))
}
