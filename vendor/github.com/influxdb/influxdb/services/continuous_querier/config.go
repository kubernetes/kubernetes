package continuous_querier

import (
	"time"

	"github.com/influxdb/influxdb/toml"
)

const (
	DefaultRecomputePreviousN = 2

	DefaultRecomputeNoOlderThan = 10 * time.Minute

	DefaultComputeRunsPerInterval = 10

	DefaultComputeNoMoreThan = 2 * time.Minute
)

// Config represents a configuration for the continuous query service.
type Config struct {
	// If this flag is set to false, both the brokers and data nodes should ignore any CQ processing.
	Enabled bool `toml:"enabled"`

	// when continuous queries are run we'll automatically recompute previous intervals
	// in case lagged data came in. Set to zero if you never have lagged data. We do
	// it this way because invalidating previously computed intervals would be insanely hard
	// and expensive.
	RecomputePreviousN int `toml:"recompute-previous-n"`

	// The RecomputePreviousN setting provides guidance for how far back to recompute, the RecomputeNoOlderThan
	// setting sets a ceiling on how far back in time it will go. For example, if you have 2 PreviousN
	// and have this set to 10m, then we'd only compute the previous two intervals for any
	// CQs that have a group by time <= 5m. For all others, we'd only recompute the previous window
	RecomputeNoOlderThan toml.Duration `toml:"recompute-no-older-than"`

	// ComputeRunsPerInterval will determine how many times the current and previous N intervals
	// will be computed. The group by time will be divided by this and it will get computed  this many times:
	// group by time seconds / runs per interval
	// This will give partial results for current group by intervals and will determine how long it will
	// be until lagged data is recomputed. For example, if this number is 10 and the group by time is 10m, it
	// will be a minute past the previous 10m bucket of time before lagged data is picked up
	ComputeRunsPerInterval int `toml:"compute-runs-per-interval"`

	// ComputeNoMoreThan paired with the RunsPerInterval will determine the ceiling of how many times smaller
	// group by times will be computed. For example, if you have RunsPerInterval set to 10 and this setting
	// to 1m. Then for a group by time(1m) will actually only get computed once per interval (and once per PreviousN).
	// If you have a group by time(5m) then you'll get five computes per interval. Any group by time window larger
	// than 10m will get computed 10 times for each interval.
	ComputeNoMoreThan toml.Duration `toml:"compute-no-more-than"`
}

// NewConfig returns a new instance of Config with defaults.
func NewConfig() Config {
	return Config{
		Enabled:                true,
		RecomputePreviousN:     DefaultRecomputePreviousN,
		RecomputeNoOlderThan:   toml.Duration(DefaultRecomputeNoOlderThan),
		ComputeRunsPerInterval: DefaultComputeRunsPerInterval,
		ComputeNoMoreThan:      toml.Duration(DefaultComputeNoMoreThan),
	}
}
