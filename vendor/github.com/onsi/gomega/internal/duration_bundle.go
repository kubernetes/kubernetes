package internal

import (
	"fmt"
	"os"
	"reflect"
	"time"
)

type DurationBundle struct {
	EventuallyTimeout           time.Duration
	EventuallyPollingInterval   time.Duration
	ConsistentlyDuration        time.Duration
	ConsistentlyPollingInterval time.Duration
}

const (
	EventuallyTimeoutEnvVarName         = "GOMEGA_DEFAULT_EVENTUALLY_TIMEOUT"
	EventuallyPollingIntervalEnvVarName = "GOMEGA_DEFAULT_EVENTUALLY_POLLING_INTERVAL"

	ConsistentlyDurationEnvVarName        = "GOMEGA_DEFAULT_CONSISTENTLY_DURATION"
	ConsistentlyPollingIntervalEnvVarName = "GOMEGA_DEFAULT_CONSISTENTLY_POLLING_INTERVAL"
)

func FetchDefaultDurationBundle() DurationBundle {
	return DurationBundle{
		EventuallyTimeout:         durationFromEnv(EventuallyTimeoutEnvVarName, time.Second),
		EventuallyPollingInterval: durationFromEnv(EventuallyPollingIntervalEnvVarName, 10*time.Millisecond),

		ConsistentlyDuration:        durationFromEnv(ConsistentlyDurationEnvVarName, 100*time.Millisecond),
		ConsistentlyPollingInterval: durationFromEnv(ConsistentlyPollingIntervalEnvVarName, 10*time.Millisecond),
	}
}

func durationFromEnv(key string, defaultDuration time.Duration) time.Duration {
	value := os.Getenv(key)
	if value == "" {
		return defaultDuration
	}
	duration, err := time.ParseDuration(value)
	if err != nil {
		panic(fmt.Sprintf("Expected a duration when using %s!  Parse error %v", key, err))
	}
	return duration
}

func toDuration(input interface{}) time.Duration {
	duration, ok := input.(time.Duration)
	if ok {
		return duration
	}

	value := reflect.ValueOf(input)
	kind := reflect.TypeOf(input).Kind()

	if reflect.Int <= kind && kind <= reflect.Int64 {
		return time.Duration(value.Int()) * time.Second
	} else if reflect.Uint <= kind && kind <= reflect.Uint64 {
		return time.Duration(value.Uint()) * time.Second
	} else if reflect.Float32 <= kind && kind <= reflect.Float64 {
		return time.Duration(value.Float() * float64(time.Second))
	} else if reflect.String == kind {
		duration, err := time.ParseDuration(value.String())
		if err != nil {
			panic(fmt.Sprintf("%#v is not a valid parsable duration string.", input))
		}
		return duration
	}

	panic(fmt.Sprintf("%v is not a valid interval.  Must be time.Duration, parsable duration string or a number.", input))
}
