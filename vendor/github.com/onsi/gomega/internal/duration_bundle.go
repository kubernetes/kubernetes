package internal

import (
	"fmt"
	"os"
	"reflect"
	"time"
)

type DurationBundle struct {
	EventuallyTimeout                       time.Duration
	EventuallyPollingInterval               time.Duration
	ConsistentlyDuration                    time.Duration
	ConsistentlyPollingInterval             time.Duration
	EnforceDefaultTimeoutsWhenUsingContexts bool
}

const (
	EventuallyTimeoutEnvVarName         = "GOMEGA_DEFAULT_EVENTUALLY_TIMEOUT"
	EventuallyPollingIntervalEnvVarName = "GOMEGA_DEFAULT_EVENTUALLY_POLLING_INTERVAL"

	ConsistentlyDurationEnvVarName        = "GOMEGA_DEFAULT_CONSISTENTLY_DURATION"
	ConsistentlyPollingIntervalEnvVarName = "GOMEGA_DEFAULT_CONSISTENTLY_POLLING_INTERVAL"

	EnforceDefaultTimeoutsWhenUsingContextsEnvVarName = "GOMEGA_ENFORCE_DEFAULT_TIMEOUTS_WHEN_USING_CONTEXTS"
)

func FetchDefaultDurationBundle() DurationBundle {
	_, EnforceDefaultTimeoutsWhenUsingContexts := os.LookupEnv(EnforceDefaultTimeoutsWhenUsingContextsEnvVarName)
	return DurationBundle{
		EventuallyTimeout:         durationFromEnv(EventuallyTimeoutEnvVarName, time.Second),
		EventuallyPollingInterval: durationFromEnv(EventuallyPollingIntervalEnvVarName, 10*time.Millisecond),

		ConsistentlyDuration:                    durationFromEnv(ConsistentlyDurationEnvVarName, 100*time.Millisecond),
		ConsistentlyPollingInterval:             durationFromEnv(ConsistentlyPollingIntervalEnvVarName, 10*time.Millisecond),
		EnforceDefaultTimeoutsWhenUsingContexts: EnforceDefaultTimeoutsWhenUsingContexts,
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

func toDuration(input interface{}) (time.Duration, error) {
	duration, ok := input.(time.Duration)
	if ok {
		return duration, nil
	}

	value := reflect.ValueOf(input)
	kind := reflect.TypeOf(input).Kind()

	if reflect.Int <= kind && kind <= reflect.Int64 {
		return time.Duration(value.Int()) * time.Second, nil
	} else if reflect.Uint <= kind && kind <= reflect.Uint64 {
		return time.Duration(value.Uint()) * time.Second, nil
	} else if reflect.Float32 <= kind && kind <= reflect.Float64 {
		return time.Duration(value.Float() * float64(time.Second)), nil
	} else if reflect.String == kind {
		duration, err := time.ParseDuration(value.String())
		if err != nil {
			return 0, fmt.Errorf("%#v is not a valid parsable duration string: %w", input, err)
		}
		return duration, nil
	}

	return 0, fmt.Errorf("%#v is not a valid interval. Must be a time.Duration, a parsable duration string, or a number.", input)
}
