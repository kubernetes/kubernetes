package testutils

import "flag"

var runningInContainer = flag.Bool("incontainer", false, "Indicates if the test is running in a container")

// IsRunningInContainer returns whether the test is running inside a container.
func IsRunningInContainer() bool {
	return (*runningInContainer)
}
