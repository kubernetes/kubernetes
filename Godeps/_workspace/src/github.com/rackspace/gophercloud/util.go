package gophercloud

import (
	"errors"
	"strings"
	"time"
)

// WaitFor polls a predicate function, once per second, up to a timeout limit.
// It usually does this to wait for the resource to transition to a certain state.
func WaitFor(timeout int, predicate func() (bool, error)) error {
	start := time.Now().Second()
	for {
		// Force a 1s sleep
		time.Sleep(1 * time.Second)

		// If a timeout is set, and that's been exceeded, shut it down
		if timeout >= 0 && time.Now().Second()-start >= timeout {
			return errors.New("A timeout occurred")
		}

		// Execute the function
		satisfied, err := predicate()
		if err != nil {
			return err
		}
		if satisfied {
			return nil
		}
	}
}

// NormalizeURL ensures that each endpoint URL has a closing `/`, as expected by ServiceClient.
func NormalizeURL(url string) string {
	if !strings.HasSuffix(url, "/") {
		return url + "/"
	}
	return url
}
