package osutil

import "os"

// Region provides a means of querying the OS_REGION_NAME environment variable.
// At present, you may also use os.Getenv("OS_REGION_NAME") as well.
func Region() string {
	return os.Getenv("OS_REGION_NAME")
}
