// +build godirwalk_debug

package godirwalk

import (
	"fmt"
	"os"
)

// debug formats and prints arguments to stderr for development builds
func debug(f string, a ...interface{}) {
	// fmt.Fprintf(os.Stderr, f, a...)
	os.Stderr.Write([]byte("godirwalk: " + fmt.Sprintf(f, a...)))
}
