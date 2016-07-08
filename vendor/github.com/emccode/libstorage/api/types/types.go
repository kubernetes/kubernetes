package types

import (
	"os"
	"strconv"
)

var (
	// Debug is a flag that indicates whether or not the environment variable
	// `LIBSTORAGE_DEBUG` is set to a boolean true value.
	Debug, _ = strconv.ParseBool(os.Getenv("LIBSTORAGE_DEBUG"))
)

func init() {
	initPaths()
}
