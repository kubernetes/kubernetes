package archive

import (
	"time"

	"github.com/pkg/errors"
)

// as at MacOS 10.12 there is apparently no way to set timestamps
// with nanosecond precision. We could fall back to utimes/lutimes
// and lose the precision as a temporary workaround.
func chtimes(path string, atime, mtime time.Time) error {
	return errors.New("OSX missing UtimesNanoAt")
}
