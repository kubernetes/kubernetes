package xattr

import (
	"fmt"
	"runtime"
)

var ErrNotSupportedPlatform = fmt.Errorf("platform and architecture is not supported %s %s", runtime.GOOS, runtime.GOARCH)
