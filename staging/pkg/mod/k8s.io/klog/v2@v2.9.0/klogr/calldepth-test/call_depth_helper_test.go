package calldepth

import (
	"github.com/go-logr/logr"
)

// Putting these functions into a separate file makes it possible to validate that
// their source code file is *not* logged because of WithCallDepth(1).

func myInfo(l logr.Logger, msg string) {
	logr.WithCallDepth(l, 1).Info(msg)
}

func myInfo2(l logr.Logger, msg string) {
	myInfo(logr.WithCallDepth(l, 1), msg)
}
