//go:build wasm

package internal

import (
	"os"
	"syscall"
)

var PROGRESS_SIGNALS = []os.Signal{syscall.SIGUSR1}
