package debugger

import "syscall"

// compareSignal is the signal to trigger cache compare. For non-windows
// environment it's SIGUSR2.
var compareSignal = syscall.SIGUSR2
