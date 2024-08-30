package timeout

import (
	"os"
	"strconv"
	"time"
)

var (
	// defaultTimeout is the timeout for most operations that is not overridden.
	defaultTimeout = 4 * time.Minute

	// defaultTimeoutTestdRetry is the retry loop timeout for testd to respond
	// for a disk to come online in LCOW.
	defaultTimeoutTestdRetry = 5 * time.Second
)

// External variables for hcnshim consumers to use.
var (
	// SystemCreate is the timeout for creating a compute system
	SystemCreate time.Duration = defaultTimeout

	// SystemStart is the timeout for starting a compute system
	SystemStart time.Duration = defaultTimeout

	// SystemPause is the timeout for pausing a compute system
	SystemPause time.Duration = defaultTimeout

	// SystemResume is the timeout for resuming a compute system
	SystemResume time.Duration = defaultTimeout

	// SystemSave is the timeout for saving a compute system
	SystemSave time.Duration = defaultTimeout

	// SyscallWatcher is the timeout before warning of a potential stuck platform syscall.
	SyscallWatcher time.Duration = defaultTimeout

	// Tar2VHD is the timeout for the tar2vhd operation to complete
	Tar2VHD time.Duration = defaultTimeout

	// ExternalCommandToStart is the timeout for external commands to start
	ExternalCommandToStart = defaultTimeout

	// ExternalCommandToComplete is the timeout for external commands to complete.
	// Generally this means copying data from their stdio pipes.
	ExternalCommandToComplete = defaultTimeout

	// TestDRetryLoop is the timeout for testd retry loop when onlining a SCSI disk in LCOW
	TestDRetryLoop = defaultTimeoutTestdRetry
)

func init() {
	SystemCreate = durationFromEnvironment("hcnshim_TIMEOUT_SYSTEMCREATE", SystemCreate)
	SystemStart = durationFromEnvironment("hcnshim_TIMEOUT_SYSTEMSTART", SystemStart)
	SystemPause = durationFromEnvironment("hcnshim_TIMEOUT_SYSTEMPAUSE", SystemPause)
	SystemResume = durationFromEnvironment("hcnshim_TIMEOUT_SYSTEMRESUME", SystemResume)
	SystemSave = durationFromEnvironment("hcnshim_TIMEOUT_SYSTEMSAVE", SystemSave)
	SyscallWatcher = durationFromEnvironment("hcnshim_TIMEOUT_SYSCALLWATCHER", SyscallWatcher)
	Tar2VHD = durationFromEnvironment("hcnshim_TIMEOUT_TAR2VHD", Tar2VHD)
	ExternalCommandToStart = durationFromEnvironment("hcnshim_TIMEOUT_EXTERNALCOMMANDSTART", ExternalCommandToStart)
	ExternalCommandToComplete = durationFromEnvironment("hcnshim_TIMEOUT_EXTERNALCOMMANDCOMPLETE", ExternalCommandToComplete)
	TestDRetryLoop = durationFromEnvironment("hcnshim_TIMEOUT_TESTDRETRYLOOP", TestDRetryLoop)
}

func durationFromEnvironment(env string, defaultValue time.Duration) time.Duration {
	envTimeout := os.Getenv(env)
	if len(envTimeout) > 0 {
		e, err := strconv.Atoi(envTimeout)
		if err == nil && e > 0 {
			return time.Second * time.Duration(e)
		}
	}
	return defaultValue
}
