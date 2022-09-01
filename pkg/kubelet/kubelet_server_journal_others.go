//go:build !linux && !windows
// +build !linux,!windows

package kubelet

// getLoggingCmd on unsupported operating systems returns the echo command and a warning message (as strings)
func getLoggingCmd(a *journalArgs, boot int) (string, []string) {
	return "echo", []string{"Operating System Not Supported"}
}
