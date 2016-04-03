// +build windows

package daemon

// checkExecSupport returns an error if the exec driver does not support exec,
// or nil if it is supported.
func checkExecSupport(DriverName string) error {
	return nil
}
