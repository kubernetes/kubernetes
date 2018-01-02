package sys

// SetOOMScore sets the oom score for the process
//
// Not implemented on Windows
func SetOOMScore(pid, score int) error {
	return nil
}
