// +build linux freebsd

package container

// setFromExitStatus is a platform specific helper function to set the state
// based on the ExitStatus structure.
func (s *State) setFromExitStatus(exitStatus *ExitStatus) {
	s.ExitCodeValue = exitStatus.ExitCode
	s.OOMKilled = exitStatus.OOMKilled
}
