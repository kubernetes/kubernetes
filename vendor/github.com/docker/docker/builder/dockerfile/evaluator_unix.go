// +build !windows

package dockerfile

// platformSupports is a short-term function to give users a quality error
// message if a Dockerfile uses a command not supported on the platform.
func platformSupports(command string) error {
	return nil
}
