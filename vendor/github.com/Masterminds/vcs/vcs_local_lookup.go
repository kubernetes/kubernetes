package vcs

import (
	"os"
	"runtime"
	"strings"
)

// DetectVcsFromFS detects the type from the local path.
// Is there a better way to do this?
func DetectVcsFromFS(vcsPath string) (Type, error) {

	// There are cases under windows that a path could start with a / and it needs
	// to be stripped. For example, a path such as /C:\foio\bar.
	if runtime.GOOS == "windows" && strings.HasPrefix(vcsPath, "/") {
		vcsPath = strings.TrimPrefix(vcsPath, "/")
	}

	// When the local directory to the package doesn't exist
	// it's not yet downloaded so we can't detect the type
	// locally.
	if _, err := os.Stat(vcsPath); os.IsNotExist(err) {
		return "", ErrCannotDetectVCS
	}

	separator := string(os.PathSeparator)

	// Walk through each of the different VCS types to see if
	// one can be detected. Do this is order of guessed popularity.
	if _, err := os.Stat(vcsPath + separator + ".git"); err == nil {
		return Git, nil
	}
	if _, err := os.Stat(vcsPath + separator + ".svn"); err == nil {
		return Svn, nil
	}
	if _, err := os.Stat(vcsPath + separator + ".hg"); err == nil {
		return Hg, nil
	}
	if _, err := os.Stat(vcsPath + separator + ".bzr"); err == nil {
		return Bzr, nil
	}

	// If one was not already detected than we default to not finding it.
	return "", ErrCannotDetectVCS

}
