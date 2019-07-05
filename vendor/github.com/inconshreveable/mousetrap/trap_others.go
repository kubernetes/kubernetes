// +build !windows

package mousetrap

// StartedByExplorer returns true if the program was invoked by the user
// double-clicking on the executable from explorer.exe
//
// It is conservative and returns false if any of the internal calls fail.
// It does not guarantee that the program was run from a terminal. It only can tell you
// whether it was launched from explorer.exe
//
// On non-Windows platforms, it always returns false.
func StartedByExplorer() bool {
	return false
}
