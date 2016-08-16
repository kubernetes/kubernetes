// +build static_build

package dbus

func lookupHomeDir() string {
	return guessHomeDir()
}
