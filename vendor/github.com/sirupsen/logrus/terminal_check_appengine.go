//go:build appengine

package logrus

func checkIfTerminal(_ any) bool {
	return true
}
