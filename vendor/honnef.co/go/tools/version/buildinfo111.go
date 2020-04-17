// +build !go1.12

package version

func printBuildInfo()                  {}
func buildInfoVersion() (string, bool) { return "", false }
