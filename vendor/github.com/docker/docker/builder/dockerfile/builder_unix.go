// +build !windows

package dockerfile

func defaultShellForPlatform(platform string) []string {
	return []string{"/bin/sh", "-c"}
}
