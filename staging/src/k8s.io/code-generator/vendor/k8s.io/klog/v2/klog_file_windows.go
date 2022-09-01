//go:build windows
// +build windows

package klog

import (
	"os"
	"strings"
)

func getUserName() string {
	userNameOnce.Do(func() {
		// On Windows, the Go 'user' package requires netapi32.dll.
		// This affects Windows Nano Server:
		//   https://github.com/golang/go/issues/21867
		// Fallback to using environment variables.
		u := os.Getenv("USERNAME")
		if len(u) == 0 {
			return
		}
		// Sanitize the USERNAME since it may contain filepath separators.
		u = strings.Replace(u, `\`, "_", -1)

		// user.Current().Username normally produces something like 'USERDOMAIN\USERNAME'
		d := os.Getenv("USERDOMAIN")
		if len(d) != 0 {
			userName = d + "_" + u
		} else {
			userName = u
		}
	})

	return userName
}
