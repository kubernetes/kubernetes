//go:build !windows
// +build !windows

package ssocreds

import "os"

func getHomeDirectory() string {
	return os.Getenv("HOME")
}
