package dbus

import (
	"os"
	"sync"
)

var (
	homeDir     string
	homeDirLock sync.Mutex
)

func getHomeDir() string {
	homeDirLock.Lock()
	defer homeDirLock.Unlock()

	if homeDir != "" {
		return homeDir
	}

	homeDir = os.Getenv("HOME")
	if homeDir != "" {
		return homeDir
	}

	homeDir = lookupHomeDir()
	return homeDir
}
