// +build !daemon

package main

const (
	// tests should not assume daemon runs on the same machine as CLI
	isLocalDaemon = false
)
