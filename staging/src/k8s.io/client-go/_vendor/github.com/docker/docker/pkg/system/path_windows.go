// +build windows

package system

// DefaultPathEnv is deliberately empty on Windows as the default path will be set by
// the container. Docker has no context of what the default path should be.
const DefaultPathEnv = ""
