//go:build !windows
// +build !windows

package cobra

var preExecHookFn func(*Command)
