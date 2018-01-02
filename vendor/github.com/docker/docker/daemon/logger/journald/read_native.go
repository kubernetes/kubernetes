// +build linux,cgo,!static_build,journald,!journald_compat

package journald

// #cgo pkg-config: libsystemd
import "C"
