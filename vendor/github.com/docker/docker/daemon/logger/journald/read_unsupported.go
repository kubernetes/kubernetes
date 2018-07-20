// +build !linux !cgo static_build !journald

package journald

func (s *journald) Close() error {
	return nil
}
