//go:build !linux

package configs

type Mount struct{}

func (m *Mount) IsBind() bool {
	return false
}
