// +build linux

package network

import (
	"errors"
)

var (
	ErrNotValidStrategyType = errors.New("not a valid network strategy type")
)

var strategies = map[string]NetworkStrategy{
	"veth":     &Veth{},
	"loopback": &Loopback{},
}

// NetworkStrategy represents a specific network configuration for
// a container's networking stack
type NetworkStrategy interface {
	Create(*Network, int, *NetworkState) error
	Initialize(*Network, *NetworkState) error
}

// GetStrategy returns the specific network strategy for the
// provided type.  If no strategy is registered for the type an
// ErrNotValidStrategyType is returned.
func GetStrategy(tpe string) (NetworkStrategy, error) {
	s, exists := strategies[tpe]
	if !exists {
		return nil, ErrNotValidStrategyType
	}
	return s, nil
}
