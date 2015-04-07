// +build linux

package network

import (
	"fmt"
)

// Loopback is a network strategy that provides a basic loopback device
type Loopback struct {
}

func (l *Loopback) Create(n *Network, nspid int, networkState *NetworkState) error {
	return nil
}

func (l *Loopback) Initialize(config *Network, networkState *NetworkState) error {
	// Do not set the MTU on the loopback interface - use the default.
	if err := InterfaceUp("lo"); err != nil {
		return fmt.Errorf("lo up %s", err)
	}
	return nil
}
