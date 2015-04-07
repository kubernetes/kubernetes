// +build linux

package network

import (
	"fmt"

	"github.com/docker/libcontainer/netlink"
	"github.com/docker/libcontainer/utils"
)

// Veth is a network strategy that uses a bridge and creates
// a veth pair, one that stays outside on the host and the other
// is placed inside the container's namespace
type Veth struct {
}

const defaultDevice = "eth0"

func (v *Veth) Create(n *Network, nspid int, networkState *NetworkState) error {
	var (
		bridge     = n.Bridge
		prefix     = n.VethPrefix
		txQueueLen = n.TxQueueLen
	)
	if bridge == "" {
		return fmt.Errorf("bridge is not specified")
	}
	if prefix == "" {
		return fmt.Errorf("veth prefix is not specified")
	}
	name1, name2, err := createVethPair(prefix, txQueueLen)
	if err != nil {
		return err
	}
	if err := SetInterfaceMaster(name1, bridge); err != nil {
		return err
	}
	if err := SetMtu(name1, n.Mtu); err != nil {
		return err
	}
	if err := InterfaceUp(name1); err != nil {
		return err
	}
	if err := SetInterfaceInNamespacePid(name2, nspid); err != nil {
		return err
	}
	networkState.VethHost = name1
	networkState.VethChild = name2

	return nil
}

func (v *Veth) Initialize(config *Network, networkState *NetworkState) error {
	var vethChild = networkState.VethChild
	if vethChild == "" {
		return fmt.Errorf("vethChild is not specified")
	}
	if err := InterfaceDown(vethChild); err != nil {
		return fmt.Errorf("interface down %s %s", vethChild, err)
	}
	if err := ChangeInterfaceName(vethChild, defaultDevice); err != nil {
		return fmt.Errorf("change %s to %s %s", vethChild, defaultDevice, err)
	}
	if config.MacAddress != "" {
		if err := SetInterfaceMac(defaultDevice, config.MacAddress); err != nil {
			return fmt.Errorf("set %s mac %s", defaultDevice, err)
		}
	}
	if err := SetInterfaceIp(defaultDevice, config.Address); err != nil {
		return fmt.Errorf("set %s ip %s", defaultDevice, err)
	}
	if config.IPv6Address != "" {
		if err := SetInterfaceIp(defaultDevice, config.IPv6Address); err != nil {
			return fmt.Errorf("set %s ipv6 %s", defaultDevice, err)
		}
	}

	if err := SetMtu(defaultDevice, config.Mtu); err != nil {
		return fmt.Errorf("set %s mtu to %d %s", defaultDevice, config.Mtu, err)
	}
	if err := InterfaceUp(defaultDevice); err != nil {
		return fmt.Errorf("%s up %s", defaultDevice, err)
	}
	if config.Gateway != "" {
		if err := SetDefaultGateway(config.Gateway, defaultDevice); err != nil {
			return fmt.Errorf("set gateway to %s on device %s failed with %s", config.Gateway, defaultDevice, err)
		}
	}
	if config.IPv6Gateway != "" {
		if err := SetDefaultGateway(config.IPv6Gateway, defaultDevice); err != nil {
			return fmt.Errorf("set gateway for ipv6 to %s on device %s failed with %s", config.IPv6Gateway, defaultDevice, err)
		}
	}
	return nil
}

// createVethPair will automatically generage two random names for
// the veth pair and ensure that they have been created
func createVethPair(prefix string, txQueueLen int) (name1 string, name2 string, err error) {
	for i := 0; i < 10; i++ {
		if name1, err = utils.GenerateRandomName(prefix, 7); err != nil {
			return
		}

		if name2, err = utils.GenerateRandomName(prefix, 7); err != nil {
			return
		}

		if err = CreateVethPair(name1, name2, txQueueLen); err != nil {
			if err == netlink.ErrInterfaceExists {
				continue
			}

			return
		}

		break
	}

	return
}
