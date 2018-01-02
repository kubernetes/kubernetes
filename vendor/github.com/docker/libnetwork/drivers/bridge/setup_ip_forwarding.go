package bridge

import (
	"fmt"
	"io/ioutil"

	"github.com/docker/libnetwork/iptables"
	"github.com/sirupsen/logrus"
)

const (
	ipv4ForwardConf     = "/proc/sys/net/ipv4/ip_forward"
	ipv4ForwardConfPerm = 0644
)

func configureIPForwarding(enable bool) error {
	var val byte
	if enable {
		val = '1'
	}
	return ioutil.WriteFile(ipv4ForwardConf, []byte{val, '\n'}, ipv4ForwardConfPerm)
}

func setupIPForwarding(enableIPTables bool) error {
	// Get current IPv4 forward setup
	ipv4ForwardData, err := ioutil.ReadFile(ipv4ForwardConf)
	if err != nil {
		return fmt.Errorf("Cannot read IP forwarding setup: %v", err)
	}

	// Enable IPv4 forwarding only if it is not already enabled
	if ipv4ForwardData[0] != '1' {
		// Enable IPv4 forwarding
		if err := configureIPForwarding(true); err != nil {
			return fmt.Errorf("Enabling IP forwarding failed: %v", err)
		}
		// When enabling ip_forward set the default policy on forward chain to
		// drop only if the daemon option iptables is not set to false.
		if !enableIPTables {
			return nil
		}
		if err := iptables.SetDefaultPolicy(iptables.Filter, "FORWARD", iptables.Drop); err != nil {
			if err := configureIPForwarding(false); err != nil {
				logrus.Errorf("Disabling IP forwarding failed, %v", err)
			}
			return err
		}
		iptables.OnReloaded(func() {
			logrus.Debug("Setting the default DROP policy on firewall reload")
			if err := iptables.SetDefaultPolicy(iptables.Filter, "FORWARD", iptables.Drop); err != nil {
				logrus.Warnf("Settig the default DROP policy on firewall reload failed, %v", err)
			}
		})
	}
	return nil
}
