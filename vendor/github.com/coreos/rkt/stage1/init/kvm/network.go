// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package kvm

import (
	"crypto/rand"
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"path/filepath"

	"github.com/containernetworking/cni/pkg/types"
	"github.com/coreos/go-systemd/unit"
	"github.com/coreos/rkt/networking"
	"github.com/hashicorp/errwrap"
)

// GetNetworkDescriptions converts activeNets to netDescribers
func GetNetworkDescriptions(n *networking.Networking) []NetDescriber {
	var nds []NetDescriber
	for _, an := range n.GetActiveNetworks() {
		nds = append(nds, an)
	}
	return nds
}

// NetDescriber is the interface that describes a network configuration
type NetDescriber interface {
	GuestIP() net.IP
	Mask() net.IP
	IfName() string
	IPMasq() bool
	Name() string
	Gateway() net.IP
	Routes() []types.Route
}

// GetKVMNetArgs returns additional arguments that need to be passed
// to lkvm tool to configure networks properly.
// Logic is based on Network configuration extracted from Networking struct
// and essentially from activeNets that expose netDescriber behavior
func GetKVMNetArgs(nds []NetDescriber) ([]string, error) {

	var lkvmArgs []string

	for _, nd := range nds {
		lkvmArgs = append(lkvmArgs, "--network")
		lkvmArg := fmt.Sprintf("mode=tap,tapif=%s,host_ip=%s,guest_ip=%s", nd.IfName(), nd.Gateway(), nd.GuestIP())
		lkvmArgs = append(lkvmArgs, lkvmArg)
	}

	return lkvmArgs, nil
}

// generateMacAddress returns net.HardwareAddr filled with fixed 3 byte prefix
// complemented by 3 random bytes.
func generateMacAddress() (net.HardwareAddr, error) {
	mac := []byte{
		2,          // locally administered unicast
		0x65, 0x02, // OUI (randomly chosen by jell)
		0, 0, 0, // bytes to randomly overwrite
	}

	_, err := rand.Read(mac[3:6])
	if err != nil {
		return nil, errwrap.Wrap(errors.New("cannot generate random mac address"), err)
	}

	return mac, nil
}

func setMacCommand(ifName, mac string) string {
	return fmt.Sprintf("/bin/ip link set dev %s address %s", ifName, mac)
}

func addAddressCommand(address, ifName string) string {
	return fmt.Sprintf("/bin/ip address add %s dev %s", address, ifName)
}

func addRouteCommand(destination, router string) string {
	return fmt.Sprintf("/bin/ip route add %s via %s", destination, router)
}

func downInterfaceCommand(ifName string) string {
	return fmt.Sprintf("/bin/ip link set dev %s down", ifName)
}

func upInterfaceCommand(ifName string) string {
	return fmt.Sprintf("/bin/ip link set dev %s up", ifName)
}

func GenerateNetworkInterfaceUnits(unitsPath string, netDescriptions []NetDescriber) error {
	for i, netDescription := range netDescriptions {
		ifName := fmt.Sprintf(networking.IfNamePattern, i)
		netAddress := net.IPNet{
			IP:   netDescription.GuestIP(),
			Mask: net.IPMask(netDescription.Mask()),
		}

		address := netAddress.String()

		mac, err := generateMacAddress()
		if err != nil {
			return err
		}

		opts := []*unit.UnitOption{
			unit.NewUnitOption("Unit", "Description", fmt.Sprintf("Network configuration for device: %v", ifName)),
			unit.NewUnitOption("Unit", "DefaultDependencies", "false"),
			unit.NewUnitOption("Service", "Type", "oneshot"),
			unit.NewUnitOption("Service", "RemainAfterExit", "true"),
			unit.NewUnitOption("Service", "ExecStartPre", downInterfaceCommand(ifName)),
			unit.NewUnitOption("Service", "ExecStartPre", setMacCommand(ifName, mac.String())),
			unit.NewUnitOption("Service", "ExecStartPre", upInterfaceCommand(ifName)),
			unit.NewUnitOption("Service", "ExecStart", addAddressCommand(address, ifName)),
			unit.NewUnitOption("Install", "RequiredBy", "default.target"),
		}

		for _, route := range netDescription.Routes() {
			gw := route.GW
			if gw == nil {
				gw = netDescription.Gateway()
			}

			opts = append(
				opts,
				unit.NewUnitOption(
					"Service",
					"ExecStartPost",
					addRouteCommand(route.Dst.String(), gw.String()),
				),
			)
		}

		unitName := fmt.Sprintf("interface-%s", ifName) + ".service"
		unitBytes, err := ioutil.ReadAll(unit.Serialize(opts))
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("failed to serialize network unit file to bytes %q", unitName), err)
		}

		err = ioutil.WriteFile(filepath.Join(unitsPath, unitName), unitBytes, 0644)
		if err != nil {
			return errwrap.Wrap(fmt.Errorf("failed to create network unit file %q", unitName), err)
		}

		diag.Printf("network unit created: %q in %q (iface=%q, addr=%q)", unitName, unitsPath, ifName, address)
	}
	return nil
}
