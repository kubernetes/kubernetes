/*
Copyright 2016 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package cloudstack

import (
	"errors"
	"fmt"
	"io/ioutil"
	"net"
	"net/http"

	"github.com/d2g/dhcp4"
	"github.com/golang/glog"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

type metadata struct {
	dhcpServer string
	zone       string
}

type metadataType string

const (
	metadataTypeExternalIP   metadataType = "public-ipv4"
	metadataTypeInternalIP   metadataType = "local-ipv4"
	metadataTypeInstanceID   metadataType = "instance-id"
	metadataTypeInstanceType metadataType = "service-offering"
	metadataTypeZone         metadataType = "availability-zone"
)

// NodeAddresses returns the addresses of the specified instance.
func (m *metadata) NodeAddresses(name types.NodeName) ([]v1.NodeAddress, error) {
	externalIP, err := m.get(metadataTypeExternalIP)
	if err != nil {
		return nil, fmt.Errorf("could not get external IP: %v", err)
	}

	internalIP, err := m.get(metadataTypeInternalIP)
	if err != nil {
		return nil, fmt.Errorf("could not get internal IP: %v", err)
	}

	return []v1.NodeAddress{
		{Type: v1.NodeExternalIP, Address: externalIP},
		{Type: v1.NodeInternalIP, Address: internalIP},
	}, nil
}

// NodeAddressesByProviderID returns the addresses of the specified instance.
func (m *metadata) NodeAddressesByProviderID(providerID string) ([]v1.NodeAddress, error) {
	return nil, errors.New("NodeAddressesByProviderID not implemented")
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (m *metadata) ExternalID(name types.NodeName) (string, error) {
	return m.InstanceID(name)
}

// InstanceID returns the cloud provider ID of the specified instance.
func (m *metadata) InstanceID(name types.NodeName) (string, error) {
	instanceID, err := m.get(metadataTypeInstanceID)
	if err != nil {
		return "", fmt.Errorf("could not get instance ID: %v", err)
	}

	zone, err := m.get(metadataTypeZone)
	if err != nil {
		return "", fmt.Errorf("could not get zone: %v", err)
	}

	return "/" + zone + "/" + instanceID, nil
}

// InstanceType returns the type of the specified instance.
func (m *metadata) InstanceType(name types.NodeName) (string, error) {
	instanceType, err := m.get(metadataTypeInstanceType)
	if err == nil {
		return "", fmt.Errorf("could not get instance type: %v", err)
	}

	return instanceType, nil
}

// InstanceTypeByProviderID returns the type of the specified instance.
func (m *metadata) InstanceTypeByProviderID(providerID string) (string, error) {
	return "", errors.New("InstanceTypeByProviderID not implemented")
}

// AddSSHKeyToAllInstances is currently not implemented.
func (m *metadata) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("AddSSHKeyToAllInstances not implemented")
}

// CurrentNodeName returns the name of the node we are currently running on.
func (m *metadata) CurrentNodeName(hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// InstanceExistsByProviderID returns if the instance still exists.
func (m *metadata) InstanceExistsByProviderID(providerID string) (bool, error) {
	return false, errors.New("InstanceExistsByProviderID not implemented")
}

// GetZone returns the Zone containing the region that the program is running in.
func (m *metadata) GetZone() (cloudprovider.Zone, error) {
	zone := cloudprovider.Zone{}

	if m.zone == "" {
		zoneName, err := m.get(metadataTypeZone)
		if err != nil {
			return zone, fmt.Errorf("could not get zone: %v", err)
		}

		m.zone = zoneName
	}

	glog.V(2).Infof("Current zone is %v", zone)
	zone.FailureDomain = m.zone
	zone.Region = m.zone

	return zone, nil
}

// GetZoneByProviderID returns the Zone, found by using the provider ID.
func (m *metadata) GetZoneByProviderID(providerID string) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{}, errors.New("GetZoneByProviderID not implemented")
}

// GetZoneByNodeName returns the Zone, found by using the node name.
func (m *metadata) GetZoneByNodeName(nodeName types.NodeName) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{}, errors.New("GetZoneByNodeName not implemented")
}

func (m *metadata) get(mdType metadataType) (string, error) {
	url := fmt.Sprintf("http://%s/latest/meta-data/%s", m.dhcpServer, mdType)

	resp, err := http.Get(url)
	if err != nil {
		return "", fmt.Errorf("error reading metadata: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("unexpected HTTP status: %d", resp.StatusCode)
	}

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("error reading response body: %d", resp.StatusCode)
	}

	return string(data), nil
}

func findDHCPServer() (string, error) {
	nics, err := net.Interfaces()
	if err != nil {
		return "", fmt.Errorf("could not get interfaces: %v", err)
	}

	for _, nic := range nics {
		if nic.Flags&net.FlagUp == 1 && nic.Flags&net.FlagLoopback == 0 && nic.Flags&net.FlagPointToPoint == 0 {
			addrs, err := nic.Addrs()
			if err != nil {
				return "", fmt.Errorf("error reading IP addresses from interface %v: %v", nic.Name, err)
			}

			if addrs != nil {
				client, err := newDHCPClient(&nic)
				if err != nil {
					return "", fmt.Errorf("error creating new DHCP client: %v", err)
				}

				discoverPacket, err := client.SendDiscoverPacket()
				if err != nil {
					return "", fmt.Errorf("error sending DHCP discover package: %v", err)
				}

				offerPacket, err := client.GetOffer(&discoverPacket)
				if err != nil {
					return "", fmt.Errorf("error recieving DHCP offer package: %v", err)
				}

				offerPacketOptions := offerPacket.ParseOptions()

				if ipaddr, ok := offerPacketOptions[dhcp4.OptionServerIdentifier]; ok {
					return net.IP(ipaddr).String(), nil
				}
			}
		}
	}

	return "", errors.New("no server found")
}
