// +build go1.7

package virtualnetwork

// Copyright 2017 Microsoft Corporation
//
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
//
//        http://www.apache.org/licenses/LICENSE-2.0
//
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

import (
	"encoding/xml"

	"github.com/Azure/azure-sdk-for-go/management"
)

const xmlNamespace = "http://schemas.microsoft.com/ServiceHosting/2011/07/NetworkConfiguration"
const xmlNamespaceXsd = "http://www.w3.org/2001/XMLSchema"
const xmlNamespaceXsi = "http://www.w3.org/2001/XMLSchema-instance"

// VirtualNetworkClient is used to perform operations on Virtual Networks.
type VirtualNetworkClient struct {
	client management.Client
}

// NetworkConfiguration represents the network configuration for an entire Azure
// subscription.
type NetworkConfiguration struct {
	XMLName         xml.Name                    `xml:"NetworkConfiguration"`
	XMLNamespaceXsd string                      `xml:"xmlns:xsd,attr"`
	XMLNamespaceXsi string                      `xml:"xmlns:xsi,attr"`
	XMLNs           string                      `xml:"xmlns,attr"`
	Configuration   VirtualNetworkConfiguration `xml:"VirtualNetworkConfiguration"`

	// TODO: Nicer builder methods for these that abstract away the
	// underlying structure.
}

// NewNetworkConfiguration creates a new empty NetworkConfiguration structure
// for further configuration. The XML namespaces are already set correctly.
func (client *VirtualNetworkClient) NewNetworkConfiguration() NetworkConfiguration {
	networkConfiguration := NetworkConfiguration{}
	networkConfiguration.setXMLNamespaces()
	return networkConfiguration
}

// setXMLNamespaces ensure that all of the required namespaces are set. It
// should be called prior to marshalling the structure to XML for use with the
// Azure REST endpoint. It is used internally prior to submitting requests, but
// since it is idempotent there is no harm in repeat calls.
func (n *NetworkConfiguration) setXMLNamespaces() {
	n.XMLNamespaceXsd = xmlNamespaceXsd
	n.XMLNamespaceXsi = xmlNamespaceXsi
	n.XMLNs = xmlNamespace
}

type VirtualNetworkConfiguration struct {
	DNS                 DNS                  `xml:"Dns,omitempty"`
	LocalNetworkSites   []LocalNetworkSite   `xml:"LocalNetworkSites>LocalNetworkSite"`
	VirtualNetworkSites []VirtualNetworkSite `xml:"VirtualNetworkSites>VirtualNetworkSite"`
}

type DNS struct {
	DNSServers []DNSServer `xml:"DnsServers>DnsServer,omitempty"`
}

type DNSServer struct {
	XMLName   xml.Name `xml:"DnsServer"`
	Name      string   `xml:"name,attr"`
	IPAddress string   `xml:"IPAddress,attr"`
}

type DNSServerRef struct {
	Name string `xml:"name,attr"`
}

type VirtualNetworkSite struct {
	Name          string         `xml:"name,attr"`
	Location      string         `xml:"Location,attr"`
	AddressSpace  AddressSpace   `xml:"AddressSpace"`
	Subnets       []Subnet       `xml:"Subnets>Subnet"`
	DNSServersRef []DNSServerRef `xml:"DnsServersRef>DnsServerRef,omitempty"`
}

type LocalNetworkSite struct {
	Name              string `xml:"name,attr"`
	VPNGatewayAddress string
	AddressSpace      AddressSpace
}

type AddressSpace struct {
	AddressPrefix []string
}

type Subnet struct {
	Name          string `xml:"name,attr"`
	AddressPrefix string
}
