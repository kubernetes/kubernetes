/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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

package ovirt_cloud

import (
	"encoding/xml"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"

	"github.com/scalingdata/gcfg"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "ovirt"

type OVirtInstance struct {
	UUID      string
	Name      string
	IPAddress string
}

type OVirtInstanceMap map[string]OVirtInstance

type OVirtCloud struct {
	VmsRequest   *url.URL
	HostsRequest *url.URL
}

type OVirtApiConfig struct {
	Connection struct {
		ApiEntry string `gcfg:"uri"`
		Username string `gcfg:"username"`
		Password string `gcfg:"password"`
	}
	Filters struct {
		VmsQuery string `gcfg:"vms"`
	}
}

type XmlVmAddress struct {
	Address string `xml:"address,attr"`
}

type XmlVmInfo struct {
	UUID      string         `xml:"id,attr"`
	Name      string         `xml:"name"`
	Hostname  string         `xml:"guest_info>fqdn"`
	Addresses []XmlVmAddress `xml:"guest_info>ips>ip"`
	State     string         `xml:"status>state"`
}

type XmlVmsList struct {
	XMLName xml.Name    `xml:"vms"`
	Vm      []XmlVmInfo `xml:"vm"`
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName,
		func(config io.Reader) (cloudprovider.Interface, error) {
			return newOVirtCloud(config)
		})
}

func newOVirtCloud(config io.Reader) (*OVirtCloud, error) {
	if config == nil {
		return nil, fmt.Errorf("missing configuration file for ovirt cloud provider")
	}

	oVirtConfig := OVirtApiConfig{}

	/* defaults */
	oVirtConfig.Connection.Username = "admin@internal"

	if err := gcfg.ReadInto(&oVirtConfig, config); err != nil {
		return nil, err
	}

	if oVirtConfig.Connection.ApiEntry == "" {
		return nil, fmt.Errorf("missing ovirt uri in cloud provider configuration")
	}

	request, err := url.Parse(oVirtConfig.Connection.ApiEntry)
	if err != nil {
		return nil, err
	}

	request.Path = path.Join(request.Path, "vms")
	request.User = url.UserPassword(oVirtConfig.Connection.Username, oVirtConfig.Connection.Password)
	request.RawQuery = url.Values{"search": {oVirtConfig.Filters.VmsQuery}}.Encode()

	return &OVirtCloud{VmsRequest: request}, nil
}

func (aws *OVirtCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (v *OVirtCloud) ProviderName() string {
	return ProviderName
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for oVirt cloud
func (v *OVirtCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

// Instances returns an implementation of Instances for oVirt cloud
func (v *OVirtCloud) Instances() (cloudprovider.Instances, bool) {
	return v, true
}

// Zones returns an implementation of Zones for oVirt cloud
func (v *OVirtCloud) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

// Routes returns an implementation of Routes for oVirt cloud
func (v *OVirtCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// NodeAddresses returns the NodeAddresses of a particular machine instance
func (v *OVirtCloud) NodeAddresses(name string) ([]api.NodeAddress, error) {
	instance, err := v.fetchInstance(name)
	if err != nil {
		return nil, err
	}

	var address net.IP

	if instance.IPAddress != "" {
		address = net.ParseIP(instance.IPAddress)
		if address == nil {
			return nil, fmt.Errorf("couldn't parse address: %s", instance.IPAddress)
		}
	} else {
		resolved, err := net.LookupIP(name)
		if err != nil || len(resolved) < 1 {
			return nil, fmt.Errorf("couldn't lookup address: %s", name)
		}
		address = resolved[0]
	}

	return []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: address.String()}}, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (v *OVirtCloud) ExternalID(name string) (string, error) {
	instance, err := v.fetchInstance(name)
	if err != nil {
		return "", err
	}
	return instance.UUID, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (v *OVirtCloud) InstanceID(name string) (string, error) {
	instance, err := v.fetchInstance(name)
	if err != nil {
		return "", err
	}
	// TODO: define a way to identify the provider instance to complete
	// the format <provider_instance_id>/<instance_id>.
	return "/" + instance.UUID, err
}

func getInstancesFromXml(body io.Reader) (OVirtInstanceMap, error) {
	if body == nil {
		return nil, fmt.Errorf("ovirt rest-api response body is missing")
	}

	content, err := ioutil.ReadAll(body)
	if err != nil {
		return nil, err
	}

	vmlist := XmlVmsList{}

	if err := xml.Unmarshal(content, &vmlist); err != nil {
		return nil, err
	}

	instances := make(OVirtInstanceMap)

	for _, vm := range vmlist.Vm {
		// Always return only vms that are up and running
		if vm.Hostname != "" && strings.ToLower(vm.State) == "up" {
			address := ""
			if len(vm.Addresses) > 0 {
				address = vm.Addresses[0].Address
			}

			instances[vm.Hostname] = OVirtInstance{
				UUID:      vm.UUID,
				Name:      vm.Name,
				IPAddress: address,
			}
		}
	}

	return instances, nil
}

func (v *OVirtCloud) fetchAllInstances() (OVirtInstanceMap, error) {
	response, err := http.Get(v.VmsRequest.String())
	if err != nil {
		return nil, err
	}

	defer response.Body.Close()

	return getInstancesFromXml(response.Body)
}

func (v *OVirtCloud) fetchInstance(name string) (*OVirtInstance, error) {
	allInstances, err := v.fetchAllInstances()
	if err != nil {
		return nil, err
	}

	instance, found := allInstances[name]
	if !found {
		return nil, fmt.Errorf("cannot find instance: %s", name)
	}

	return &instance, nil
}

func (m *OVirtInstanceMap) ListSortedNames() []string {
	var names []string

	for k := range *m {
		names = append(names, k)
	}

	sort.Strings(names)

	return names
}

// List enumerates the set of minions instances known by the cloud provider
func (v *OVirtCloud) List(filter string) ([]string, error) {
	instances, err := v.fetchAllInstances()
	if err != nil {
		return nil, err
	}
	return instances.ListSortedNames(), nil
}

// Implementation of Instances.CurrentNodeName
func (v *OVirtCloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

func (v *OVirtCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}
