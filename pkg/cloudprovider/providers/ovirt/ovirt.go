/*
Copyright 2014 The Kubernetes Authors.

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

package ovirt

import (
	"context"
	"encoding/xml"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"path"
	"sort"
	"strings"

	"gopkg.in/gcfg.v1"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	cloudprovider "k8s.io/cloud-provider"
)

// ProviderName is the name of this cloud provider.
const ProviderName = "ovirt"

// Instance specifies UUID, name and IP address of the instance.
type Instance struct {
	UUID      string
	Name      string
	IPAddress string
}

// InstanceMap provides the map of Ovirt instances.
type InstanceMap map[string]Instance

var _ cloudprovider.Interface = (*Cloud)(nil)
var _ cloudprovider.Instances = (*Cloud)(nil)

// Cloud is an implementation of the cloud provider interface for Ovirt.
type Cloud struct {
	VmsRequest   *url.URL
	HostsRequest *url.URL
}

// APIConfig wraps the api settings for the Ovirt.
type APIConfig struct {
	Connection struct {
		APIEntry string `gcfg:"uri"`
		Username string `gcfg:"username"`
		Password string `gcfg:"password"`
	}
	Filters struct {
		VmsQuery string `gcfg:"vms"`
	}
}

// XMLVMAddress is an implementation for the Ovirt instance IP address in xml.
type XMLVMAddress struct {
	Address string `xml:"address,attr"`
}

// XMLVMInfo is an implementation for the Ovirt instance details in xml.
type XMLVMInfo struct {
	UUID      string         `xml:"id,attr"`
	Name      string         `xml:"name"`
	Hostname  string         `xml:"guest_info>fqdn"`
	Addresses []XMLVMAddress `xml:"guest_info>ips>ip"`
	State     string         `xml:"status>state"`
}

// XMLVmsList is an implementation to provide the list of Ovirt instances.
type XMLVmsList struct {
	XMLName xml.Name    `xml:"vms"`
	VM      []XMLVMInfo `xml:"vm"`
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName,
		func(config io.Reader) (cloudprovider.Interface, error) {
			return newOVirtCloud(config)
		})
}

func newOVirtCloud(config io.Reader) (*Cloud, error) {
	if config == nil {
		return nil, fmt.Errorf("missing configuration file for ovirt cloud provider")
	}

	oVirtConfig := APIConfig{}

	/* defaults */
	oVirtConfig.Connection.Username = "admin@internal"

	if err := gcfg.ReadInto(&oVirtConfig, config); err != nil {
		return nil, err
	}

	if oVirtConfig.Connection.APIEntry == "" {
		return nil, fmt.Errorf("missing ovirt uri in cloud provider configuration")
	}

	request, err := url.Parse(oVirtConfig.Connection.APIEntry)
	if err != nil {
		return nil, err
	}

	request.Path = path.Join(request.Path, "vms")
	request.User = url.UserPassword(oVirtConfig.Connection.Username, oVirtConfig.Connection.Password)
	request.RawQuery = url.Values{"search": {oVirtConfig.Filters.VmsQuery}}.Encode()

	return &Cloud{VmsRequest: request}, nil
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (v *Cloud) Initialize(clientBuilder cloudprovider.ControllerClientBuilder, stop <-chan struct{}) {
}

// Clusters returns the list of clusters.
func (v *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (v *Cloud) ProviderName() string {
	return ProviderName
}

// HasClusterID returns true if the cluster has a clusterID
func (v *Cloud) HasClusterID() bool {
	return true
}

// LoadBalancer returns an implementation of LoadBalancer for oVirt cloud
func (v *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}

// Instances returns an implementation of Instances for oVirt cloud
func (v *Cloud) Instances() (cloudprovider.Instances, bool) {
	return v, true
}

// Zones returns an implementation of Zones for oVirt cloud
func (v *Cloud) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

// Routes returns an implementation of Routes for oVirt cloud
func (v *Cloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// NodeAddresses returns the NodeAddresses of the instance with the specified nodeName.
func (v *Cloud) NodeAddresses(ctx context.Context, nodeName types.NodeName) ([]v1.NodeAddress, error) {
	name := mapNodeNameToInstanceName(nodeName)
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

	return []v1.NodeAddress{
		{Type: v1.NodeInternalIP, Address: address.String()},
		{Type: v1.NodeExternalIP, Address: address.String()},
	}, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (v *Cloud) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	return []v1.NodeAddress{}, cloudprovider.NotImplemented
}

// mapNodeNameToInstanceName maps from a k8s NodeName to an ovirt instance name (the hostname)
// This is a simple string cast
func mapNodeNameToInstanceName(nodeName types.NodeName) string {
	return string(nodeName)
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists and is running.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (v *Cloud) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, cloudprovider.NotImplemented
}

// InstanceShutdownByProviderID returns true if the instance is in safe state to detach volumes
func (v *Cloud) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	return false, cloudprovider.NotImplemented
}

// InstanceID returns the cloud provider ID of the node with the specified NodeName.
func (v *Cloud) InstanceID(ctx context.Context, nodeName types.NodeName) (string, error) {
	name := mapNodeNameToInstanceName(nodeName)
	instance, err := v.fetchInstance(name)
	if err != nil {
		return "", err
	}
	// TODO: define a way to identify the provider instance to complete
	// the format <provider_instance_id>/<instance_id>.
	return "/" + instance.UUID, err
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (v *Cloud) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	return "", cloudprovider.NotImplemented
}

// InstanceType returns the type of the specified instance.
func (v *Cloud) InstanceType(ctx context.Context, name types.NodeName) (string, error) {
	return "", nil
}

func getInstancesFromXML(body io.Reader) (InstanceMap, error) {
	if body == nil {
		return nil, fmt.Errorf("ovirt rest-api response body is missing")
	}

	content, err := ioutil.ReadAll(body)
	if err != nil {
		return nil, err
	}

	vmlist := XMLVmsList{}

	if err := xml.Unmarshal(content, &vmlist); err != nil {
		return nil, err
	}

	instances := make(InstanceMap)

	for _, vm := range vmlist.VM {
		// Always return only vms that are up and running
		if vm.Hostname != "" && strings.ToLower(vm.State) == "up" {
			address := ""
			if len(vm.Addresses) > 0 {
				address = vm.Addresses[0].Address
			}

			instances[vm.Hostname] = Instance{
				UUID:      vm.UUID,
				Name:      vm.Name,
				IPAddress: address,
			}
		}
	}

	return instances, nil
}

func (v *Cloud) fetchAllInstances() (InstanceMap, error) {
	response, err := http.Get(v.VmsRequest.String())
	if err != nil {
		return nil, err
	}

	defer response.Body.Close()

	return getInstancesFromXML(response.Body)
}

func (v *Cloud) fetchInstance(name string) (*Instance, error) {
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

// ListSortedNames returns the list of sorted Ovirt instances name.
func (m *InstanceMap) ListSortedNames() []string {
	var names []string

	for k := range *m {
		names = append(names, k)
	}

	sort.Strings(names)

	return names
}

// CurrentNodeName is implementation of Instances.CurrentNodeName.
func (v *Cloud) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return types.NodeName(hostname), nil
}

// AddSSHKeyToAllInstances is currently not implemented.
func (v *Cloud) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}
