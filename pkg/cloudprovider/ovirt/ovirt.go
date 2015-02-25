/*
Copyright 2014 Google Inc. All rights reserved.

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
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"path"
	"strings"

	"code.google.com/p/gcfg"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

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

type XmlVmInfo struct {
	Hostname string `xml:"guest_info>fqdn"`
	State    string `xml:"status>state"`
}

type XmlVmsList struct {
	XMLName xml.Name    `xml:"vms"`
	Vm      []XmlVmInfo `xml:"vm"`
}

func init() {
	cloudprovider.RegisterCloudProvider("ovirt",
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

// IPAddress returns the address of a particular machine instance
func (v *OVirtCloud) IPAddress(instance string) (net.IP, error) {
	// since the instance now is the IP in the ovirt env, this is trivial no-op
	ip, err := net.LookupIP(instance)
	if err != nil || len(ip) < 1 {
		return nil, fmt.Errorf("cannot find ip address for: %s", instance)
	}
	return ip[0], nil
}

func getInstancesFromXml(body io.Reader) ([]string, error) {
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

	var instances []string

	for _, vm := range vmlist.Vm {
		// Always return only vms that are up and running
		if vm.Hostname != "" && strings.ToLower(vm.State) == "up" {
			instances = append(instances, vm.Hostname)
		}
	}

	return instances, nil
}

// List enumerates the set of minions instances known by the cloud provider
func (v *OVirtCloud) List(filter string) ([]string, error) {
	response, err := http.Get(v.VmsRequest.String())
	if err != nil {
		return nil, err
	}

	defer response.Body.Close()

	return getInstancesFromXml(response.Body)
}

func (v *OVirtCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	return nil, nil
}
