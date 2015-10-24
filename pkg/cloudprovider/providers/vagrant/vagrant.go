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

package vagrant_cloud

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	neturl "net/url"
	"sort"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

const ProviderName = "vagrant"

// VagrantCloud is an implementation of Interface, TCPLoadBalancer and Instances for developer managed Vagrant cluster.
type VagrantCloud struct {
	saltURL  string
	saltUser string
	saltPass string
	saltAuth string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) { return newVagrantCloud() })
}

// SaltToken is an authorization token required by Salt REST API.
type SaltToken struct {
	Token string `json:"token"`
	User  string `json:"user"`
	EAuth string `json:"eauth"`
}

// SaltLoginResponse is the response object for a /login operation against Salt REST API.
type SaltLoginResponse struct {
	Data []SaltToken `json:"return"`
}

// SaltMinion is a machine managed by the Salt service.
type SaltMinion struct {
	Roles []string `json:"roles"`
	IP    string   `json:"node_ip"`
	Host  string   `json:"host"`
}

// SaltMinions is a map of minion name to machine information.
type SaltMinions map[string]SaltMinion

// SaltMinionsResponse is the response object for a /minions operation against Salt REST API
type SaltMinionsResponse struct {
	Minions []SaltMinions `json:"return"`
}

// newVagrantCloud creates a new instance of VagrantCloud configured to talk to the Salt REST API.
func newVagrantCloud() (*VagrantCloud, error) {
	return &VagrantCloud{
		saltURL:  "http://kubernetes-master:8000",
		saltUser: "vagrant",
		saltPass: "vagrant",
		saltAuth: "pam",
	}, nil
}

func (v *VagrantCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (v *VagrantCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (v *VagrantCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Vagrant cloud.
func (v *VagrantCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

// Instances returns an implementation of Instances for Vagrant cloud.
func (v *VagrantCloud) Instances() (cloudprovider.Instances, bool) {
	return v, true
}

// Zones returns an implementation of Zones for Vagrant cloud.
func (v *VagrantCloud) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

// Routes returns an implementation of Routes for Vagrant cloud.
func (v *VagrantCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// getInstanceByAddress retuns
func (v *VagrantCloud) getInstanceByAddress(address string) (*SaltMinion, error) {
	token, err := v.saltLogin()
	if err != nil {
		return nil, err
	}
	minions, err := v.saltMinions(token)
	if err != nil {
		return nil, err
	}
	filteredMinions := v.saltMinionsByRole(minions, "kubernetes-pool")
	for _, minion := range filteredMinions {
		// Due to vagrant not running with a dedicated DNS setup, we return the IP address of a minion as its hostname at this time
		if minion.IP == address {
			return &minion, nil
		}
	}
	return nil, fmt.Errorf("unable to find instance for address: %s", address)
}

func (v *VagrantCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// Implementation of Instances.CurrentNodeName
func (v *VagrantCloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

// NodeAddresses returns the NodeAddresses of a particular machine instance.
func (v *VagrantCloud) NodeAddresses(instance string) ([]api.NodeAddress, error) {
	// Due to vagrant not running with a dedicated DNS setup, we return the IP address of a minion as its hostname at this time
	minion, err := v.getInstanceByAddress(instance)
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(minion.IP)
	return []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: ip.String()}}, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (v *VagrantCloud) ExternalID(instance string) (string, error) {
	// Due to vagrant not running with a dedicated DNS setup, we return the IP address of a minion as its hostname at this time
	minion, err := v.getInstanceByAddress(instance)
	if err != nil {
		return "", err
	}
	return minion.IP, nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (v *VagrantCloud) InstanceID(instance string) (string, error) {
	minion, err := v.getInstanceByAddress(instance)
	if err != nil {
		return "", err
	}
	return minion.IP, nil
}

// saltMinionsByRole filters a list of minions that have a matching role.
func (v *VagrantCloud) saltMinionsByRole(minions []SaltMinion, role string) []SaltMinion {
	var filteredMinions []SaltMinion
	for _, value := range minions {
		sort.Strings(value.Roles)
		if pos := sort.SearchStrings(value.Roles, role); pos < len(value.Roles) {
			filteredMinions = append(filteredMinions, value)
		}
	}
	return filteredMinions
}

// saltMinions invokes the Salt API for minions using provided token.
func (v *VagrantCloud) saltMinions(token SaltToken) ([]SaltMinion, error) {
	var minions []SaltMinion

	url := v.saltURL + "/minions"
	req, err := http.NewRequest("GET", url, nil)
	req.Header.Add("X-Auth-Token", token.Token)

	client := &http.Client{}
	resp, err := client.Do(req)

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return minions, err
	}

	var minionsResp SaltMinionsResponse
	if err = json.Unmarshal(body, &minionsResp); err != nil {
		return minions, err
	}

	for _, value := range minionsResp.Minions[0] {
		minions = append(minions, value)
	}

	return minions, nil
}

// saltLogin invokes the Salt API to get an authorization token.
func (v *VagrantCloud) saltLogin() (SaltToken, error) {
	url := v.saltURL + "/login"
	data := neturl.Values{
		"username": {v.saltUser},
		"password": {v.saltPass},
		"eauth":    {v.saltAuth},
	}

	var token SaltToken
	resp, err := http.PostForm(url, data)
	if err != nil {
		return token, err
	}

	defer resp.Body.Close()
	body, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return token, err
	}

	var loginResp SaltLoginResponse
	if err := json.Unmarshal(body, &loginResp); err != nil {
		return token, err
	}

	if len(loginResp.Data) == 0 {
		return token, errors.New("No token found in response")
	}

	return loginResp.Data[0], nil
}

// List enumerates the set of minions instances known by the cloud provider.
func (v *VagrantCloud) List(filter string) ([]string, error) {
	token, err := v.saltLogin()
	if err != nil {
		return nil, err
	}

	minions, err := v.saltMinions(token)
	if err != nil {
		return nil, err
	}

	filteredMinions := v.saltMinionsByRole(minions, "kubernetes-pool")
	var instances []string
	for _, instance := range filteredMinions {
		// With no dedicated DNS setup in cluster, IP address is used as hostname
		instances = append(instances, instance.IP)
	}

	return instances, nil
}
