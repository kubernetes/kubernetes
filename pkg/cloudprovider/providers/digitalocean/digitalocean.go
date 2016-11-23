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

package digitalocean

import (
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"strconv"
	"strings"

	"gopkg.in/gcfg.v1"

	"github.com/digitalocean/godo"
	"golang.org/x/oauth2"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
	"k8s.io/kubernetes/pkg/types"
)

const ProviderName = "digitalocean"

var ErrNotFound = errors.New("Failed to find object")
var ErrMultipleResults = errors.New("Multiple results where only one expected")
var ErrNoAddressFound = errors.New("No address found for host")
var ErrAttrNotFound = errors.New("Expected attribute not found")

type DigitalOcean struct {
	provider       *godo.Client
	region         string
	selfDOInstance *doDroplet
}

type Config struct {
	Global struct {
		ApiKey string `gcfg:"apikey"`
		Region string `gcfg:"region"`
	}
}

type doDroplet struct {
	ID          int
	Name        string
	PrivateIPv4 string
	PublicIPv4  string
	SizeSlug    string
	Region      string
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readConfig(config)
		if err != nil {
			return nil, err
		}
		return newDigitalOcean(cfg)
	})
}

func readConfig(config io.Reader) (Config, error) {
	if config == nil {
		err := fmt.Errorf("no DigitalOcean cloud provider config file given. Restart process with --cloud-provider=digitalocean --cloud-config=[path_to_config_file]")
		return Config{}, err
	}

	var cfg Config
	err := gcfg.ReadInto(&cfg, config)
	return cfg, err
}

func (t *TokenSource) Token() (*oauth2.Token, error) {
	token := &oauth2.Token{
		AccessToken: t.AccessToken,
	}
	return token, nil
}

type TokenSource struct {
	AccessToken string
}

func newDigitalOcean(cfg Config) (*DigitalOcean, error) {
	tokenSource := &TokenSource{
		AccessToken: cfg.Global.ApiKey,
	}
	oauthClient := oauth2.NewClient(oauth2.NoContext, tokenSource)
	provider := godo.NewClient(oauthClient)

	_, _, err := provider.Account.Get()
	if err != nil {
		return nil, err
	}
	do := DigitalOcean{
		provider: provider,
		region:   cfg.Global.Region,
	}

	// build self DigitalOcean Instance information
	selfDOInstance, err := do.buildSelfDOInstance()
	if err != nil {
		return nil, err
	}
	do.selfDOInstance = selfDOInstance

	glog.V(2).Infof("DigitalOcean Droplet with droplet ID: %d", do.selfDOInstance.ID)

	return &do, nil
}

func (do *DigitalOcean) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// implementation of interfaces
func (do *DigitalOcean) Instances() (cloudprovider.Instances, bool) {
	return do, true
}
func (do *DigitalOcean) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return nil, false
}
func (do *DigitalOcean) Zones() (cloudprovider.Zones, bool) {
	return do, false
}
func (do *DigitalOcean) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ScrubDNS filters DNS settings for pods.
func (do *DigitalOcean) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// ProviderName returns the cloud provider ID.
func (do *DigitalOcean) ProviderName() string {
	return ProviderName
}

// helper func
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
func (do *DigitalOcean) findDroplet(name types.NodeName) (*godo.Droplet, error) {
	// not in cache, do request
	glog.V(2).Infof("DigitalOcean: doing findDroplet request for: %s", name)
	listOptions := &godo.ListOptions{
		Page:    1,
		PerPage: 200,
	}
	droplets, _, err := do.provider.Droplets.List(listOptions)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(droplets); i++ {
		if strings.ToLower(string(name)) == strings.ToLower(droplets[i].Name) {
			// cache droplet name and size if this is ourselves
			if do.selfDOInstance.ID == droplets[i].ID {
				err = do.cacheDroplet(&droplets[i])
				if err != nil {
					glog.V(2).Infof("Unable to cache droplet")
				}
			}
			return &droplets[i], nil
		}
		ipv4, err := droplets[i].PrivateIPv4()
		if err == nil && string(name) == ipv4 {
			return &droplets[i], nil
		}
		ipv4, err = droplets[i].PublicIPv4()
		if err == nil && string(name) == ipv4 {
			return &droplets[i], nil
		}
	}
	return nil, ErrNotFound
}
func (do *DigitalOcean) findDropletByFilter(filter string) ([]types.NodeName, error) {
	list := []types.NodeName{}
	listOptions := &godo.ListOptions{
		Page:    1,
		PerPage: 200,
	}
	droplets, _, err := do.provider.Droplets.List(listOptions)
	if err != nil {
		return nil, err
	}
	for i := 0; i < len(droplets); i++ {
		if strings.Contains(droplets[i].Name, filter) {
			list = append(list, types.NodeName(droplets[i].Name))
		}
	}
	return list, nil
}
func (do *DigitalOcean) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	internalIP := ""
	externalIP := ""
	if string(name) == do.selfDOInstance.Name {
		// cached, get from selfDOInstance
		internalIP = do.selfDOInstance.PrivateIPv4
		externalIP = do.selfDOInstance.PublicIPv4
	} else {
		// do api call
		droplet, err := do.findDroplet(name)
		if err != nil {
			return nil, err
		}
		internalIP, err = droplet.PrivateIPv4()
		if err != nil {
			return nil, err
		}
		externalIP, err = droplet.PublicIPv4()
		if err != nil {
			return nil, err
		}
	}
	addresses := []api.NodeAddress{}
	addresses = append(addresses, api.NodeAddress{Type: api.NodeInternalIP, Address: internalIP})
	// Legacy compatibility: the private ip was the legacy host ip
	addresses = append(addresses, api.NodeAddress{Type: api.NodeLegacyHostIP, Address: internalIP})
	addresses = append(addresses, api.NodeAddress{Type: api.NodeExternalIP, Address: externalIP})
	return addresses, nil
}
func (do *DigitalOcean) ExternalID(nodeName types.NodeName) (string, error) {
	return do.InstanceID(nodeName)
}

func (do *DigitalOcean) InstanceID(nodeName types.NodeName) (string, error) {
	if string(nodeName) == do.selfDOInstance.Name {
		// cached, get from selfDOInstance
		return strconv.Itoa(do.selfDOInstance.ID), nil
	} else {
		droplet, err := do.findDroplet(nodeName)
		if err != nil {
			return "", cloudprovider.InstanceNotFound
		} else {
			return strconv.Itoa(droplet.ID), nil
		}
	}
}
func (do *DigitalOcean) LocalInstanceID() (string, error) {
	return strconv.Itoa(do.selfDOInstance.ID), nil
}

func (do *DigitalOcean) InstanceType(nodeName types.NodeName) (string, error) {
	if string(nodeName) == do.selfDOInstance.Name {
		// cached, get from selfDOInstance
		return do.selfDOInstance.SizeSlug, nil
	} else {
		droplet, err := do.findDroplet(nodeName)
		if err != nil {
			return "", cloudprovider.InstanceNotFound
		} else {
			return droplet.SizeSlug, nil
		}
	}
}
func (do *DigitalOcean) List(filter string) ([]types.NodeName, error) {
	list, err := do.findDropletByFilter(filter)
	if err != nil {
		return nil, cloudprovider.InstanceNotFound
	} else {
		return list, nil
	}
}

// AddSSHKeyToAllInstances is currently unimplemented
func (do *DigitalOcean) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}
func (do *DigitalOcean) CurrentNodeName(hostname string) (types.NodeName, error) {
	if hostname == do.selfDOInstance.Name {
		// cached, get from selfDOInstance
		return types.NodeName(do.selfDOInstance.Name), nil
	} else {
		droplet, err := do.findDroplet(types.NodeName(hostname))
		if err != nil {
			return "", cloudprovider.InstanceNotFound
		} else {
			return types.NodeName(strings.ToLower(droplet.Name)), nil
		}
	}
}

func (do *DigitalOcean) GetRegion() string {
	return do.region
}

// Zones
func (do *DigitalOcean) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: do.selfDOInstance.Region,
		Region:        do.selfDOInstance.Region,
	}, nil
}

// TODO: mock this in tests
func getValueFromDOMetadata(path string) (string, error) {
	resp, err := http.Get("http://169.254.169.254/metadata/v1/" + path)
	if err != nil {
		glog.V(2).Infof("error fetching %v from metadata service: %v", path, err)
		return "", err
	}
	defer resp.Body.Close()
	value, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		glog.V(2).Infof("error reading droplet %v: %v", path, err)
		return "", err
	}
	return string(value), nil
}

// metadata
func (do *DigitalOcean) buildSelfDOInstance() (*doDroplet, error) {
	if do.selfDOInstance != nil {
		panic("do not call buildSelfDOInstance directly")
	}

	dropletRegion, err := getValueFromDOMetadata("region")
	if err != nil {
		return nil, err
	}

	dropletID, err := getValueFromDOMetadata("id")
	if err != nil {
		return nil, err
	}
	intDropletID, err := strconv.Atoi(string(dropletID))
	if err != nil {
		glog.V(2).Infof("dropletID %v is invalid, expected an integer", dropletID)
		return nil, err
	}

	hostname, err := getValueFromDOMetadata("hostname")
	if err != nil {
		return nil, err
	}

	// TODO: figure out what happens if there is no private IP
	privateIPv4, err := getValueFromDOMetadata("interfaces/private/0/ipv4/address")
	if err != nil {
		return nil, err
	}

	publicIPv4, err := getValueFromDOMetadata("interfaces/public/0/ipv4/address")
	if err != nil {
		return nil, err
	}

	self := &doDroplet{
		ID:          intDropletID,
		Name:        string(hostname),
		PrivateIPv4: string(privateIPv4),
		PublicIPv4:  string(publicIPv4),
		Region:      string(dropletRegion),
	}
	return self, nil
}

func (do *DigitalOcean) cacheDroplet(droplet *godo.Droplet) error {
	do.selfDOInstance.Name = strings.ToLower(droplet.Name)
	do.selfDOInstance.Region = droplet.Region.Slug
	do.selfDOInstance.SizeSlug = droplet.SizeSlug
	privateIPv4, err := droplet.PrivateIPv4()
	if err == nil {
		do.selfDOInstance.PrivateIPv4 = ""
	} else {
		do.selfDOInstance.PrivateIPv4 = privateIPv4
	}
	publicIPv4, err := droplet.PublicIPv4()
	if err == nil {
		do.selfDOInstance.PublicIPv4 = ""
	} else {
		do.selfDOInstance.PublicIPv4 = publicIPv4
	}
	return nil
}
