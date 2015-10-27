/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

package mesos

import (
	"errors"
	"fmt"
	"io"
	"net"
	"regexp"

	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/detector"
	"golang.org/x/net/context"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/cloudprovider"
)

var (
	ProviderName  = "mesos"
	CloudProvider *MesosCloud

	noHostNameSpecified = errors.New("No hostname specified")
)

func init() {
	cloudprovider.RegisterCloudProvider(
		ProviderName,
		func(configReader io.Reader) (cloudprovider.Interface, error) {
			provider, err := newMesosCloud(configReader)
			if err == nil {
				CloudProvider = provider
			}
			return provider, err
		})
}

type MesosCloud struct {
	client *mesosClient
	config *Config
}

func (c *MesosCloud) MasterURI() string {
	return c.config.MesosMaster
}

func newMesosCloud(configReader io.Reader) (*MesosCloud, error) {
	config, err := readConfig(configReader)
	if err != nil {
		return nil, err
	}

	log.V(1).Infof("new mesos cloud, master='%v'", config.MesosMaster)
	if d, err := detector.New(config.MesosMaster); err != nil {
		log.V(1).Infof("failed to create master detector: %v", err)
		return nil, err
	} else if cl, err := newMesosClient(d,
		config.MesosHttpClientTimeout.Duration,
		config.StateCacheTTL.Duration); err != nil {
		log.V(1).Infof("failed to create mesos cloud client: %v", err)
		return nil, err
	} else {
		return &MesosCloud{client: cl, config: config}, nil
	}
}

// Implementation of Instances.CurrentNodeName
func (c *MesosCloud) CurrentNodeName(hostname string) (string, error) {
	return hostname, nil
}

func (c *MesosCloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// Instances returns a copy of the Mesos cloud Instances implementation.
// Mesos natively provides minimal cloud-type resources. More robust cloud
// support requires a combination of Mesos and cloud-specific knowledge.
func (c *MesosCloud) Instances() (cloudprovider.Instances, bool) {
	return c, true
}

// TCPLoadBalancer always returns nil, false in this implementation.
// Mesos does not provide any type of native load balancing by default,
// so this implementation always returns (nil, false).
func (c *MesosCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	return nil, false
}

// Zones always returns nil, false in this implementation.
// Mesos does not provide any type of native region or zone awareness,
// so this implementation always returns (nil, false).
func (c *MesosCloud) Zones() (cloudprovider.Zones, bool) {
	return nil, false
}

// Clusters returns a copy of the Mesos cloud Clusters implementation.
// Mesos does not provide support for multiple clusters.
func (c *MesosCloud) Clusters() (cloudprovider.Clusters, bool) {
	return c, true
}

// Routes always returns nil, false in this implementation.
func (c *MesosCloud) Routes() (cloudprovider.Routes, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (c *MesosCloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (c *MesosCloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// ListClusters lists the names of the available Mesos clusters.
func (c *MesosCloud) ListClusters() ([]string, error) {
	// Always returns a single cluster (this one!)
	ctx, cancel := context.WithCancel(context.TODO())
	defer cancel()
	name, err := c.client.clusterName(ctx)
	return []string{name}, err
}

// Master gets back the address (either DNS name or IP address) of the leading Mesos master node for the cluster.
func (c *MesosCloud) Master(clusterName string) (string, error) {
	clusters, err := c.ListClusters()
	if err != nil {
		return "", err
	}
	for _, name := range clusters {
		if name == clusterName {
			if c.client.master == "" {
				return "", errors.New("The currently leading master is unknown.")
			}

			host, _, err := net.SplitHostPort(c.client.master)
			if err != nil {
				return "", err
			}

			return host, nil
		}
	}
	return "", errors.New(fmt.Sprintf("The supplied cluster '%v' does not exist", clusterName))
}

// ipAddress returns an IP address of the specified instance.
func ipAddress(name string) (net.IP, error) {
	if name == "" {
		return nil, noHostNameSpecified
	}
	ipaddr := net.ParseIP(name)
	if ipaddr != nil {
		return ipaddr, nil
	}
	iplist, err := net.LookupIP(name)
	if err != nil {
		log.V(2).Infof("failed to resolve IP from host name '%v': %v", name, err)
		return nil, err
	}
	ipaddr = iplist[0]
	log.V(2).Infof("resolved host '%v' to '%v'", name, ipaddr)
	return ipaddr, nil
}

// ExternalID returns the cloud provider ID of the specified instance (deprecated).
func (c *MesosCloud) ExternalID(instance string) (string, error) {
	ip, err := ipAddress(instance)
	if err != nil {
		return "", err
	}
	return ip.String(), nil
}

// InstanceID returns the cloud provider ID of the specified instance.
func (c *MesosCloud) InstanceID(name string) (string, error) {
	return "", nil
}

// List lists instances that match 'filter' which is a regular expression
// which must match the entire instance name (fqdn).
func (c *MesosCloud) List(filter string) ([]string, error) {
	//TODO(jdef) use a timeout here? 15s?
	ctx, cancel := context.WithCancel(context.TODO())
	defer cancel()

	nodes, err := c.client.listSlaves(ctx)
	if err != nil {
		return nil, err
	}
	if len(nodes) == 0 {
		log.V(2).Info("no slaves found, are any running?")
		return nil, nil
	}
	filterRegex, err := regexp.Compile(filter)
	if err != nil {
		return nil, err
	}
	addr := []string{}
	for _, node := range nodes {
		if filterRegex.MatchString(node.hostname) {
			addr = append(addr, node.hostname)
		}
	}
	return addr, err
}

// NodeAddresses returns the addresses of the specified instance.
func (c *MesosCloud) NodeAddresses(name string) ([]api.NodeAddress, error) {
	ip, err := ipAddress(name)
	if err != nil {
		return nil, err
	}
	return []api.NodeAddress{{Type: api.NodeLegacyHostIP, Address: ip.String()}}, nil
}
