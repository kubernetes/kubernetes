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

package cloudprovider

import (
	"fmt"
	"io/ioutil"
	"net"
	"net/http"
	"os/exec"
	"strconv"
	"strings"
	"time"

	"code.google.com/p/goauth2/compute/serviceaccount"
	compute "code.google.com/p/google-api-go-client/compute/v1"
)

// GCECloud is an implementation of Interface, TCPLoadBalancer and Instances for Google Compute Engine.
type GCECloud struct {
	service    *compute.Service
	projectID  string
	zone       string
	instanceRE string
}

func getProjectAndZone() (string, string, error) {
	client := http.Client{}
	url := "http://metadata/computeMetadata/v1/instance/zone"
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return "", "", err
	}
	req.Header.Add("X-Google-Metadata-Request", "True")
	res, err := client.Do(req)
	if err != nil {
		return "", "", err
	}
	defer res.Body.Close()
	data, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return "", "", err
	}
	parts := strings.Split(string(data), "/")
	if len(parts) != 4 {
		return "", "", fmt.Errorf("Unexpected response: %s", string(data))
	}
	return parts[1], parts[3], nil
}

// NewGCECloud creates a new instance of GCECloud.
func NewGCECloud() (*GCECloud, error) {
	projectID, zone, err := getProjectAndZone()
	if err != nil {
		return nil, err
	}
	client, err := serviceaccount.NewClient(&serviceaccount.Options{})
	if err != nil {
		return nil, err
	}
	svc, err := compute.New(client)
	if err != nil {
		return nil, err
	}
	return &GCECloud{
		service:   svc,
		projectID: projectID,
		zone:      zone,
	}, nil
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Google Compute Engine.
func (gce *GCECloud) TCPLoadBalancer() (TCPLoadBalancer, bool) {
	return gce, true
}

// Instances returns an implementation of Instances for Google Compute Engine.
func (gce *GCECloud) Instances() (Instances, bool) {
	return gce, true
}

// Zones returns an implementation of Zones for Google Compute Engine.
func (gce *GCECloud) Zones() (Zones, bool) {
	return gce, true
}

func makeHostLink(projectID, zone, host string) string {
	ix := strings.Index(host, ".")
	if ix != -1 {
		host = host[:ix]
	}
	return fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/zones/%s/instances/%s",
		projectID, zone, host)
}

func (gce *GCECloud) makeTargetPool(name, region string, hosts []string) (string, error) {
	var instances []string
	for _, host := range hosts {
		instances = append(instances, makeHostLink(gce.projectID, gce.zone, host))
	}
	pool := &compute.TargetPool{
		Name:      name,
		Instances: instances,
	}
	_, err := gce.service.TargetPools.Insert(gce.projectID, region, pool).Do()
	if err != nil {
		return "", err
	}
	link := fmt.Sprintf("https://www.googleapis.com/compute/v1/projects/%s/regions/%s/targetPools/%s", gce.projectID, region, name)
	return link, nil
}

func (gce *GCECloud) waitForRegionOp(op *compute.Operation, region string) error {
	pollOp := op
	for pollOp.Status != "DONE" {
		var err error
		time.Sleep(time.Second * 10)
		pollOp, err = gce.service.RegionOperations.Get(gce.projectID, region, op.Name).Do()
		if err != nil {
			return err
		}
	}
	return nil
}

// TCPLoadBalancerExists is an implementation of TCPLoadBalancer.TCPLoadBalancerExists.
func (gce *GCECloud) TCPLoadBalancerExists(name, region string) (bool, error) {
	_, err := gce.service.ForwardingRules.Get(gce.projectID, region, name).Do()
	return false, err
}

// CreateTCPLoadBalancer is an implementation of TCPLoadBalancer.CreateTCPLoadBalancer.
func (gce *GCECloud) CreateTCPLoadBalancer(name, region string, port int, hosts []string) error {
	pool, err := gce.makeTargetPool(name, region, hosts)
	if err != nil {
		return err
	}
	req := &compute.ForwardingRule{
		Name:       name,
		IPProtocol: "TCP",
		PortRange:  strconv.Itoa(port),
		Target:     pool,
	}
	_, err = gce.service.ForwardingRules.Insert(gce.projectID, region, req).Do()
	return err
}

// UpdateTCPLoadBalancer is an implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
func (gce *GCECloud) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	var refs []*compute.InstanceReference
	for _, host := range hosts {
		refs = append(refs, &compute.InstanceReference{host})
	}
	req := &compute.TargetPoolsAddInstanceRequest{
		Instances: refs,
	}

	_, err := gce.service.TargetPools.AddInstance(gce.projectID, region, name, req).Do()
	return err
}

// DeleteTCPLoadBalancer is an implementation of TCPLoadBalancer.DeleteTCPLoadBalancer.
func (gce *GCECloud) DeleteTCPLoadBalancer(name, region string) error {
	_, err := gce.service.ForwardingRules.Delete(gce.projectID, region, name).Do()
	if err != nil {
		return err
	}
	_, err = gce.service.TargetPools.Delete(gce.projectID, region, name).Do()
	return err
}

// IPAddress is an implementation of Instances.IPAddress.
func (gce *GCECloud) IPAddress(instance string) (net.IP, error) {
	res, err := gce.service.Instances.Get(gce.projectID, gce.zone, instance).Do()
	if err != nil {
		return nil, err
	}
	ip := net.ParseIP(res.NetworkInterfaces[0].AccessConfigs[0].NatIP)
	if ip == nil {
		return nil, fmt.Errorf("Invalid network IP: %s", res.NetworkInterfaces[0].AccessConfigs[0].NatIP)
	}
	return ip, nil
}

// This is hacky, compute the delta between hostame and hostname -f
func fqdnSuffix() (string, error) {
	fullHostname, err := exec.Command("hostname", "-f").Output()
	if err != nil {
		return "", err
	}
	hostname, err := exec.Command("hostname").Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(fullHostname)[len(string(hostname)):]), nil
}

// List is an implementation of Instances.List.
func (gce *GCECloud) List(filter string) ([]string, error) {
	// GCE gives names without their fqdn suffix, so get that here for appending.
	// This is needed because the kubelet looks for its jobs in /registry/hosts/<fqdn>/pods
	// We should really just replace this convention, with a negotiated naming protocol for kubelet's
	// to register with the master.
	suffix, err := fqdnSuffix()
	if err != nil {
		return []string{}, err
	}
	if len(suffix) > 0 {
		suffix = "." + suffix
	}
	listCall := gce.service.Instances.List(gce.projectID, gce.zone)
	if len(filter) > 0 {
		listCall = listCall.Filter("name eq " + filter)
	}
	res, err := listCall.Do()
	if err != nil {
		return nil, err
	}
	var instances []string
	for _, instance := range res.Items {
		instances = append(instances, instance.Name+suffix)
	}
	return instances, nil
}

func (gce *GCECloud) GetZone() (string, error) {
	return gce.zone, nil
}
