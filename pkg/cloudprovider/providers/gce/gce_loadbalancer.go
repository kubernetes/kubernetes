/*
Copyright 2017 The Kubernetes Authors.

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

package gce

import (
	"flag"
	"fmt"
	"net"
	"strings"

	"github.com/golang/glog"

	"k8s.io/api/core/v1"
	"k8s.io/kubernetes/pkg/cloudprovider"
	netsets "k8s.io/kubernetes/pkg/util/net/sets"
)

type cidrs struct {
	ipn   netsets.IPNet
	isSet bool
}

var (
	lbSrcRngsFlag cidrs
)

func newLoadBalancerMetricContext(request, region string) *metricContext {
	return newGenericMetricContext("loadbalancer", request, region, unusedMetricLabel, computeV1Version)
}

type lbScheme string

const (
	schemeExternal lbScheme = "EXTERNAL"
	schemeInternal lbScheme = "INTERNAL"
)

func init() {
	var err error
	// LB L7 proxies and all L3/4/7 health checkers have client addresses within these known CIDRs.
	lbSrcRngsFlag.ipn, err = netsets.ParseIPNets([]string{"130.211.0.0/22", "35.191.0.0/16", "209.85.152.0/22", "209.85.204.0/22"}...)
	if err != nil {
		panic("Incorrect default GCE L7 source ranges")
	}

	flag.Var(&lbSrcRngsFlag, "cloud-provider-gce-lb-src-cidrs", "CIDRS opened in GCE firewall for LB traffic proxy & health checks")
}

// String is the method to format the flag's value, part of the flag.Value interface.
func (c *cidrs) String() string {
	return strings.Join(c.ipn.StringSlice(), ",")
}

// Set supports a value of CSV or the flag repeated multiple times
func (c *cidrs) Set(value string) error {
	// On first Set(), clear the original defaults
	if !c.isSet {
		c.isSet = true
		c.ipn = make(netsets.IPNet)
	} else {
		return fmt.Errorf("GCE LB CIDRS have already been set")
	}

	for _, cidr := range strings.Split(value, ",") {
		_, ipnet, err := net.ParseCIDR(cidr)
		if err != nil {
			return err
		}

		c.ipn.Insert(ipnet)
	}
	return nil
}

// LoadBalancerSrcRanges contains the ranges of ips used by the GCE load balancers (l4 & L7)
// for proxying client requests and performing health checks.
func LoadBalancerSrcRanges() []string {
	return lbSrcRngsFlag.ipn.StringSlice()
}

// GetLoadBalancer is an implementation of LoadBalancer.GetLoadBalancer
func (gce *GCECloud) GetLoadBalancer(clusterName string, svc *v1.Service) (*v1.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	fwd, err := gce.GetRegionForwardingRule(loadBalancerName, gce.region)
	if err == nil {
		status := &v1.LoadBalancerStatus{}
		status.Ingress = []v1.LoadBalancerIngress{{IP: fwd.IPAddress}}

		return status, true, nil
	}
	return nil, false, ignoreNotFound(err)
}

// EnsureLoadBalancer is an implementation of LoadBalancer.EnsureLoadBalancer.
func (gce *GCECloud) EnsureLoadBalancer(clusterName string, svc *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	desiredScheme := getSvcScheme(svc)
	clusterID, err := gce.ClusterID.GetID()
	if err != nil {
		return nil, err
	}

	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v): ensure %v loadbalancer", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, desiredScheme)

	existingFwdRule, err := gce.GetRegionForwardingRule(loadBalancerName, gce.region)
	if err != nil && !isNotFound(err) {
		return nil, err
	}

	if existingFwdRule != nil {
		existingScheme := lbScheme(strings.ToUpper(existingFwdRule.LoadBalancingScheme))

		// If the loadbalancer type changes between INTERNAL and EXTERNAL, the old load balancer should be deleted.
		if existingScheme != desiredScheme {
			glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v): deleting existing %v loadbalancer", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, existingScheme)
			switch existingScheme {
			case schemeInternal:
				err = gce.ensureInternalLoadBalancerDeleted(clusterName, clusterID, svc)
			default:
				err = gce.ensureExternalLoadBalancerDeleted(clusterName, clusterID, svc)
			}
			glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v): done deleting existing %v loadbalancer. err: %v", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, existingScheme, err)
			if err != nil {
				return nil, err
			}

			// Assume the ensureDeleted function successfully deleted the forwarding rule.
			existingFwdRule = nil
		}
	}

	var status *v1.LoadBalancerStatus
	switch desiredScheme {
	case schemeInternal:
		status, err = gce.ensureInternalLoadBalancer(clusterName, clusterID, svc, existingFwdRule, nodes)
	default:
		status, err = gce.ensureExternalLoadBalancer(clusterName, clusterID, svc, existingFwdRule, nodes)
	}
	glog.V(4).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v): done ensuring loadbalancer. err: %v", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, err)
	return status, err
}

// UpdateLoadBalancer is an implementation of LoadBalancer.UpdateLoadBalancer.
func (gce *GCECloud) UpdateLoadBalancer(clusterName string, svc *v1.Service, nodes []*v1.Node) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	scheme := getSvcScheme(svc)
	clusterID, err := gce.ClusterID.GetID()
	if err != nil {
		return err
	}

	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v, %v, %v): updating with %d nodes", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, len(nodes))

	switch scheme {
	case schemeInternal:
		err = gce.updateInternalLoadBalancer(clusterName, clusterID, svc, nodes)
	default:
		err = gce.updateExternalLoadBalancer(clusterName, svc, nodes)
	}
	glog.V(4).Infof("UpdateLoadBalancer(%v, %v, %v, %v, %v): done updating. err: %v", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, err)
	return err
}

// EnsureLoadBalancerDeleted is an implementation of LoadBalancer.EnsureLoadBalancerDeleted.
func (gce *GCECloud) EnsureLoadBalancerDeleted(clusterName string, svc *v1.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(svc)
	scheme := getSvcScheme(svc)
	clusterID, err := gce.ClusterID.GetID()
	if err != nil {
		return err
	}

	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v, %v, %v, %v, %v): deleting loadbalancer", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region)

	switch scheme {
	case schemeInternal:
		err = gce.ensureInternalLoadBalancerDeleted(clusterName, clusterID, svc)
	default:
		err = gce.ensureExternalLoadBalancerDeleted(clusterName, clusterID, svc)
	}
	glog.V(4).Infof("EnsureLoadBalancerDeleted(%v, %v, %v, %v, %v): done deleting loadbalancer. err: %v", clusterName, svc.Namespace, svc.Name, loadBalancerName, gce.region, err)
	return err
}

func getSvcScheme(svc *v1.Service) lbScheme {
	if typ, ok := GetLoadBalancerAnnotationType(svc); ok && typ == LBTypeInternal {
		return schemeInternal
	}
	return schemeExternal
}
