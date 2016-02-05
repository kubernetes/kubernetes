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

package main

import (
	"net"

	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
)

type ELBConfig struct {
	Target  *api.Service
	Weights []HostWeight
}

type HostWeight struct {
	Host   net.IP
	Weight int
}

func sync() {
	configs := make([]*ELBConfig, 0)
	for _, o := range services.List() {
		svc := o.(*api.Service)
		config := serviceELBConfig(svc)
		configs = append(configs, config)
	}

	syncIPRoute(configs)
}

func serviceELBConfig(svc *api.Service) *ELBConfig {
	config := &ELBConfig{
		Target:  svc,
		Weights: make([]HostWeight, 0),
	}

	glog.V(4).Info("Building config for service ", svc.Namespace, "/", svc.Name)
	for _, ep := range endpointsOfService(svc) {
		for _, host := range hostsOfEndpoint(ep) {
			config.recordHostHit(host)
		}
	}

	return config
}

func endpointsOfService(svc *api.Service) []*api.Endpoints {
	eps := make([]*api.Endpoints, 0)
	for _, o := range endpoints.List() {
		ep := o.(*api.Endpoints)
		if ep.Namespace == svc.Namespace && ep.Name == svc.Name {
			eps = append(eps, ep)
		}
	}
	return eps
}

func hostsOfEndpoint(ep *api.Endpoints) []net.IP {
	ips := make([]net.IP, 0)
	for _, ss := range ep.Subsets {
		for _, a := range ss.Addresses {
			glog.V(4).Info("- ", a.IP, " -> ", a.TargetRef)
			ref := a.TargetRef
			if ref == nil {
				continue
			}
			key := ref.Namespace + "/" + ref.Name
			o, exists, err := pods.GetByKey(key)
			if err != nil {
				panic(err)
			}
			if !exists {
				continue
			}
			pod := o.(*api.Pod)
			if pod.Status.HostIP == "" {
				continue
			}
			ips = append(ips, net.ParseIP(pod.Status.HostIP))
		}
	}
	return ips
}

func (c *ELBConfig) recordHostHit(host net.IP) {
	found := false
	for _, hw := range c.Weights {
		if host.Equal(hw.Host) {
			hw.Weight += 1
			found = true
			break
		}
	}
	if !found {
		w := HostWeight{host, 1}
		c.Weights = append(c.Weights, w)
	}
}
