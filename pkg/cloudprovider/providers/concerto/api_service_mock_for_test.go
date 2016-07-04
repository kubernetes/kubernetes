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

package concerto_cloud

import (
	"errors"
	"fmt"
	"math/rand"
	"regexp"

	"k8s.io/kubernetes/pkg/cloudprovider"
)

// ConcertoAPIService mock implementation
type ConcertoAPIServiceMock struct {
	instances         []ConcertoInstance
	balancers         []ConcertoLoadBalancer
	balancedInstances map[string][]string // key is LB id, value is a list of nodes ids
}

func (mock *ConcertoAPIServiceMock) GetInstanceByName(name string) (ConcertoInstance, error) {
	for _, ci := range mock.instances {
		if ci.Name == name {
			return ci, nil
		}
	}
	return ConcertoInstance{}, cloudprovider.InstanceNotFound
}

func (mock *ConcertoAPIServiceMock) GetInstanceList() ([]ConcertoInstance, error) {
	return mock.instances, nil
}

func (mock *ConcertoAPIServiceMock) GetLoadBalancerByName(name string) (*ConcertoLoadBalancer, error) {
	matchError, _ := regexp.MatchString("Error", name)
	if matchError {
		return nil, fmt.Errorf("Take this!")
	}
	for _, lb := range mock.balancers {
		if lb.Name == name {
			return &lb, nil
		}
	}
	return nil, nil
}

func (mock *ConcertoAPIServiceMock) DeleteLoadBalancerById(id string) error {
	for i, lb := range mock.balancers {
		if lb.Id == id {
			if i == len(mock.balancers)-1 {
				mock.balancers = mock.balancers[:i]
			} else {
				mock.balancers = append(mock.balancers[:i], mock.balancers[i+1:]...)
			}
			return nil
		}
	}
	return errors.New("load balancer delete error")
}

func (mock *ConcertoAPIServiceMock) GetLoadBalancerNodes(loadBalancerId string) ([]ConcertoLoadBalancerNode, error) {
	var nodes []ConcertoLoadBalancerNode

	ips, err := mock.GetLoadBalancerNodesAsIPs(loadBalancerId)
	if err != nil {
		return nil, err
	}

	for _, ip := range ips {
		nodes = append(nodes, ConcertoLoadBalancerNode{ip, ip})
	}

	return nodes, nil
}

func (mock *ConcertoAPIServiceMock) GetLoadBalancerNodesAsIPs(loadBalancerId string) ([]string, error) {
	return mock.balancedInstances[loadBalancerId], nil
}

func (mock *ConcertoAPIServiceMock) RegisterInstancesWithLoadBalancer(loadBalancerId string, instancesIds []string) error {
	nodes := mock.balancedInstances[loadBalancerId]
	mock.balancedInstances[loadBalancerId] = append(nodes, instancesIds...)
	return nil
}

func (mock *ConcertoAPIServiceMock) DeregisterInstancesFromLoadBalancer(loadBalancerId string, instancesIds []string) error {
	mock.balancedInstances[loadBalancerId] = subtractStringArrays(mock.balancedInstances[loadBalancerId], instancesIds)
	return nil
}

func (mock *ConcertoAPIServiceMock) CreateLoadBalancer(name string, port int, nodePort int) (*ConcertoLoadBalancer, error) {
	lb := ConcertoLoadBalancer{
		Id:   string(rand.Int63()),
		Name: name,
		Port: port,
	}
	mock.balancers = append(mock.balancers, lb)
	return &lb, nil
}
