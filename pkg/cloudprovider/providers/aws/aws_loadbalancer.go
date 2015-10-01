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

package aws_cloud

import (
	"fmt"
	"strconv"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/util/sets"
)

func (s *AWSCloud) ensureLoadBalancer(name string, listeners []*elb.Listener, subnetIDs []string, securityGroupIDs []string) (*elb.LoadBalancerDescription, error) {
	loadBalancer, err := s.describeLoadBalancer(name)
	if err != nil {
		return nil, err
	}

	dirty := false

	if loadBalancer == nil {
		createRequest := &elb.CreateLoadBalancerInput{}
		createRequest.LoadBalancerName = aws.String(name)

		createRequest.Listeners = listeners

		// We are supposed to specify one subnet per AZ.
		// TODO: What happens if we have more than one subnet per AZ?
		createRequest.Subnets = stringPointerArray(subnetIDs)

		createRequest.SecurityGroups = stringPointerArray(securityGroupIDs)

		glog.Info("Creating load balancer with name: ", name)
		_, err := s.elb.CreateLoadBalancer(createRequest)
		if err != nil {
			return nil, err
		}
		dirty = true
	} else {
		{
			// Sync subnets
			expected := sets.NewString(subnetIDs...)
			actual := stringSetFromPointers(loadBalancer.Subnets)

			additions := expected.Difference(actual)
			removals := actual.Difference(expected)

			if removals.Len() != 0 {
				request := &elb.DetachLoadBalancerFromSubnetsInput{}
				request.LoadBalancerName = aws.String(name)
				request.Subnets = stringSetToPointers(removals)
				glog.V(2).Info("Detaching load balancer from removed subnets")
				_, err := s.elb.DetachLoadBalancerFromSubnets(request)
				if err != nil {
					return nil, fmt.Errorf("error detaching AWS loadbalancer from subnets: %v", err)
				}
				dirty = true
			}

			if additions.Len() != 0 {
				request := &elb.AttachLoadBalancerToSubnetsInput{}
				request.LoadBalancerName = aws.String(name)
				request.Subnets = stringSetToPointers(additions)
				glog.V(2).Info("Attaching load balancer to added subnets")
				_, err := s.elb.AttachLoadBalancerToSubnets(request)
				if err != nil {
					return nil, fmt.Errorf("error attaching AWS loadbalancer to subnets: %v", err)
				}
				dirty = true
			}
		}

		{
			// Sync security groups
			expected := sets.NewString(securityGroupIDs...)
			actual := stringSetFromPointers(loadBalancer.SecurityGroups)

			if !expected.Equal(actual) {
				// This call just replaces the security groups, unlike e.g. subnets (!)
				request := &elb.ApplySecurityGroupsToLoadBalancerInput{}
				request.LoadBalancerName = aws.String(name)
				request.SecurityGroups = stringPointerArray(securityGroupIDs)
				glog.V(2).Info("Applying updated security groups to load balancer")
				_, err := s.elb.ApplySecurityGroupsToLoadBalancer(request)
				if err != nil {
					return nil, fmt.Errorf("error applying AWS loadbalancer security groups: %v", err)
				}
				dirty = true
			}
		}

		{
			// Sync listeners
			listenerDescriptions := loadBalancer.ListenerDescriptions

			foundSet := make(map[int]bool)
			removals := []*int64{}
			for _, listenerDescription := range listenerDescriptions {
				actual := listenerDescription.Listener
				if actual == nil {
					glog.Warning("Ignoring empty listener in AWS loadbalancer: ", name)
					continue
				}

				found := -1
				for i, expected := range listeners {
					if orEmpty(actual.Protocol) != orEmpty(expected.Protocol) {
						continue
					}
					if orEmpty(actual.InstanceProtocol) != orEmpty(expected.InstanceProtocol) {
						continue
					}
					if orZero(actual.InstancePort) != orZero(expected.InstancePort) {
						continue
					}
					if orZero(actual.LoadBalancerPort) != orZero(expected.LoadBalancerPort) {
						continue
					}
					if orEmpty(actual.SSLCertificateId) != orEmpty(expected.SSLCertificateId) {
						continue
					}
					found = i
				}
				if found != -1 {
					foundSet[found] = true
				} else {
					removals = append(removals, actual.LoadBalancerPort)
				}
			}

			additions := []*elb.Listener{}
			for i := range listeners {
				if foundSet[i] {
					continue
				}
				additions = append(additions, listeners[i])
			}

			if len(removals) != 0 {
				request := &elb.DeleteLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(name)
				request.LoadBalancerPorts = removals
				glog.V(2).Info("Deleting removed load balancer listeners")
				_, err := s.elb.DeleteLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error deleting AWS loadbalancer listeners: %v", err)
				}
				dirty = true
			}

			if len(additions) != 0 {
				request := &elb.CreateLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(name)
				request.Listeners = additions
				glog.V(2).Info("Creating added load balancer listeners")
				_, err := s.elb.CreateLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error creating AWS loadbalancer listeners: %v", err)
				}
				dirty = true
			}
		}
	}

	if dirty {
		loadBalancer, err = s.describeLoadBalancer(name)
		if err != nil {
			glog.Warning("Unable to retrieve load balancer after creation/update")
			return nil, err
		}
	}

	return loadBalancer, nil
}

// Makes sure that the health check for an ELB matches the configured listeners
func (s *AWSCloud) ensureLoadBalancerHealthCheck(loadBalancer *elb.LoadBalancerDescription, listeners []*elb.Listener) error {
	actual := loadBalancer.HealthCheck

	// Default AWS settings
	expectedHealthyThreshold := int64(10)
	expectedUnhealthyThreshold := int64(2)
	expectedTimeout := int64(5)
	expectedInterval := int64(30)

	// We only a TCP health-check on the first port
	expectedTarget := ""
	for _, listener := range listeners {
		if listener.InstancePort == nil {
			continue
		}
		expectedTarget = "TCP:" + strconv.FormatInt(*listener.InstancePort, 10)
		break
	}

	if expectedTarget == "" {
		return fmt.Errorf("unable to determine health check port (no valid listeners)")
	}

	if expectedTarget == orEmpty(actual.Target) &&
		expectedHealthyThreshold == orZero(actual.HealthyThreshold) &&
		expectedUnhealthyThreshold == orZero(actual.UnhealthyThreshold) &&
		expectedTimeout == orZero(actual.Timeout) &&
		expectedInterval == orZero(actual.Interval) {
		return nil
	}

	glog.V(2).Info("Updating load-balancer health-check")

	healthCheck := &elb.HealthCheck{}
	healthCheck.HealthyThreshold = &expectedHealthyThreshold
	healthCheck.UnhealthyThreshold = &expectedUnhealthyThreshold
	healthCheck.Timeout = &expectedTimeout
	healthCheck.Interval = &expectedInterval
	healthCheck.Target = &expectedTarget

	request := &elb.ConfigureHealthCheckInput{}
	request.HealthCheck = healthCheck
	request.LoadBalancerName = loadBalancer.LoadBalancerName

	_, err := s.elb.ConfigureHealthCheck(request)
	if err != nil {
		return fmt.Errorf("error configuring load-balancer health-check: %v", err)
	}

	return nil
}

// Makes sure that exactly the specified hosts are registered as instances with the load balancer
func (s *AWSCloud) ensureLoadBalancerInstances(loadBalancerName string, lbInstances []*elb.Instance, instances []*ec2.Instance) error {
	expected := sets.NewString()
	for _, instance := range instances {
		expected.Insert(orEmpty(instance.InstanceId))
	}

	actual := sets.NewString()
	for _, lbInstance := range lbInstances {
		actual.Insert(orEmpty(lbInstance.InstanceId))
	}

	additions := expected.Difference(actual)
	removals := actual.Difference(expected)

	addInstances := []*elb.Instance{}
	for _, instanceId := range additions.List() {
		addInstance := &elb.Instance{}
		addInstance.InstanceId = aws.String(instanceId)
		addInstances = append(addInstances, addInstance)
	}

	removeInstances := []*elb.Instance{}
	for _, instanceId := range removals.List() {
		removeInstance := &elb.Instance{}
		removeInstance.InstanceId = aws.String(instanceId)
		removeInstances = append(removeInstances, removeInstance)
	}

	if len(addInstances) > 0 {
		registerRequest := &elb.RegisterInstancesWithLoadBalancerInput{}
		registerRequest.Instances = addInstances
		registerRequest.LoadBalancerName = aws.String(loadBalancerName)
		_, err := s.elb.RegisterInstancesWithLoadBalancer(registerRequest)
		if err != nil {
			return err
		}
		glog.V(1).Infof("Instances added to load-balancer %s", loadBalancerName)
	}

	if len(removeInstances) > 0 {
		deregisterRequest := &elb.DeregisterInstancesFromLoadBalancerInput{}
		deregisterRequest.Instances = removeInstances
		deregisterRequest.LoadBalancerName = aws.String(loadBalancerName)
		_, err := s.elb.DeregisterInstancesFromLoadBalancer(deregisterRequest)
		if err != nil {
			return err
		}
		glog.V(1).Infof("Instances removed from load-balancer %s", loadBalancerName)
	}

	return nil
}
