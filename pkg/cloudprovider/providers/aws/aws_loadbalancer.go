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

package aws

import (
	"fmt"
	"strconv"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
)

const ProxyProtocolPolicyName = "k8s-proxyprotocol-enabled"

func (c *Cloud) ensureLoadBalancer(namespacedName types.NamespacedName, loadBalancerName string, listeners []*elb.Listener, subnetIDs []string, securityGroupIDs []string, internalELB, proxyProtocol bool) (*elb.LoadBalancerDescription, error) {
	loadBalancer, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return nil, err
	}

	dirty := false

	if loadBalancer == nil {
		createRequest := &elb.CreateLoadBalancerInput{}
		createRequest.LoadBalancerName = aws.String(loadBalancerName)

		createRequest.Listeners = listeners

		if internalELB {
			createRequest.Scheme = aws.String("internal")
		}

		// We are supposed to specify one subnet per AZ.
		// TODO: What happens if we have more than one subnet per AZ?
		createRequest.Subnets = stringPointerArray(subnetIDs)

		createRequest.SecurityGroups = stringPointerArray(securityGroupIDs)

		createRequest.Tags = []*elb.Tag{
			{Key: aws.String(TagNameKubernetesCluster), Value: aws.String(c.getClusterName())},
			{Key: aws.String(TagNameKubernetesService), Value: aws.String(namespacedName.String())},
		}

		glog.Infof("Creating load balancer for %v with name: %s", namespacedName, loadBalancerName)
		_, err := c.elb.CreateLoadBalancer(createRequest)
		if err != nil {
			return nil, err
		}

		if proxyProtocol {
			err = c.createProxyProtocolPolicy(loadBalancerName)
			if err != nil {
				return nil, err
			}

			for _, listener := range listeners {
				glog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to true", *listener.InstancePort)
				err := c.setBackendPolicies(loadBalancerName, *listener.InstancePort, []*string{aws.String(ProxyProtocolPolicyName)})
				if err != nil {
					return nil, err
				}
			}
		}

		dirty = true
	} else {
		// TODO: Sync internal vs non-internal

		{
			// Sync subnets
			expected := sets.NewString(subnetIDs...)
			actual := stringSetFromPointers(loadBalancer.Subnets)

			additions := expected.Difference(actual)
			removals := actual.Difference(expected)

			if removals.Len() != 0 {
				request := &elb.DetachLoadBalancerFromSubnetsInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Subnets = stringSetToPointers(removals)
				glog.V(2).Info("Detaching load balancer from removed subnets")
				_, err := c.elb.DetachLoadBalancerFromSubnets(request)
				if err != nil {
					return nil, fmt.Errorf("error detaching AWS loadbalancer from subnets: %v", err)
				}
				dirty = true
			}

			if additions.Len() != 0 {
				request := &elb.AttachLoadBalancerToSubnetsInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Subnets = stringSetToPointers(additions)
				glog.V(2).Info("Attaching load balancer to added subnets")
				_, err := c.elb.AttachLoadBalancerToSubnets(request)
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
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.SecurityGroups = stringPointerArray(securityGroupIDs)
				glog.V(2).Info("Applying updated security groups to load balancer")
				_, err := c.elb.ApplySecurityGroupsToLoadBalancer(request)
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
					glog.Warning("Ignoring empty listener in AWS loadbalancer: ", loadBalancerName)
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
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.LoadBalancerPorts = removals
				glog.V(2).Info("Deleting removed load balancer listeners")
				_, err := c.elb.DeleteLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error deleting AWS loadbalancer listeners: %v", err)
				}
				dirty = true
			}

			if len(additions) != 0 {
				request := &elb.CreateLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Listeners = additions
				glog.V(2).Info("Creating added load balancer listeners")
				_, err := c.elb.CreateLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error creating AWS loadbalancer listeners: %v", err)
				}
				dirty = true
			}
		}

		{
			// Sync proxy protocol state for new and existing listeners

			proxyPolicies := make([]*string, 0)
			if proxyProtocol {
				// Ensure the backend policy exists

				// NOTE The documentation for the AWS API indicates we could get an HTTP 400
				// back if a policy of the same name already exists. However, the aws-sdk does not
				// seem to return an error to us in these cases. Therefore this will issue an API
				// request every time.
				err := c.createProxyProtocolPolicy(loadBalancerName)
				if err != nil {
					return nil, err
				}

				proxyPolicies = append(proxyPolicies, aws.String(ProxyProtocolPolicyName))
			}

			foundBackends := make(map[int64]bool)
			proxyProtocolBackends := make(map[int64]bool)
			for _, backendListener := range loadBalancer.BackendServerDescriptions {
				foundBackends[*backendListener.InstancePort] = false
				proxyProtocolBackends[*backendListener.InstancePort] = proxyProtocolEnabled(backendListener)
			}

			for _, listener := range listeners {
				setPolicy := false
				instancePort := *listener.InstancePort

				if currentState, ok := proxyProtocolBackends[instancePort]; !ok {
					// This is a new ELB backend so we only need to worry about
					// potentially adding a policy and not removing an
					// existing one
					setPolicy = proxyProtocol
				} else {
					foundBackends[instancePort] = true
					// This is an existing ELB backend so we need to determine
					// if the state changed
					setPolicy = (currentState != proxyProtocol)
				}

				if setPolicy {
					glog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to %t", instancePort, proxyProtocol)
					err := c.setBackendPolicies(loadBalancerName, instancePort, proxyPolicies)
					if err != nil {
						return nil, err
					}
					dirty = true
				}
			}

			// We now need to figure out if any backend policies need removed
			// because these old policies will stick around even if there is no
			// corresponding listener anymore
			for instancePort, found := range foundBackends {
				if !found {
					glog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to false", instancePort)
					err := c.setBackendPolicies(loadBalancerName, instancePort, []*string{})
					if err != nil {
						return nil, err
					}
					dirty = true
				}
			}
		}
	}

	if dirty {
		loadBalancer, err = c.describeLoadBalancer(loadBalancerName)
		if err != nil {
			glog.Warning("Unable to retrieve load balancer after creation/update")
			return nil, err
		}
	}

	return loadBalancer, nil
}

// Makes sure that the health check for an ELB matches the configured listeners
func (c *Cloud) ensureLoadBalancerHealthCheck(loadBalancer *elb.LoadBalancerDescription, listeners []*elb.Listener) error {
	actual := loadBalancer.HealthCheck

	// Default AWS settings
	expectedHealthyThreshold := int64(2)
	expectedUnhealthyThreshold := int64(6)
	expectedTimeout := int64(5)
	expectedInterval := int64(10)

	// We only configure a TCP health-check on the first port
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

	_, err := c.elb.ConfigureHealthCheck(request)
	if err != nil {
		return fmt.Errorf("error configuring load-balancer health-check: %v", err)
	}

	return nil
}

// Makes sure that exactly the specified hosts are registered as instances with the load balancer
func (c *Cloud) ensureLoadBalancerInstances(loadBalancerName string, lbInstances []*elb.Instance, instances []*ec2.Instance) error {
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
		_, err := c.elb.RegisterInstancesWithLoadBalancer(registerRequest)
		if err != nil {
			return err
		}
		glog.V(1).Infof("Instances added to load-balancer %s", loadBalancerName)
	}

	if len(removeInstances) > 0 {
		deregisterRequest := &elb.DeregisterInstancesFromLoadBalancerInput{}
		deregisterRequest.Instances = removeInstances
		deregisterRequest.LoadBalancerName = aws.String(loadBalancerName)
		_, err := c.elb.DeregisterInstancesFromLoadBalancer(deregisterRequest)
		if err != nil {
			return err
		}
		glog.V(1).Infof("Instances removed from load-balancer %s", loadBalancerName)
	}

	return nil
}

func (c *Cloud) createProxyProtocolPolicy(loadBalancerName string) error {
	request := &elb.CreateLoadBalancerPolicyInput{
		LoadBalancerName: aws.String(loadBalancerName),
		PolicyName:       aws.String(ProxyProtocolPolicyName),
		PolicyTypeName:   aws.String("ProxyProtocolPolicyType"),
		PolicyAttributes: []*elb.PolicyAttribute{
			{
				AttributeName:  aws.String("ProxyProtocol"),
				AttributeValue: aws.String("true"),
			},
		},
	}
	glog.V(2).Info("Creating proxy protocol policy on load balancer")
	_, err := c.elb.CreateLoadBalancerPolicy(request)
	if err != nil {
		return fmt.Errorf("error creating proxy protocol policy on load balancer: %v", err)
	}

	return nil
}

func (c *Cloud) setBackendPolicies(loadBalancerName string, instancePort int64, policies []*string) error {
	request := &elb.SetLoadBalancerPoliciesForBackendServerInput{
		InstancePort:     aws.Int64(instancePort),
		LoadBalancerName: aws.String(loadBalancerName),
		PolicyNames:      policies,
	}
	if len(policies) > 0 {
		glog.V(2).Infof("Adding AWS loadbalancer backend policies on node port %d", instancePort)
	} else {
		glog.V(2).Infof("Removing AWS loadbalancer backend policies on node port %d", instancePort)
	}
	_, err := c.elb.SetLoadBalancerPoliciesForBackendServer(request)
	if err != nil {
		return fmt.Errorf("error adjusting AWS loadbalancer backend policies: %v", err)
	}

	return nil
}

func proxyProtocolEnabled(backend *elb.BackendServerDescription) bool {
	for _, policy := range backend.PolicyNames {
		if aws.StringValue(policy) == ProxyProtocolPolicyName {
			return true
		}
	}

	return false
}
