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
	"crypto/sha1"
	"fmt"
	"reflect"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/aws/aws-sdk-go/service/elbv2"
	"k8s.io/klog"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
)

const (
	// ProxyProtocolPolicyName is the tag named used for the proxy protocol
	// policy
	ProxyProtocolPolicyName = "k8s-proxyprotocol-enabled"

	// SSLNegotiationPolicyNameFormat is a format string used for the SSL
	// negotiation policy tag name
	SSLNegotiationPolicyNameFormat = "k8s-SSLNegotiationPolicy-%s"
)

var (
	// Defaults for ELB Healthcheck
	defaultHCHealthyThreshold   = int64(2)
	defaultHCUnhealthyThreshold = int64(6)
	defaultHCTimeout            = int64(5)
	defaultHCInterval           = int64(10)
)

func isNLB(annotations map[string]string) bool {
	if annotations[ServiceAnnotationLoadBalancerType] == "nlb" {
		return true
	}
	return false
}

type nlbPortMapping struct {
	FrontendPort int64
	TrafficPort  int64
	ClientCIDR   string

	HealthCheckPort     int64
	HealthCheckPath     string
	HealthCheckProtocol string
}

// getLoadBalancerAdditionalTags converts the comma separated list of key-value
// pairs in the ServiceAnnotationLoadBalancerAdditionalTags annotation and returns
// it as a map.
func getLoadBalancerAdditionalTags(annotations map[string]string) map[string]string {
	additionalTags := make(map[string]string)
	if additionalTagsList, ok := annotations[ServiceAnnotationLoadBalancerAdditionalTags]; ok {
		additionalTagsList = strings.TrimSpace(additionalTagsList)

		// Break up list of "Key1=Val,Key2=Val2"
		tagList := strings.Split(additionalTagsList, ",")

		// Break up "Key=Val"
		for _, tagSet := range tagList {
			tag := strings.Split(strings.TrimSpace(tagSet), "=")

			// Accept "Key=val" or "Key=" or just "Key"
			if len(tag) >= 2 && len(tag[0]) != 0 {
				// There is a key and a value, so save it
				additionalTags[tag[0]] = tag[1]
			} else if len(tag) == 1 && len(tag[0]) != 0 {
				// Just "Key"
				additionalTags[tag[0]] = ""
			}
		}
	}

	return additionalTags
}

// ensureLoadBalancerv2 ensures a v2 load balancer is created
func (c *Cloud) ensureLoadBalancerv2(namespacedName types.NamespacedName, loadBalancerName string, mappings []nlbPortMapping, instanceIDs, subnetIDs []string, internalELB bool, annotations map[string]string) (*elbv2.LoadBalancer, error) {
	loadBalancer, err := c.describeLoadBalancerv2(loadBalancerName)
	if err != nil {
		return nil, err
	}

	dirty := false

	// Get additional tags set by the user
	tags := getLoadBalancerAdditionalTags(annotations)
	// Add default tags
	tags[TagNameKubernetesService] = namespacedName.String()
	tags = c.tagging.buildTags(ResourceLifecycleOwned, tags)

	if loadBalancer == nil {
		// Create the LB
		createRequest := &elbv2.CreateLoadBalancerInput{
			Type: aws.String(elbv2.LoadBalancerTypeEnumNetwork),
			Name: aws.String(loadBalancerName),
		}
		if internalELB {
			createRequest.Scheme = aws.String("internal")
		}

		// We are supposed to specify one subnet per AZ.
		// TODO: What happens if we have more than one subnet per AZ?
		createRequest.SubnetMappings = createSubnetMappings(subnetIDs)

		for k, v := range tags {
			createRequest.Tags = append(createRequest.Tags, &elbv2.Tag{
				Key: aws.String(k), Value: aws.String(v),
			})
		}

		klog.Infof("Creating load balancer for %v with name: %s", namespacedName, loadBalancerName)
		createResponse, err := c.elbv2.CreateLoadBalancer(createRequest)
		if err != nil {
			return nil, fmt.Errorf("Error creating load balancer: %q", err)
		}

		loadBalancer = createResponse.LoadBalancers[0]

		// Create Target Groups
		resourceArns := make([]*string, 0, len(mappings))

		for i := range mappings {
			// It is easier to keep track of updates by having possibly
			// duplicate target groups where the backend port is the same
			_, targetGroupArn, err := c.createListenerV2(createResponse.LoadBalancers[0].LoadBalancerArn, mappings[i], namespacedName, instanceIDs, *createResponse.LoadBalancers[0].VpcId)
			if err != nil {
				return nil, fmt.Errorf("Error creating listener: %q", err)
			}
			resourceArns = append(resourceArns, targetGroupArn)

		}

		// Add tags to targets
		targetGroupTags := make([]*elbv2.Tag, 0, len(tags))

		for k, v := range tags {
			targetGroupTags = append(targetGroupTags, &elbv2.Tag{
				Key: aws.String(k), Value: aws.String(v),
			})
		}
		if len(resourceArns) > 0 && len(targetGroupTags) > 0 {
			// elbv2.AddTags doesn't allow to tag multiple resources at once
			for _, arn := range resourceArns {
				_, err = c.elbv2.AddTags(&elbv2.AddTagsInput{
					ResourceArns: []*string{arn},
					Tags:         targetGroupTags,
				})
				if err != nil {
					return nil, fmt.Errorf("Error adding tags after creating Load Balancer: %q", err)
				}
			}
		}
	} else {
		// TODO: Sync internal vs non-internal

		// sync mappings
		{
			listenerDescriptions, err := c.elbv2.DescribeListeners(
				&elbv2.DescribeListenersInput{
					LoadBalancerArn: loadBalancer.LoadBalancerArn,
				},
			)
			if err != nil {
				return nil, fmt.Errorf("Error describing listeners: %q", err)
			}

			// actual maps FrontendPort to an elbv2.Listener
			actual := map[int64]*elbv2.Listener{}
			for _, listener := range listenerDescriptions.Listeners {
				actual[*listener.Port] = listener
			}

			actualTargetGroups, err := c.elbv2.DescribeTargetGroups(
				&elbv2.DescribeTargetGroupsInput{
					LoadBalancerArn: loadBalancer.LoadBalancerArn,
				},
			)
			if err != nil {
				return nil, fmt.Errorf("Error listing target groups: %q", err)
			}

			nodePortTargetGroup := map[int64]*elbv2.TargetGroup{}
			for _, targetGroup := range actualTargetGroups.TargetGroups {
				nodePortTargetGroup[*targetGroup.Port] = targetGroup
			}

			// Create Target Groups
			addTagsInput := &elbv2.AddTagsInput{
				ResourceArns: []*string{},
				Tags:         []*elbv2.Tag{},
			}

			// Handle additions/modifications
			for _, mapping := range mappings {
				frontendPort := mapping.FrontendPort
				nodePort := mapping.TrafficPort

				// modifications
				if listener, ok := actual[frontendPort]; ok {
					// nodePort must have changed, we'll need to delete old TG
					// and recreate
					if targetGroup, ok := nodePortTargetGroup[nodePort]; !ok {
						// Create new Target group
						targetName := createTargetName(namespacedName, frontendPort, nodePort)
						targetGroup, err = c.ensureTargetGroup(
							nil,
							mapping,
							instanceIDs,
							targetName,
							*loadBalancer.VpcId,
						)
						if err != nil {
							return nil, err
						}

						// Associate new target group to LB
						_, err := c.elbv2.ModifyListener(&elbv2.ModifyListenerInput{
							ListenerArn: listener.ListenerArn,
							Port:        aws.Int64(frontendPort),
							Protocol:    aws.String("TCP"),
							DefaultActions: []*elbv2.Action{{
								TargetGroupArn: targetGroup.TargetGroupArn,
								Type:           aws.String("forward"),
							}},
						})
						if err != nil {
							return nil, fmt.Errorf("Error updating load balancer listener: %q", err)
						}

						// Delete old target group
						_, err = c.elbv2.DeleteTargetGroup(&elbv2.DeleteTargetGroupInput{
							TargetGroupArn: listener.DefaultActions[0].TargetGroupArn,
						})
						if err != nil {
							return nil, fmt.Errorf("Error deleting old target group: %q", err)
						}

					} else {
						// Run ensureTargetGroup to make sure instances in service are up-to-date
						targetName := createTargetName(namespacedName, frontendPort, nodePort)
						_, err = c.ensureTargetGroup(
							targetGroup,
							mapping,
							instanceIDs,
							targetName,
							*loadBalancer.VpcId,
						)
						if err != nil {
							return nil, err
						}
					}
					dirty = true
					continue
				}

				// Additions
				_, targetGroupArn, err := c.createListenerV2(loadBalancer.LoadBalancerArn, mapping, namespacedName, instanceIDs, *loadBalancer.VpcId)
				if err != nil {
					return nil, err
				}
				addTagsInput.ResourceArns = append(addTagsInput.ResourceArns, targetGroupArn)
				dirty = true
			}

			frontEndPorts := map[int64]bool{}
			for i := range mappings {
				frontEndPorts[mappings[i].FrontendPort] = true
			}

			// handle deletions
			for port, listener := range actual {
				if _, ok := frontEndPorts[port]; !ok {
					err := c.deleteListenerV2(listener)
					if err != nil {
						return nil, err
					}
					dirty = true
				}
			}

			// Add tags to new targets
			for k, v := range tags {
				addTagsInput.Tags = append(addTagsInput.Tags, &elbv2.Tag{
					Key: aws.String(k), Value: aws.String(v),
				})
			}
			if len(addTagsInput.ResourceArns) > 0 && len(addTagsInput.Tags) > 0 {
				_, err = c.elbv2.AddTags(addTagsInput)
				if err != nil {
					return nil, fmt.Errorf("Error adding tags after modifying load balancer targets: %q", err)
				}
			}
		}

		desiredLoadBalancerAttributes := map[string]string{}
		// Default values to ensured a remove annotation reverts back to the default
		desiredLoadBalancerAttributes["load_balancing.cross_zone.enabled"] = "false"

		// Determine if cross zone load balancing enabled/disabled has been specified
		crossZoneLoadBalancingEnabledAnnotation := annotations[ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled]
		if crossZoneLoadBalancingEnabledAnnotation != "" {
			crossZoneEnabled, err := strconv.ParseBool(crossZoneLoadBalancingEnabledAnnotation)
			if err != nil {
				return nil, fmt.Errorf("error parsing service annotation: %s=%s",
					ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled,
					crossZoneLoadBalancingEnabledAnnotation,
				)
			}

			if crossZoneEnabled {
				desiredLoadBalancerAttributes["load_balancing.cross_zone.enabled"] = "true"
			}
		}

		// Whether the ELB was new or existing, sync attributes regardless. This accounts for things
		// that cannot be specified at the time of creation and can only be modified after the fact,
		// e.g. idle connection timeout.
		describeAttributesRequest := &elbv2.DescribeLoadBalancerAttributesInput{
			LoadBalancerArn: loadBalancer.LoadBalancerArn,
		}
		describeAttributesOutput, err := c.elbv2.DescribeLoadBalancerAttributes(describeAttributesRequest)
		if err != nil {
			return nil, fmt.Errorf("Unable to retrieve load balancer attributes during attribute sync: %q", err)
		}

		changedAttributes := []*elbv2.LoadBalancerAttribute{}

		// Identify to be changed attributes
		for _, foundAttribute := range describeAttributesOutput.Attributes {
			if targetValue, ok := desiredLoadBalancerAttributes[*foundAttribute.Key]; ok {
				if targetValue != *foundAttribute.Value {
					changedAttributes = append(changedAttributes, &elbv2.LoadBalancerAttribute{
						Key:   foundAttribute.Key,
						Value: aws.String(targetValue),
					})
				}
			}
		}

		// Update attributes requiring changes
		if len(changedAttributes) > 0 {
			klog.V(2).Infof("Updating load-balancer attributes for %q", loadBalancerName)

			_, err = c.elbv2.ModifyLoadBalancerAttributes(&elbv2.ModifyLoadBalancerAttributesInput{
				LoadBalancerArn: loadBalancer.LoadBalancerArn,
				Attributes:      changedAttributes,
			})
			if err != nil {
				return nil, fmt.Errorf("Unable to update load balancer attributes during attribute sync: %q", err)
			}
		}

		// Subnets cannot be modified on NLBs
		if dirty {
			loadBalancers, err := c.elbv2.DescribeLoadBalancers(
				&elbv2.DescribeLoadBalancersInput{
					LoadBalancerArns: []*string{
						loadBalancer.LoadBalancerArn,
					},
				},
			)
			if err != nil {
				return nil, fmt.Errorf("Error retrieving load balancer after update: %q", err)
			}
			loadBalancer = loadBalancers.LoadBalancers[0]
		}
	}
	return loadBalancer, nil
}

// create a valid target group name - ensure name is not over 32 characters
func createTargetName(namespacedName types.NamespacedName, frontendPort, nodePort int64) string {
	sha := fmt.Sprintf("%x", sha1.Sum([]byte(namespacedName.String())))[:13]
	return fmt.Sprintf("k8s-tg-%s-%d-%d", sha, frontendPort, nodePort)
}

func (c *Cloud) createListenerV2(loadBalancerArn *string, mapping nlbPortMapping, namespacedName types.NamespacedName, instanceIDs []string, vpcID string) (listener *elbv2.Listener, targetGroupArn *string, err error) {
	targetName := createTargetName(namespacedName, mapping.FrontendPort, mapping.TrafficPort)

	klog.Infof("Creating load balancer target group for %v with name: %s", namespacedName, targetName)
	target, err := c.ensureTargetGroup(
		nil,
		mapping,
		instanceIDs,
		targetName,
		vpcID,
	)
	if err != nil {
		return nil, aws.String(""), err
	}

	createListernerInput := &elbv2.CreateListenerInput{
		LoadBalancerArn: loadBalancerArn,
		Port:            aws.Int64(mapping.FrontendPort),
		Protocol:        aws.String("TCP"),
		DefaultActions: []*elbv2.Action{{
			TargetGroupArn: target.TargetGroupArn,
			Type:           aws.String(elbv2.ActionTypeEnumForward),
		}},
	}
	klog.Infof("Creating load balancer listener for %v", namespacedName)
	createListenerOutput, err := c.elbv2.CreateListener(createListernerInput)
	if err != nil {
		return nil, aws.String(""), fmt.Errorf("Error creating load balancer listener: %q", err)
	}
	return createListenerOutput.Listeners[0], target.TargetGroupArn, nil
}

// cleans up listener and corresponding target group
func (c *Cloud) deleteListenerV2(listener *elbv2.Listener) error {
	_, err := c.elbv2.DeleteListener(&elbv2.DeleteListenerInput{ListenerArn: listener.ListenerArn})
	if err != nil {
		return fmt.Errorf("Error deleting load balancer listener: %q", err)
	}
	_, err = c.elbv2.DeleteTargetGroup(&elbv2.DeleteTargetGroupInput{TargetGroupArn: listener.DefaultActions[0].TargetGroupArn})
	if err != nil {
		return fmt.Errorf("Error deleting load balancer target group: %q", err)
	}
	return nil
}

// ensureTargetGroup creates a target group with a set of instances
func (c *Cloud) ensureTargetGroup(targetGroup *elbv2.TargetGroup, mapping nlbPortMapping, instances []string, name string, vpcID string) (*elbv2.TargetGroup, error) {
	dirty := false
	if targetGroup == nil {

		input := &elbv2.CreateTargetGroupInput{
			VpcId:                      aws.String(vpcID),
			Name:                       aws.String(name),
			Port:                       aws.Int64(mapping.TrafficPort),
			Protocol:                   aws.String("TCP"),
			TargetType:                 aws.String("instance"),
			HealthCheckIntervalSeconds: aws.Int64(30),
			HealthCheckPort:            aws.String("traffic-port"),
			HealthCheckProtocol:        aws.String("TCP"),
			HealthyThresholdCount:      aws.Int64(3),
			UnhealthyThresholdCount:    aws.Int64(3),
		}

		input.HealthCheckProtocol = aws.String(mapping.HealthCheckProtocol)
		if mapping.HealthCheckProtocol != elbv2.ProtocolEnumTcp {
			input.HealthCheckPath = aws.String(mapping.HealthCheckPath)
		}

		// Account for externalTrafficPolicy = "Local"
		if mapping.HealthCheckPort != mapping.TrafficPort {
			input.HealthCheckPort = aws.String(strconv.Itoa(int(mapping.HealthCheckPort)))
		}

		result, err := c.elbv2.CreateTargetGroup(input)
		if err != nil {
			return nil, fmt.Errorf("Error creating load balancer target group: %q", err)
		}
		if len(result.TargetGroups) != 1 {
			return nil, fmt.Errorf("Expected only one target group on CreateTargetGroup, got %d groups", len(result.TargetGroups))
		}

		registerInput := &elbv2.RegisterTargetsInput{
			TargetGroupArn: result.TargetGroups[0].TargetGroupArn,
			Targets:        []*elbv2.TargetDescription{},
		}
		for _, instanceID := range instances {
			registerInput.Targets = append(registerInput.Targets, &elbv2.TargetDescription{
				Id:   aws.String(string(instanceID)),
				Port: aws.Int64(mapping.TrafficPort),
			})
		}

		_, err = c.elbv2.RegisterTargets(registerInput)
		if err != nil {
			return nil, fmt.Errorf("Error registering targets for load balancer: %q", err)
		}

		return result.TargetGroups[0], nil
	}

	// handle instances in service
	{
		healthResponse, err := c.elbv2.DescribeTargetHealth(&elbv2.DescribeTargetHealthInput{TargetGroupArn: targetGroup.TargetGroupArn})
		if err != nil {
			return nil, fmt.Errorf("Error describing target group health: %q", err)
		}
		actualIDs := []string{}
		for _, healthDescription := range healthResponse.TargetHealthDescriptions {
			if healthDescription.TargetHealth.Reason != nil {
				switch aws.StringValue(healthDescription.TargetHealth.Reason) {
				case elbv2.TargetHealthReasonEnumTargetDeregistrationInProgress:
					// We don't need to count this instance in service if it is
					// on its way out
				default:
					actualIDs = append(actualIDs, *healthDescription.Target.Id)
				}
			}
		}

		actual := sets.NewString(actualIDs...)
		expected := sets.NewString(instances...)

		additions := expected.Difference(actual)
		removals := actual.Difference(expected)

		if len(additions) > 0 {
			registerInput := &elbv2.RegisterTargetsInput{
				TargetGroupArn: targetGroup.TargetGroupArn,
				Targets:        []*elbv2.TargetDescription{},
			}
			for instanceID := range additions {
				registerInput.Targets = append(registerInput.Targets, &elbv2.TargetDescription{
					Id:   aws.String(instanceID),
					Port: aws.Int64(mapping.TrafficPort),
				})
			}
			_, err := c.elbv2.RegisterTargets(registerInput)
			if err != nil {
				return nil, fmt.Errorf("Error registering new targets in target group: %q", err)
			}
			dirty = true
		}

		if len(removals) > 0 {
			deregisterInput := &elbv2.DeregisterTargetsInput{
				TargetGroupArn: targetGroup.TargetGroupArn,
				Targets:        []*elbv2.TargetDescription{},
			}
			for instanceID := range removals {
				deregisterInput.Targets = append(deregisterInput.Targets, &elbv2.TargetDescription{
					Id:   aws.String(instanceID),
					Port: aws.Int64(mapping.TrafficPort),
				})
			}
			_, err := c.elbv2.DeregisterTargets(deregisterInput)
			if err != nil {
				return nil, fmt.Errorf("Error trying to deregister targets in target group: %q", err)
			}
			dirty = true
		}
	}

	// ensure the health check is correct
	{
		dirtyHealthCheck := false

		input := &elbv2.ModifyTargetGroupInput{
			TargetGroupArn: targetGroup.TargetGroupArn,
		}

		if aws.StringValue(targetGroup.HealthCheckProtocol) != mapping.HealthCheckProtocol {
			input.HealthCheckProtocol = aws.String(mapping.HealthCheckProtocol)
			dirtyHealthCheck = true
		}
		if aws.StringValue(targetGroup.HealthCheckPort) != strconv.Itoa(int(mapping.HealthCheckPort)) {
			input.HealthCheckPort = aws.String(strconv.Itoa(int(mapping.HealthCheckPort)))
			dirtyHealthCheck = true
		}
		if mapping.HealthCheckPath != "" && mapping.HealthCheckProtocol != elbv2.ProtocolEnumTcp {
			input.HealthCheckPath = aws.String(mapping.HealthCheckPath)
			dirtyHealthCheck = true
		}

		if dirtyHealthCheck {
			_, err := c.elbv2.ModifyTargetGroup(input)
			if err != nil {
				return nil, fmt.Errorf("Error modifying target group health check: %q", err)
			}

			dirty = true
		}
	}

	if dirty {
		result, err := c.elbv2.DescribeTargetGroups(&elbv2.DescribeTargetGroupsInput{
			Names: []*string{aws.String(name)},
		})
		if err != nil {
			return nil, fmt.Errorf("Error retrieving target group after creation/update: %q", err)
		}
		targetGroup = result.TargetGroups[0]
	}

	return targetGroup, nil
}

func portsForNLB(lbName string, sg *ec2.SecurityGroup, clientTraffic bool) sets.Int64 {
	response := sets.NewInt64()
	var annotation string
	if clientTraffic {
		annotation = fmt.Sprintf("%s=%s", NLBClientRuleDescription, lbName)
	} else {
		annotation = fmt.Sprintf("%s=%s", NLBHealthCheckRuleDescription, lbName)
	}

	for i := range sg.IpPermissions {
		for j := range sg.IpPermissions[i].IpRanges {
			description := aws.StringValue(sg.IpPermissions[i].IpRanges[j].Description)
			if description == annotation {
				// TODO  should probably check FromPort == ToPort
				response.Insert(aws.Int64Value(sg.IpPermissions[i].FromPort))
			}
		}
	}
	return response
}

// filterForIPRangeDescription filters in security groups that have IpRange Descriptions that match a loadBalancerName
func filterForIPRangeDescription(securityGroups []*ec2.SecurityGroup, lbName string) []*ec2.SecurityGroup {
	response := []*ec2.SecurityGroup{}
	clientRule := fmt.Sprintf("%s=%s", NLBClientRuleDescription, lbName)
	healthRule := fmt.Sprintf("%s=%s", NLBHealthCheckRuleDescription, lbName)
	alreadyAdded := sets.NewString()
	for i := range securityGroups {
		for j := range securityGroups[i].IpPermissions {
			for k := range securityGroups[i].IpPermissions[j].IpRanges {
				description := aws.StringValue(securityGroups[i].IpPermissions[j].IpRanges[k].Description)
				if description == clientRule || description == healthRule {
					sgIDString := aws.StringValue(securityGroups[i].GroupId)
					if !alreadyAdded.Has(sgIDString) {
						response = append(response, securityGroups[i])
						alreadyAdded.Insert(sgIDString)
					}
				}
			}
		}
	}
	return response
}

func (c *Cloud) getVpcCidrBlocks() ([]string, error) {
	vpcs, err := c.ec2.DescribeVpcs(&ec2.DescribeVpcsInput{
		VpcIds: []*string{aws.String(c.vpcID)},
	})
	if err != nil {
		return nil, fmt.Errorf("Error querying VPC for ELB: %q", err)
	}
	if len(vpcs.Vpcs) != 1 {
		return nil, fmt.Errorf("Error querying VPC for ELB, got %d vpcs for %s", len(vpcs.Vpcs), c.vpcID)
	}

	cidrBlocks := make([]string, 0, len(vpcs.Vpcs[0].CidrBlockAssociationSet))
	for _, cidr := range vpcs.Vpcs[0].CidrBlockAssociationSet {
		cidrBlocks = append(cidrBlocks, aws.StringValue(cidr.CidrBlock))
	}
	return cidrBlocks, nil
}

// abstraction for updating SG rules
// if clientTraffic is false, then only update HealthCheck rules
func (c *Cloud) updateInstanceSecurityGroupsForNLBTraffic(actualGroups []*ec2.SecurityGroup, desiredSgIds []string, ports []int64, lbName string, clientCidrs []string, clientTraffic bool) error {

	klog.V(8).Infof("updateInstanceSecurityGroupsForNLBTraffic: actualGroups=%v, desiredSgIds=%v, ports=%v, clientTraffic=%v", actualGroups, desiredSgIds, ports, clientTraffic)
	// Map containing the groups we want to make changes on; the ports to make
	// changes on; and whether to add or remove it. true to add, false to remove
	portChanges := map[string]map[int64]bool{}

	for _, id := range desiredSgIds {
		// consider everything an addition for now
		if _, ok := portChanges[id]; !ok {
			portChanges[id] = make(map[int64]bool)
		}
		for _, port := range ports {
			portChanges[id][port] = true
		}
	}

	// Compare to actual groups
	for _, actualGroup := range actualGroups {
		actualGroupID := aws.StringValue(actualGroup.GroupId)
		if actualGroupID == "" {
			klog.Warning("Ignoring group without ID: ", actualGroup)
			continue
		}

		addingMap, ok := portChanges[actualGroupID]
		if ok {
			desiredSet := sets.NewInt64()
			for port := range addingMap {
				desiredSet.Insert(port)
			}
			existingSet := portsForNLB(lbName, actualGroup, clientTraffic)

			// remove from portChanges ports that are already allowed
			if intersection := desiredSet.Intersection(existingSet); intersection.Len() > 0 {
				for p := range intersection {
					delete(portChanges[actualGroupID], p)
				}
			}

			// allowed ports that need to be removed
			if difference := existingSet.Difference(desiredSet); difference.Len() > 0 {
				for p := range difference {
					portChanges[actualGroupID][p] = false
				}
			}
		}
	}

	// Make changes we've planned on
	for instanceSecurityGroupID, portMap := range portChanges {
		adds := []*ec2.IpPermission{}
		removes := []*ec2.IpPermission{}
		for port, add := range portMap {
			if add {
				if clientTraffic {
					klog.V(2).Infof("Adding rule for client MTU discovery from the network load balancer (%s) to instances (%s)", clientCidrs, instanceSecurityGroupID)
					klog.V(2).Infof("Adding rule for client traffic from the network load balancer (%s) to instances (%s), port (%v)", clientCidrs, instanceSecurityGroupID, port)
				} else {
					klog.V(2).Infof("Adding rule for health check traffic from the network load balancer (%s) to instances (%s), port (%v)", clientCidrs, instanceSecurityGroupID, port)
				}
			} else {
				if clientTraffic {
					klog.V(2).Infof("Removing rule for client MTU discovery from the network load balancer (%s) to instances (%s)", clientCidrs, instanceSecurityGroupID)
					klog.V(2).Infof("Removing rule for client traffic from the network load balancer (%s) to instance (%s), port (%v)", clientCidrs, instanceSecurityGroupID, port)
				}
				klog.V(2).Infof("Removing rule for health check traffic from the network load balancer (%s) to instance (%s), port (%v)", clientCidrs, instanceSecurityGroupID, port)
			}

			if clientTraffic {
				clientRuleAnnotation := fmt.Sprintf("%s=%s", NLBClientRuleDescription, lbName)
				// Client Traffic
				permission := &ec2.IpPermission{
					FromPort:   aws.Int64(port),
					ToPort:     aws.Int64(port),
					IpProtocol: aws.String("tcp"),
				}
				ranges := []*ec2.IpRange{}
				for _, cidr := range clientCidrs {
					ranges = append(ranges, &ec2.IpRange{
						CidrIp:      aws.String(cidr),
						Description: aws.String(clientRuleAnnotation),
					})
				}
				permission.IpRanges = ranges
				if add {
					adds = append(adds, permission)
				} else {
					removes = append(removes, permission)
				}
			} else {
				healthRuleAnnotation := fmt.Sprintf("%s=%s", NLBHealthCheckRuleDescription, lbName)

				// NLB HealthCheck
				permission := &ec2.IpPermission{
					FromPort:   aws.Int64(port),
					ToPort:     aws.Int64(port),
					IpProtocol: aws.String("tcp"),
				}
				ranges := []*ec2.IpRange{}
				for _, cidr := range clientCidrs {
					ranges = append(ranges, &ec2.IpRange{
						CidrIp:      aws.String(cidr),
						Description: aws.String(healthRuleAnnotation),
					})
				}
				permission.IpRanges = ranges
				if add {
					adds = append(adds, permission)
				} else {
					removes = append(removes, permission)
				}
			}
		}

		if len(adds) > 0 {
			changed, err := c.addSecurityGroupIngress(instanceSecurityGroupID, adds)
			if err != nil {
				return err
			}
			if !changed {
				klog.Warning("Allowing ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		}

		if len(removes) > 0 {
			changed, err := c.removeSecurityGroupIngress(instanceSecurityGroupID, removes)
			if err != nil {
				return err
			}
			if !changed {
				klog.Warning("Revoking ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		}

		if clientTraffic {
			// MTU discovery
			mtuRuleAnnotation := fmt.Sprintf("%s=%s", NLBMtuDiscoveryRuleDescription, lbName)
			mtuPermission := &ec2.IpPermission{
				IpProtocol: aws.String("icmp"),
				FromPort:   aws.Int64(3),
				ToPort:     aws.Int64(4),
			}
			ranges := []*ec2.IpRange{}
			for _, cidr := range clientCidrs {
				ranges = append(ranges, &ec2.IpRange{
					CidrIp:      aws.String(cidr),
					Description: aws.String(mtuRuleAnnotation),
				})
			}
			mtuPermission.IpRanges = ranges

			group, err := c.findSecurityGroup(instanceSecurityGroupID)
			if err != nil {
				klog.Warningf("Error retrieving security group: %q", err)
				return err
			}

			if group == nil {
				klog.Warning("Security group not found: ", instanceSecurityGroupID)
				return nil
			}

			icmpExists := false
			permCount := 0
			for _, perm := range group.IpPermissions {
				if *perm.IpProtocol == "icmp" {
					icmpExists = true
					continue
				}

				if perm.FromPort != nil {
					permCount++
				}
			}

			if !icmpExists && permCount > 0 {
				// the icmp permission is missing
				changed, err := c.addSecurityGroupIngress(instanceSecurityGroupID, []*ec2.IpPermission{mtuPermission})
				if err != nil {
					klog.Warningf("Error adding MTU permission to security group: %q", err)
					return err
				}
				if !changed {
					klog.Warning("Allowing ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
				}
			} else if icmpExists && permCount == 0 {
				// there is no additional permissions, remove icmp
				changed, err := c.removeSecurityGroupIngress(instanceSecurityGroupID, []*ec2.IpPermission{mtuPermission})
				if err != nil {
					klog.Warningf("Error removing MTU permission to security group: %q", err)
					return err
				}
				if !changed {
					klog.Warning("Revoking ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
				}
			}
		}
	}
	return nil
}

// Add SG rules for a given NLB
func (c *Cloud) updateInstanceSecurityGroupsForNLB(mappings []nlbPortMapping, instances map[InstanceID]*ec2.Instance, lbName string, clientCidrs []string) error {
	if c.cfg.Global.DisableSecurityGroupIngress {
		return nil
	}

	vpcCidrBlocks, err := c.getVpcCidrBlocks()
	if err != nil {
		return err
	}

	// Unlike the classic ELB, NLB does not have a security group that we can
	// filter against all existing groups to see if they allow access. Instead
	// we use the IpRange.Description field to annotate NLB health check and
	// client traffic rules

	// Get the actual list of groups that allow ingress for the load-balancer
	var actualGroups []*ec2.SecurityGroup
	{
		// Server side filter
		describeRequest := &ec2.DescribeSecurityGroupsInput{}
		filters := []*ec2.Filter{
			newEc2Filter("ip-permission.protocol", "tcp"),
		}
		describeRequest.Filters = c.tagging.addFilters(filters)
		response, err := c.ec2.DescribeSecurityGroups(describeRequest)
		if err != nil {
			return fmt.Errorf("Error querying security groups for NLB: %q", err)
		}
		for _, sg := range response {
			if !c.tagging.hasClusterTag(sg.Tags) {
				continue
			}
			actualGroups = append(actualGroups, sg)
		}

		// client-side filter
		// Filter out groups that don't have IP Rules we've annotated for this service
		actualGroups = filterForIPRangeDescription(actualGroups, lbName)
	}

	taggedSecurityGroups, err := c.getTaggedSecurityGroups()
	if err != nil {
		return fmt.Errorf("Error querying for tagged security groups: %q", err)
	}

	externalTrafficPolicyIsLocal := false
	trafficPorts := []int64{}
	for i := range mappings {
		trafficPorts = append(trafficPorts, mappings[i].TrafficPort)
		if mappings[i].TrafficPort != mappings[i].HealthCheckPort {
			externalTrafficPolicyIsLocal = true
		}
	}

	healthCheckPorts := trafficPorts
	// if externalTrafficPolicy is Local, all listeners use the same health
	// check port
	if externalTrafficPolicyIsLocal && len(mappings) > 0 {
		healthCheckPorts = []int64{mappings[0].HealthCheckPort}
	}

	desiredGroupIds := []string{}
	// Scan instances for groups we want open
	for _, instance := range instances {
		securityGroup, err := findSecurityGroupForInstance(instance, taggedSecurityGroups)
		if err != nil {
			return err
		}

		if securityGroup == nil {
			klog.Warningf("Ignoring instance without security group: %s", aws.StringValue(instance.InstanceId))
			continue
		}

		id := aws.StringValue(securityGroup.GroupId)
		if id == "" {
			klog.Warningf("found security group without id: %v", securityGroup)
			continue
		}

		desiredGroupIds = append(desiredGroupIds, id)
	}

	// Run once for Client traffic
	err = c.updateInstanceSecurityGroupsForNLBTraffic(actualGroups, desiredGroupIds, trafficPorts, lbName, clientCidrs, true)
	if err != nil {
		return err
	}

	// Run once for health check traffic
	err = c.updateInstanceSecurityGroupsForNLBTraffic(actualGroups, desiredGroupIds, healthCheckPorts, lbName, vpcCidrBlocks, false)
	if err != nil {
		return err
	}

	return nil
}

func (c *Cloud) ensureLoadBalancer(namespacedName types.NamespacedName, loadBalancerName string, listeners []*elb.Listener, subnetIDs []string, securityGroupIDs []string, internalELB, proxyProtocol bool, loadBalancerAttributes *elb.LoadBalancerAttributes, annotations map[string]string) (*elb.LoadBalancerDescription, error) {
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
		if subnetIDs == nil {
			createRequest.Subnets = nil
		} else {
			createRequest.Subnets = aws.StringSlice(subnetIDs)
		}

		if securityGroupIDs == nil {
			createRequest.SecurityGroups = nil
		} else {
			createRequest.SecurityGroups = aws.StringSlice(securityGroupIDs)
		}

		// Get additional tags set by the user
		tags := getLoadBalancerAdditionalTags(annotations)

		// Add default tags
		tags[TagNameKubernetesService] = namespacedName.String()
		tags = c.tagging.buildTags(ResourceLifecycleOwned, tags)

		for k, v := range tags {
			createRequest.Tags = append(createRequest.Tags, &elb.Tag{
				Key: aws.String(k), Value: aws.String(v),
			})
		}

		klog.Infof("Creating load balancer for %v with name: %s", namespacedName, loadBalancerName)
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
				klog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to true", *listener.InstancePort)
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
				klog.V(2).Info("Detaching load balancer from removed subnets")
				_, err := c.elb.DetachLoadBalancerFromSubnets(request)
				if err != nil {
					return nil, fmt.Errorf("error detaching AWS loadbalancer from subnets: %q", err)
				}
				dirty = true
			}

			if additions.Len() != 0 {
				request := &elb.AttachLoadBalancerToSubnetsInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Subnets = stringSetToPointers(additions)
				klog.V(2).Info("Attaching load balancer to added subnets")
				_, err := c.elb.AttachLoadBalancerToSubnets(request)
				if err != nil {
					return nil, fmt.Errorf("error attaching AWS loadbalancer to subnets: %q", err)
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
				if securityGroupIDs == nil {
					request.SecurityGroups = nil
				} else {
					request.SecurityGroups = aws.StringSlice(securityGroupIDs)
				}
				klog.V(2).Info("Applying updated security groups to load balancer")
				_, err := c.elb.ApplySecurityGroupsToLoadBalancer(request)
				if err != nil {
					return nil, fmt.Errorf("error applying AWS loadbalancer security groups: %q", err)
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
					klog.Warning("Ignoring empty listener in AWS loadbalancer: ", loadBalancerName)
					continue
				}

				found := -1
				for i, expected := range listeners {
					if elbProtocolsAreEqual(actual.Protocol, expected.Protocol) {
						continue
					}
					if elbProtocolsAreEqual(actual.InstanceProtocol, expected.InstanceProtocol) {
						continue
					}
					if aws.Int64Value(actual.InstancePort) != aws.Int64Value(expected.InstancePort) {
						continue
					}
					if aws.Int64Value(actual.LoadBalancerPort) != aws.Int64Value(expected.LoadBalancerPort) {
						continue
					}
					if awsArnEquals(actual.SSLCertificateId, expected.SSLCertificateId) {
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
				klog.V(2).Info("Deleting removed load balancer listeners")
				_, err := c.elb.DeleteLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error deleting AWS loadbalancer listeners: %q", err)
				}
				dirty = true
			}

			if len(additions) != 0 {
				request := &elb.CreateLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Listeners = additions
				klog.V(2).Info("Creating added load balancer listeners")
				_, err := c.elb.CreateLoadBalancerListeners(request)
				if err != nil {
					return nil, fmt.Errorf("error creating AWS loadbalancer listeners: %q", err)
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
				// seem to return an error to us in these cases. Therefore, this will issue an API
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
					klog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to %t", instancePort, proxyProtocol)
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
					klog.V(2).Infof("Adjusting AWS loadbalancer proxy protocol on node port %d. Setting to false", instancePort)
					err := c.setBackendPolicies(loadBalancerName, instancePort, []*string{})
					if err != nil {
						return nil, err
					}
					dirty = true
				}
			}
		}

		{
			// Add additional tags
			klog.V(2).Infof("Creating additional load balancer tags for %s", loadBalancerName)
			tags := getLoadBalancerAdditionalTags(annotations)
			if len(tags) > 0 {
				err := c.addLoadBalancerTags(loadBalancerName, tags)
				if err != nil {
					return nil, fmt.Errorf("unable to create additional load balancer tags: %v", err)
				}
			}
		}
	}

	// Whether the ELB was new or existing, sync attributes regardless. This accounts for things
	// that cannot be specified at the time of creation and can only be modified after the fact,
	// e.g. idle connection timeout.
	{
		describeAttributesRequest := &elb.DescribeLoadBalancerAttributesInput{}
		describeAttributesRequest.LoadBalancerName = aws.String(loadBalancerName)
		describeAttributesOutput, err := c.elb.DescribeLoadBalancerAttributes(describeAttributesRequest)
		if err != nil {
			klog.Warning("Unable to retrieve load balancer attributes during attribute sync")
			return nil, err
		}

		foundAttributes := &describeAttributesOutput.LoadBalancerAttributes

		// Update attributes if they're dirty
		if !reflect.DeepEqual(loadBalancerAttributes, foundAttributes) {
			klog.V(2).Infof("Updating load-balancer attributes for %q", loadBalancerName)

			modifyAttributesRequest := &elb.ModifyLoadBalancerAttributesInput{}
			modifyAttributesRequest.LoadBalancerName = aws.String(loadBalancerName)
			modifyAttributesRequest.LoadBalancerAttributes = loadBalancerAttributes
			_, err = c.elb.ModifyLoadBalancerAttributes(modifyAttributesRequest)
			if err != nil {
				return nil, fmt.Errorf("Unable to update load balancer attributes during attribute sync: %q", err)
			}
			dirty = true
		}
	}

	if dirty {
		loadBalancer, err = c.describeLoadBalancer(loadBalancerName)
		if err != nil {
			klog.Warning("Unable to retrieve load balancer after creation/update")
			return nil, err
		}
	}

	return loadBalancer, nil
}

func createSubnetMappings(subnetIDs []string) []*elbv2.SubnetMapping {
	response := []*elbv2.SubnetMapping{}

	for _, id := range subnetIDs {
		// Ignore AllocationId for now
		response = append(response, &elbv2.SubnetMapping{SubnetId: aws.String(id)})
	}

	return response
}

// elbProtocolsAreEqual checks if two ELB protocol strings are considered the same
// Comparison is case insensitive
func elbProtocolsAreEqual(l, r *string) bool {
	if l == nil || r == nil {
		return l == r
	}
	return strings.EqualFold(aws.StringValue(l), aws.StringValue(r))
}

// awsArnEquals checks if two ARN strings are considered the same
// Comparison is case insensitive
func awsArnEquals(l, r *string) bool {
	if l == nil || r == nil {
		return l == r
	}
	return strings.EqualFold(aws.StringValue(l), aws.StringValue(r))
}

// getExpectedHealthCheck returns an elb.Healthcheck for the provided target
// and using either sensible defaults or overrides via Service annotations
func (c *Cloud) getExpectedHealthCheck(target string, annotations map[string]string) (*elb.HealthCheck, error) {
	healthcheck := &elb.HealthCheck{Target: &target}
	getOrDefault := func(annotation string, defaultValue int64) (*int64, error) {
		i64 := defaultValue
		var err error
		if s, ok := annotations[annotation]; ok {
			i64, err = strconv.ParseInt(s, 10, 0)
			if err != nil {
				return nil, fmt.Errorf("failed parsing health check annotation value: %v", err)
			}
		}
		return &i64, nil
	}
	var err error
	healthcheck.HealthyThreshold, err = getOrDefault(ServiceAnnotationLoadBalancerHCHealthyThreshold, defaultHCHealthyThreshold)
	if err != nil {
		return nil, err
	}
	healthcheck.UnhealthyThreshold, err = getOrDefault(ServiceAnnotationLoadBalancerHCUnhealthyThreshold, defaultHCUnhealthyThreshold)
	if err != nil {
		return nil, err
	}
	healthcheck.Timeout, err = getOrDefault(ServiceAnnotationLoadBalancerHCTimeout, defaultHCTimeout)
	if err != nil {
		return nil, err
	}
	healthcheck.Interval, err = getOrDefault(ServiceAnnotationLoadBalancerHCInterval, defaultHCInterval)
	if err != nil {
		return nil, err
	}
	if err = healthcheck.Validate(); err != nil {
		return nil, fmt.Errorf("some of the load balancer health check parameters are invalid: %v", err)
	}
	return healthcheck, nil
}

// Makes sure that the health check for an ELB matches the configured health check node port
func (c *Cloud) ensureLoadBalancerHealthCheck(loadBalancer *elb.LoadBalancerDescription, protocol string, port int32, path string, annotations map[string]string) error {
	name := aws.StringValue(loadBalancer.LoadBalancerName)

	actual := loadBalancer.HealthCheck
	expectedTarget := protocol + ":" + strconv.FormatInt(int64(port), 10) + path
	expected, err := c.getExpectedHealthCheck(expectedTarget, annotations)
	if err != nil {
		return fmt.Errorf("cannot update health check for load balancer %q: %q", name, err)
	}

	// comparing attributes 1 by 1 to avoid breakage in case a new field is
	// added to the HC which breaks the equality
	if aws.StringValue(expected.Target) == aws.StringValue(actual.Target) &&
		aws.Int64Value(expected.HealthyThreshold) == aws.Int64Value(actual.HealthyThreshold) &&
		aws.Int64Value(expected.UnhealthyThreshold) == aws.Int64Value(actual.UnhealthyThreshold) &&
		aws.Int64Value(expected.Interval) == aws.Int64Value(actual.Interval) &&
		aws.Int64Value(expected.Timeout) == aws.Int64Value(actual.Timeout) {
		return nil
	}

	request := &elb.ConfigureHealthCheckInput{}
	request.HealthCheck = expected
	request.LoadBalancerName = loadBalancer.LoadBalancerName

	_, err = c.elb.ConfigureHealthCheck(request)
	if err != nil {
		return fmt.Errorf("error configuring load balancer health check for %q: %q", name, err)
	}

	return nil
}

// Makes sure that exactly the specified hosts are registered as instances with the load balancer
func (c *Cloud) ensureLoadBalancerInstances(loadBalancerName string, lbInstances []*elb.Instance, instanceIDs map[InstanceID]*ec2.Instance) error {
	expected := sets.NewString()
	for id := range instanceIDs {
		expected.Insert(string(id))
	}

	actual := sets.NewString()
	for _, lbInstance := range lbInstances {
		actual.Insert(aws.StringValue(lbInstance.InstanceId))
	}

	additions := expected.Difference(actual)
	removals := actual.Difference(expected)

	addInstances := []*elb.Instance{}
	for _, instanceID := range additions.List() {
		addInstance := &elb.Instance{}
		addInstance.InstanceId = aws.String(instanceID)
		addInstances = append(addInstances, addInstance)
	}

	removeInstances := []*elb.Instance{}
	for _, instanceID := range removals.List() {
		removeInstance := &elb.Instance{}
		removeInstance.InstanceId = aws.String(instanceID)
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
		klog.V(1).Infof("Instances added to load-balancer %s", loadBalancerName)
	}

	if len(removeInstances) > 0 {
		deregisterRequest := &elb.DeregisterInstancesFromLoadBalancerInput{}
		deregisterRequest.Instances = removeInstances
		deregisterRequest.LoadBalancerName = aws.String(loadBalancerName)
		_, err := c.elb.DeregisterInstancesFromLoadBalancer(deregisterRequest)
		if err != nil {
			return err
		}
		klog.V(1).Infof("Instances removed from load-balancer %s", loadBalancerName)
	}

	return nil
}

func (c *Cloud) getLoadBalancerTLSPorts(loadBalancer *elb.LoadBalancerDescription) []int64 {
	ports := []int64{}

	for _, listenerDescription := range loadBalancer.ListenerDescriptions {
		protocol := aws.StringValue(listenerDescription.Listener.Protocol)
		if protocol == "SSL" || protocol == "HTTPS" {
			ports = append(ports, aws.Int64Value(listenerDescription.Listener.LoadBalancerPort))
		}
	}
	return ports
}

func (c *Cloud) ensureSSLNegotiationPolicy(loadBalancer *elb.LoadBalancerDescription, policyName string) error {
	klog.V(2).Info("Describing load balancer policies on load balancer")
	result, err := c.elb.DescribeLoadBalancerPolicies(&elb.DescribeLoadBalancerPoliciesInput{
		LoadBalancerName: loadBalancer.LoadBalancerName,
		PolicyNames: []*string{
			aws.String(fmt.Sprintf(SSLNegotiationPolicyNameFormat, policyName)),
		},
	})
	if err != nil {
		if aerr, ok := err.(awserr.Error); ok {
			switch aerr.Code() {
			case elb.ErrCodePolicyNotFoundException:
			default:
				return fmt.Errorf("error describing security policies on load balancer: %q", err)
			}
		}
	}

	if len(result.PolicyDescriptions) > 0 {
		return nil
	}

	klog.V(2).Infof("Creating SSL negotiation policy '%s' on load balancer", fmt.Sprintf(SSLNegotiationPolicyNameFormat, policyName))
	// there is an upper limit of 98 policies on an ELB, we're pretty safe from
	// running into it
	_, err = c.elb.CreateLoadBalancerPolicy(&elb.CreateLoadBalancerPolicyInput{
		LoadBalancerName: loadBalancer.LoadBalancerName,
		PolicyName:       aws.String(fmt.Sprintf(SSLNegotiationPolicyNameFormat, policyName)),
		PolicyTypeName:   aws.String("SSLNegotiationPolicyType"),
		PolicyAttributes: []*elb.PolicyAttribute{
			{
				AttributeName:  aws.String("Reference-Security-Policy"),
				AttributeValue: aws.String(policyName),
			},
		},
	})
	if err != nil {
		return fmt.Errorf("error creating security policy on load balancer: %q", err)
	}
	return nil
}

func (c *Cloud) setSSLNegotiationPolicy(loadBalancerName, sslPolicyName string, port int64) error {
	policyName := fmt.Sprintf(SSLNegotiationPolicyNameFormat, sslPolicyName)
	request := &elb.SetLoadBalancerPoliciesOfListenerInput{
		LoadBalancerName: aws.String(loadBalancerName),
		LoadBalancerPort: aws.Int64(port),
		PolicyNames: []*string{
			aws.String(policyName),
		},
	}
	klog.V(2).Infof("Setting SSL negotiation policy '%s' on load balancer", policyName)
	_, err := c.elb.SetLoadBalancerPoliciesOfListener(request)
	if err != nil {
		return fmt.Errorf("error setting SSL negotiation policy '%s' on load balancer: %q", policyName, err)
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
	klog.V(2).Info("Creating proxy protocol policy on load balancer")
	_, err := c.elb.CreateLoadBalancerPolicy(request)
	if err != nil {
		return fmt.Errorf("error creating proxy protocol policy on load balancer: %q", err)
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
		klog.V(2).Infof("Adding AWS loadbalancer backend policies on node port %d", instancePort)
	} else {
		klog.V(2).Infof("Removing AWS loadbalancer backend policies on node port %d", instancePort)
	}
	_, err := c.elb.SetLoadBalancerPoliciesForBackendServer(request)
	if err != nil {
		return fmt.Errorf("error adjusting AWS loadbalancer backend policies: %q", err)
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

// findInstancesForELB gets the EC2 instances corresponding to the Nodes, for setting up an ELB
// We ignore Nodes (with a log message) where the instanceid cannot be determined from the provider,
// and we ignore instances which are not found
func (c *Cloud) findInstancesForELB(nodes []*v1.Node) (map[InstanceID]*ec2.Instance, error) {
	// Map to instance ids ignoring Nodes where we cannot find the id (but logging)
	instanceIDs := mapToAWSInstanceIDsTolerant(nodes)

	cacheCriteria := cacheCriteria{
		// MaxAge not required, because we only care about security groups, which should not change
		HasInstances: instanceIDs, // Refresh if any of the instance ids are missing
	}
	snapshot, err := c.instanceCache.describeAllInstancesCached(cacheCriteria)
	if err != nil {
		return nil, err
	}

	instances := snapshot.FindInstances(instanceIDs)
	// We ignore instances that cannot be found

	return instances, nil
}
