// +build !providerless

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
	"encoding/hex"
	"fmt"
	"reflect"
	"regexp"
	"strconv"
	"strings"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/aws/aws-sdk-go/service/elbv2"
	"k8s.io/klog/v2"

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

	lbAttrLoadBalancingCrossZoneEnabled = "load_balancing.cross_zone.enabled"
	lbAttrAccessLogsS3Enabled           = "access_logs.s3.enabled"
	lbAttrAccessLogsS3Bucket            = "access_logs.s3.bucket"
	lbAttrAccessLogsS3Prefix            = "access_logs.s3.prefix"
)

var (
	// Defaults for ELB Healthcheck
	defaultElbHCHealthyThreshold   = int64(2)
	defaultElbHCUnhealthyThreshold = int64(6)
	defaultElbHCTimeout            = int64(5)
	defaultElbHCInterval           = int64(10)
	defaultNlbHealthCheckInterval  = int64(30)
	defaultNlbHealthCheckTimeout   = int64(10)
	defaultNlbHealthCheckThreshold = int64(3)
	defaultHealthCheckPort         = "traffic-port"
	defaultHealthCheckPath         = "/"
)

func isNLB(annotations map[string]string) bool {
	if annotations[ServiceAnnotationLoadBalancerType] == "nlb" {
		return true
	}
	return false
}

func isLBExternal(annotations map[string]string) bool {
	if val := annotations[ServiceAnnotationLoadBalancerType]; val == "nlb-ip" || val == "external" {
		return true
	}
	return false
}

type healthCheckConfig struct {
	Port               string
	Path               string
	Protocol           string
	Interval           int64
	Timeout            int64
	HealthyThreshold   int64
	UnhealthyThreshold int64
}

type nlbPortMapping struct {
	FrontendPort     int64
	FrontendProtocol string

	TrafficPort     int64
	TrafficProtocol string

	SSLCertificateARN string
	SSLPolicy         string
	HealthCheckConfig healthCheckConfig
}

// getKeyValuePropertiesFromAnnotation converts the comma separated list of key-value
// pairs from the specified annotation and returns it as a map.
func getKeyValuePropertiesFromAnnotation(annotations map[string]string, annotation string) map[string]string {
	additionalTags := make(map[string]string)
	if additionalTagsList, ok := annotations[annotation]; ok {
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
	tags := getKeyValuePropertiesFromAnnotation(annotations, ServiceAnnotationLoadBalancerAdditionalTags)
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

		var allocationIDs []string
		if eipList, present := annotations[ServiceAnnotationLoadBalancerEIPAllocations]; present {
			allocationIDs = strings.Split(eipList, ",")
			if len(allocationIDs) != len(subnetIDs) {
				return nil, fmt.Errorf("error creating load balancer: Must have same number of EIP AllocationIDs (%d) and SubnetIDs (%d)", len(allocationIDs), len(subnetIDs))
			}
		}

		// We are supposed to specify one subnet per AZ.
		// TODO: What happens if we have more than one subnet per AZ?
		createRequest.SubnetMappings = createSubnetMappings(subnetIDs, allocationIDs)

		for k, v := range tags {
			createRequest.Tags = append(createRequest.Tags, &elbv2.Tag{
				Key: aws.String(k), Value: aws.String(v),
			})
		}

		klog.Infof("Creating load balancer for %v with name: %s", namespacedName, loadBalancerName)
		createResponse, err := c.elbv2.CreateLoadBalancer(createRequest)
		if err != nil {
			return nil, fmt.Errorf("error creating load balancer: %q", err)
		}

		loadBalancer = createResponse.LoadBalancers[0]
		for i := range mappings {
			// It is easier to keep track of updates by having possibly
			// duplicate target groups where the backend port is the same
			_, err := c.createListenerV2(createResponse.LoadBalancers[0].LoadBalancerArn, mappings[i], namespacedName, instanceIDs, *createResponse.LoadBalancers[0].VpcId, tags)
			if err != nil {
				return nil, fmt.Errorf("error creating listener: %q", err)
			}
		}
		if err := c.reconcileLBAttributes(aws.StringValue(loadBalancer.LoadBalancerArn), annotations); err != nil {
			return nil, err
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
				return nil, fmt.Errorf("error describing listeners: %q", err)
			}

			// actual maps FrontendPort to an elbv2.Listener
			actual := map[int64]map[string]*elbv2.Listener{}
			for _, listener := range listenerDescriptions.Listeners {
				if actual[*listener.Port] == nil {
					actual[*listener.Port] = map[string]*elbv2.Listener{}
				}
				actual[*listener.Port][*listener.Protocol] = listener
			}

			actualTargetGroups, err := c.elbv2.DescribeTargetGroups(
				&elbv2.DescribeTargetGroupsInput{
					LoadBalancerArn: loadBalancer.LoadBalancerArn,
				},
			)
			if err != nil {
				return nil, fmt.Errorf("error listing target groups: %q", err)
			}

			nodePortTargetGroup := map[int64]*elbv2.TargetGroup{}
			for _, targetGroup := range actualTargetGroups.TargetGroups {
				nodePortTargetGroup[*targetGroup.Port] = targetGroup
			}

			// Handle additions/modifications
			for _, mapping := range mappings {
				frontendPort := mapping.FrontendPort
				frontendProtocol := mapping.FrontendProtocol
				nodePort := mapping.TrafficPort
				// modifications
				if listener, ok := actual[frontendPort][frontendProtocol]; ok {
					listenerNeedsModification := false

					if aws.StringValue(listener.Protocol) != mapping.FrontendProtocol {
						listenerNeedsModification = true
					}
					switch mapping.FrontendProtocol {
					case elbv2.ProtocolEnumTls:
						{
							if aws.StringValue(listener.SslPolicy) != mapping.SSLPolicy {
								listenerNeedsModification = true
							}
							if len(listener.Certificates) == 0 || aws.StringValue(listener.Certificates[0].CertificateArn) != mapping.SSLCertificateARN {
								listenerNeedsModification = true
							}
						}
					case elbv2.ProtocolEnumTcp:
						{
							if aws.StringValue(listener.SslPolicy) != "" {
								listenerNeedsModification = true
							}
							if len(listener.Certificates) != 0 {
								listenerNeedsModification = true
							}
						}
					}

					// recreate targetGroup if trafficPort, protocol or HealthCheckProtocol changed
					healthCheckModified := false
					targetGroupRecreated := false
					targetGroup, ok := nodePortTargetGroup[nodePort]

					if targetGroup != nil && (!strings.EqualFold(mapping.HealthCheckConfig.Protocol, aws.StringValue(targetGroup.HealthCheckProtocol)) ||
						mapping.HealthCheckConfig.Interval != aws.Int64Value(targetGroup.HealthCheckIntervalSeconds)) {
						healthCheckModified = true
					}

					if !ok || aws.StringValue(targetGroup.Protocol) != mapping.TrafficProtocol || healthCheckModified {
						// create new target group
						targetGroup, err = c.ensureTargetGroup(
							nil,
							namespacedName,
							mapping,
							instanceIDs,
							*loadBalancer.VpcId,
							tags,
						)
						if err != nil {
							return nil, err
						}
						targetGroupRecreated = true
						listenerNeedsModification = true
					}

					if listenerNeedsModification {
						modifyListenerInput := &elbv2.ModifyListenerInput{
							ListenerArn: listener.ListenerArn,
							Port:        aws.Int64(frontendPort),
							Protocol:    aws.String(mapping.FrontendProtocol),
							DefaultActions: []*elbv2.Action{{
								TargetGroupArn: targetGroup.TargetGroupArn,
								Type:           aws.String("forward"),
							}},
						}
						if mapping.FrontendProtocol == elbv2.ProtocolEnumTls {
							if mapping.SSLPolicy != "" {
								modifyListenerInput.SslPolicy = aws.String(mapping.SSLPolicy)
							}
							modifyListenerInput.Certificates = []*elbv2.Certificate{
								{
									CertificateArn: aws.String(mapping.SSLCertificateARN),
								},
							}
						}
						if _, err := c.elbv2.ModifyListener(modifyListenerInput); err != nil {
							return nil, fmt.Errorf("error updating load balancer listener: %q", err)
						}
					}

					// Delete old targetGroup if needed
					if targetGroupRecreated {
						if _, err := c.elbv2.DeleteTargetGroup(&elbv2.DeleteTargetGroupInput{
							TargetGroupArn: listener.DefaultActions[0].TargetGroupArn,
						}); err != nil {
							return nil, fmt.Errorf("error deleting old target group: %q", err)
						}
					} else {
						// Run ensureTargetGroup to make sure instances in service are up-to-date
						_, err = c.ensureTargetGroup(
							targetGroup,
							namespacedName,
							mapping,
							instanceIDs,
							*loadBalancer.VpcId,
							tags,
						)
						if err != nil {
							return nil, err
						}
					}
					dirty = true
					continue
				}

				// Additions
				_, err := c.createListenerV2(loadBalancer.LoadBalancerArn, mapping, namespacedName, instanceIDs, *loadBalancer.VpcId, tags)
				if err != nil {
					return nil, err
				}
				dirty = true
			}

			frontEndPorts := map[int64]map[string]bool{}
			for i := range mappings {
				if frontEndPorts[mappings[i].FrontendPort] == nil {
					frontEndPorts[mappings[i].FrontendPort] = map[string]bool{}
				}
				frontEndPorts[mappings[i].FrontendPort][mappings[i].FrontendProtocol] = true
			}

			// handle deletions
			for port := range actual {
				for protocol := range actual[port] {
					if _, ok := frontEndPorts[port][protocol]; !ok {
						err := c.deleteListenerV2(actual[port][protocol])
						if err != nil {
							return nil, err
						}
						dirty = true
					}
				}
			}
		}
		if err := c.reconcileLBAttributes(aws.StringValue(loadBalancer.LoadBalancerArn), annotations); err != nil {
			return nil, err
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
				return nil, fmt.Errorf("error retrieving load balancer after update: %q", err)
			}
			loadBalancer = loadBalancers.LoadBalancers[0]
		}
	}
	return loadBalancer, nil
}

func (c *Cloud) reconcileLBAttributes(loadBalancerArn string, annotations map[string]string) error {
	desiredLoadBalancerAttributes := map[string]string{}

	desiredLoadBalancerAttributes[lbAttrLoadBalancingCrossZoneEnabled] = "false"
	crossZoneLoadBalancingEnabledAnnotation := annotations[ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled]
	if crossZoneLoadBalancingEnabledAnnotation != "" {
		crossZoneEnabled, err := strconv.ParseBool(crossZoneLoadBalancingEnabledAnnotation)
		if err != nil {
			return fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled,
				crossZoneLoadBalancingEnabledAnnotation,
			)
		}

		if crossZoneEnabled {
			desiredLoadBalancerAttributes[lbAttrLoadBalancingCrossZoneEnabled] = "true"
		}
	}

	desiredLoadBalancerAttributes[lbAttrAccessLogsS3Enabled] = "false"
	accessLogsS3EnabledAnnotation := annotations[ServiceAnnotationLoadBalancerAccessLogEnabled]
	if accessLogsS3EnabledAnnotation != "" {
		accessLogsS3Enabled, err := strconv.ParseBool(accessLogsS3EnabledAnnotation)
		if err != nil {
			return fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerAccessLogEnabled,
				accessLogsS3EnabledAnnotation,
			)
		}

		if accessLogsS3Enabled {
			desiredLoadBalancerAttributes[lbAttrAccessLogsS3Enabled] = "true"
		}
	}

	desiredLoadBalancerAttributes[lbAttrAccessLogsS3Bucket] = annotations[ServiceAnnotationLoadBalancerAccessLogS3BucketName]
	desiredLoadBalancerAttributes[lbAttrAccessLogsS3Prefix] = annotations[ServiceAnnotationLoadBalancerAccessLogS3BucketPrefix]

	currentLoadBalancerAttributes := map[string]string{}
	describeAttributesOutput, err := c.elbv2.DescribeLoadBalancerAttributes(&elbv2.DescribeLoadBalancerAttributesInput{
		LoadBalancerArn: aws.String(loadBalancerArn),
	})
	if err != nil {
		return fmt.Errorf("unable to retrieve load balancer attributes during attribute sync: %q", err)
	}
	for _, attr := range describeAttributesOutput.Attributes {
		currentLoadBalancerAttributes[aws.StringValue(attr.Key)] = aws.StringValue(attr.Value)
	}

	var changedAttributes []*elbv2.LoadBalancerAttribute
	if desiredLoadBalancerAttributes[lbAttrLoadBalancingCrossZoneEnabled] != currentLoadBalancerAttributes[lbAttrLoadBalancingCrossZoneEnabled] {
		changedAttributes = append(changedAttributes, &elbv2.LoadBalancerAttribute{
			Key:   aws.String(lbAttrLoadBalancingCrossZoneEnabled),
			Value: aws.String(desiredLoadBalancerAttributes[lbAttrLoadBalancingCrossZoneEnabled]),
		})
	}
	if desiredLoadBalancerAttributes[lbAttrAccessLogsS3Enabled] != currentLoadBalancerAttributes[lbAttrAccessLogsS3Enabled] {
		changedAttributes = append(changedAttributes, &elbv2.LoadBalancerAttribute{
			Key:   aws.String(lbAttrAccessLogsS3Enabled),
			Value: aws.String(desiredLoadBalancerAttributes[lbAttrAccessLogsS3Enabled]),
		})
	}

	// ELBV2 API forbids us to set bucket to an empty bucket, so we keep it unchanged if AccessLogsS3Enabled==false.
	if desiredLoadBalancerAttributes[lbAttrAccessLogsS3Enabled] == "true" {
		if desiredLoadBalancerAttributes[lbAttrAccessLogsS3Bucket] != currentLoadBalancerAttributes[lbAttrAccessLogsS3Bucket] {
			changedAttributes = append(changedAttributes, &elbv2.LoadBalancerAttribute{
				Key:   aws.String(lbAttrAccessLogsS3Bucket),
				Value: aws.String(desiredLoadBalancerAttributes[lbAttrAccessLogsS3Bucket]),
			})
		}
		if desiredLoadBalancerAttributes[lbAttrAccessLogsS3Prefix] != currentLoadBalancerAttributes[lbAttrAccessLogsS3Prefix] {
			changedAttributes = append(changedAttributes, &elbv2.LoadBalancerAttribute{
				Key:   aws.String(lbAttrAccessLogsS3Prefix),
				Value: aws.String(desiredLoadBalancerAttributes[lbAttrAccessLogsS3Prefix]),
			})
		}
	}

	if len(changedAttributes) > 0 {
		klog.V(2).Infof("updating load-balancer attributes for %q", loadBalancerArn)

		_, err = c.elbv2.ModifyLoadBalancerAttributes(&elbv2.ModifyLoadBalancerAttributesInput{
			LoadBalancerArn: aws.String(loadBalancerArn),
			Attributes:      changedAttributes,
		})
		if err != nil {
			return fmt.Errorf("unable to update load balancer attributes during attribute sync: %q", err)
		}
	}
	return nil
}

var invalidELBV2NameRegex = regexp.MustCompile("[^[:alnum:]]")

// buildTargetGroupName will build unique name for targetGroup of service & port.
// the name is in format k8s-{namespace:8}-{name:8}-{uuid:10} (chosen to benefit most common use cases).
// Note: nodePort & targetProtocol & targetType are included since they cannot be modified on existing targetGroup.
func (c *Cloud) buildTargetGroupName(serviceName types.NamespacedName, servicePort int64, nodePort int64, targetProtocol string, targetType string, mapping nlbPortMapping) string {
	hasher := sha1.New()
	_, _ = hasher.Write([]byte(c.tagging.clusterID()))
	_, _ = hasher.Write([]byte(serviceName.Namespace))
	_, _ = hasher.Write([]byte(serviceName.Name))
	_, _ = hasher.Write([]byte(strconv.FormatInt(servicePort, 10)))
	_, _ = hasher.Write([]byte(strconv.FormatInt(nodePort, 10)))
	_, _ = hasher.Write([]byte(targetProtocol))
	_, _ = hasher.Write([]byte(targetType))
	_, _ = hasher.Write([]byte(mapping.HealthCheckConfig.Protocol))
	_, _ = hasher.Write([]byte(strconv.FormatInt(mapping.HealthCheckConfig.Interval, 10)))
	tgUUID := hex.EncodeToString(hasher.Sum(nil))

	sanitizedNamespace := invalidELBV2NameRegex.ReplaceAllString(serviceName.Namespace, "")
	sanitizedServiceName := invalidELBV2NameRegex.ReplaceAllString(serviceName.Name, "")
	return fmt.Sprintf("k8s-%.8s-%.8s-%.10s", sanitizedNamespace, sanitizedServiceName, tgUUID)
}

func (c *Cloud) createListenerV2(loadBalancerArn *string, mapping nlbPortMapping, namespacedName types.NamespacedName, instanceIDs []string, vpcID string, tags map[string]string) (listener *elbv2.Listener, err error) {
	target, err := c.ensureTargetGroup(
		nil,
		namespacedName,
		mapping,
		instanceIDs,
		vpcID,
		tags,
	)
	if err != nil {
		return nil, err
	}

	createListernerInput := &elbv2.CreateListenerInput{
		LoadBalancerArn: loadBalancerArn,
		Port:            aws.Int64(mapping.FrontendPort),
		Protocol:        aws.String(mapping.FrontendProtocol),
		DefaultActions: []*elbv2.Action{{
			TargetGroupArn: target.TargetGroupArn,
			Type:           aws.String(elbv2.ActionTypeEnumForward),
		}},
	}
	if mapping.FrontendProtocol == "TLS" {
		if mapping.SSLPolicy != "" {
			createListernerInput.SslPolicy = aws.String(mapping.SSLPolicy)
		}
		createListernerInput.Certificates = []*elbv2.Certificate{
			{
				CertificateArn: aws.String(mapping.SSLCertificateARN),
			},
		}
	}

	klog.Infof("Creating load balancer listener for %v", namespacedName)
	createListenerOutput, err := c.elbv2.CreateListener(createListernerInput)
	if err != nil {
		return nil, fmt.Errorf("error creating load balancer listener: %q", err)
	}
	return createListenerOutput.Listeners[0], nil
}

// cleans up listener and corresponding target group
func (c *Cloud) deleteListenerV2(listener *elbv2.Listener) error {
	_, err := c.elbv2.DeleteListener(&elbv2.DeleteListenerInput{ListenerArn: listener.ListenerArn})
	if err != nil {
		return fmt.Errorf("error deleting load balancer listener: %q", err)
	}
	_, err = c.elbv2.DeleteTargetGroup(&elbv2.DeleteTargetGroupInput{TargetGroupArn: listener.DefaultActions[0].TargetGroupArn})
	if err != nil {
		return fmt.Errorf("error deleting load balancer target group: %q", err)
	}
	return nil
}

// ensureTargetGroup creates a target group with a set of instances.
func (c *Cloud) ensureTargetGroup(targetGroup *elbv2.TargetGroup, serviceName types.NamespacedName, mapping nlbPortMapping, instances []string, vpcID string, tags map[string]string) (*elbv2.TargetGroup, error) {
	dirty := false
	if targetGroup == nil {
		targetType := "instance"
		name := c.buildTargetGroupName(serviceName, mapping.FrontendPort, mapping.TrafficPort, mapping.TrafficProtocol, targetType, mapping)
		klog.Infof("Creating load balancer target group for %v with name: %s", serviceName, name)
		input := &elbv2.CreateTargetGroupInput{
			VpcId:                      aws.String(vpcID),
			Name:                       aws.String(name),
			Port:                       aws.Int64(mapping.TrafficPort),
			Protocol:                   aws.String(mapping.TrafficProtocol),
			TargetType:                 aws.String(targetType),
			HealthCheckIntervalSeconds: aws.Int64(mapping.HealthCheckConfig.Interval),
			HealthCheckPort:            aws.String(mapping.HealthCheckConfig.Port),
			HealthCheckProtocol:        aws.String(mapping.HealthCheckConfig.Protocol),
			HealthyThresholdCount:      aws.Int64(mapping.HealthCheckConfig.HealthyThreshold),
			UnhealthyThresholdCount:    aws.Int64(mapping.HealthCheckConfig.UnhealthyThreshold),
			// HealthCheckTimeoutSeconds:  Currently not configurable, 6 seconds for HTTP, 10 for TCP/HTTPS
		}

		if mapping.HealthCheckConfig.Protocol != elbv2.ProtocolEnumTcp {
			input.HealthCheckPath = aws.String(mapping.HealthCheckConfig.Path)
		}

		result, err := c.elbv2.CreateTargetGroup(input)
		if err != nil {
			return nil, fmt.Errorf("error creating load balancer target group: %q", err)
		}
		if len(result.TargetGroups) != 1 {
			return nil, fmt.Errorf("expected only one target group on CreateTargetGroup, got %d groups", len(result.TargetGroups))
		}

		if len(tags) != 0 {
			targetGroupTags := make([]*elbv2.Tag, 0, len(tags))
			for k, v := range tags {
				targetGroupTags = append(targetGroupTags, &elbv2.Tag{
					Key: aws.String(k), Value: aws.String(v),
				})
			}
			tgArn := aws.StringValue(result.TargetGroups[0].TargetGroupArn)
			if _, err := c.elbv2.AddTags(&elbv2.AddTagsInput{
				ResourceArns: []*string{aws.String(tgArn)},
				Tags:         targetGroupTags,
			}); err != nil {
				return nil, fmt.Errorf("error adding tags for targetGroup %s due to %q", tgArn, err)
			}
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
			return nil, fmt.Errorf("error registering targets for load balancer: %q", err)
		}

		return result.TargetGroups[0], nil
	}

	// handle instances in service
	{
		healthResponse, err := c.elbv2.DescribeTargetHealth(&elbv2.DescribeTargetHealthInput{TargetGroupArn: targetGroup.TargetGroupArn})
		if err != nil {
			return nil, fmt.Errorf("error describing target group health: %q", err)
		}
		actualIDs := []string{}
		for _, healthDescription := range healthResponse.TargetHealthDescriptions {
			if aws.StringValue(healthDescription.TargetHealth.State) == elbv2.TargetHealthStateEnumHealthy {
				actualIDs = append(actualIDs, *healthDescription.Target.Id)
			} else if healthDescription.TargetHealth.Reason != nil {
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
				return nil, fmt.Errorf("error registering new targets in target group: %q", err)
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
				return nil, fmt.Errorf("error trying to deregister targets in target group: %q", err)
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
		if mapping.HealthCheckConfig.Port != aws.StringValue(targetGroup.HealthCheckPort) {
			input.HealthCheckPort = aws.String(mapping.HealthCheckConfig.Port)
			dirtyHealthCheck = true
		}
		if mapping.HealthCheckConfig.HealthyThreshold != aws.Int64Value(targetGroup.HealthyThresholdCount) {
			dirtyHealthCheck = true
			input.HealthyThresholdCount = aws.Int64(mapping.HealthCheckConfig.HealthyThreshold)
			input.UnhealthyThresholdCount = aws.Int64(mapping.HealthCheckConfig.UnhealthyThreshold)
		}
		if !strings.EqualFold(mapping.HealthCheckConfig.Protocol, elbv2.ProtocolEnumTcp) {
			if mapping.HealthCheckConfig.Path != aws.StringValue(input.HealthCheckPath) {
				input.HealthCheckPath = aws.String(mapping.HealthCheckConfig.Path)
				dirtyHealthCheck = true
			}
		}

		if dirtyHealthCheck {
			_, err := c.elbv2.ModifyTargetGroup(input)
			if err != nil {
				return nil, fmt.Errorf("error modifying target group health check: %q", err)
			}

			dirty = true
		}
	}

	if dirty {
		result, err := c.elbv2.DescribeTargetGroups(&elbv2.DescribeTargetGroupsInput{
			TargetGroupArns: []*string{targetGroup.TargetGroupArn},
		})
		if err != nil {
			return nil, fmt.Errorf("error retrieving target group after creation/update: %q", err)
		}
		targetGroup = result.TargetGroups[0]
	}

	return targetGroup, nil
}

// updateInstanceSecurityGroupsForNLB will adjust securityGroup's settings to allow inbound traffic into instances from clientCIDRs and portMappings.
// TIP: if either instances or clientCIDRs or portMappings are nil, then the securityGroup rules for lbName are cleared.
func (c *Cloud) updateInstanceSecurityGroupsForNLB(lbName string, instances map[InstanceID]*ec2.Instance, subnetCIDRs []string, clientCIDRs []string, portMappings []nlbPortMapping) error {
	if c.cfg.Global.DisableSecurityGroupIngress {
		return nil
	}

	clusterSGs, err := c.getTaggedSecurityGroups()
	if err != nil {
		return fmt.Errorf("error querying for tagged security groups: %q", err)
	}
	// scan instances for groups we want to open
	desiredSGIDs := sets.String{}
	for _, instance := range instances {
		sg, err := findSecurityGroupForInstance(instance, clusterSGs)
		if err != nil {
			return err
		}
		if sg == nil {
			klog.Warningf("Ignoring instance without security group: %s", aws.StringValue(instance.InstanceId))
			continue
		}
		desiredSGIDs.Insert(aws.StringValue(sg.GroupId))
	}

	// TODO(@M00nF1sh): do we really needs to support SG without cluster tag at current version?
	// findSecurityGroupForInstance might return SG that are not tagged.
	{
		for sgID := range desiredSGIDs.Difference(sets.StringKeySet(clusterSGs)) {
			sg, err := c.findSecurityGroup(sgID)
			if err != nil {
				return fmt.Errorf("error finding instance group: %q", err)
			}
			clusterSGs[sgID] = sg
		}
	}

	{
		clientPorts := sets.Int64{}
		clientProtocol := "tcp"
		healthCheckPorts := sets.Int64{}
		for _, port := range portMappings {
			clientPorts.Insert(port.TrafficPort)
			hcPort := port.TrafficPort
			if port.HealthCheckConfig.Port != defaultHealthCheckPort {
				var err error
				if hcPort, err = strconv.ParseInt(port.HealthCheckConfig.Port, 10, 0); err != nil {
					return fmt.Errorf("Invalid health check port %v", port.HealthCheckConfig.Port)
				}
			}
			healthCheckPorts.Insert(hcPort)
			if port.TrafficProtocol == string(v1.ProtocolUDP) {
				clientProtocol = "udp"
			}
		}
		clientRuleAnnotation := fmt.Sprintf("%s=%s", NLBClientRuleDescription, lbName)
		healthRuleAnnotation := fmt.Sprintf("%s=%s", NLBHealthCheckRuleDescription, lbName)
		for sgID, sg := range clusterSGs {
			sgPerms := NewIPPermissionSet(sg.IpPermissions...).Ungroup()
			if desiredSGIDs.Has(sgID) {
				if err := c.updateInstanceSecurityGroupForNLBTraffic(sgID, sgPerms, healthRuleAnnotation, "tcp", healthCheckPorts, subnetCIDRs); err != nil {
					return err
				}
				if err := c.updateInstanceSecurityGroupForNLBTraffic(sgID, sgPerms, clientRuleAnnotation, clientProtocol, clientPorts, clientCIDRs); err != nil {
					return err
				}
			} else {
				if err := c.updateInstanceSecurityGroupForNLBTraffic(sgID, sgPerms, healthRuleAnnotation, "tcp", nil, nil); err != nil {
					return err
				}
				if err := c.updateInstanceSecurityGroupForNLBTraffic(sgID, sgPerms, clientRuleAnnotation, clientProtocol, nil, nil); err != nil {
					return err
				}
			}
			if !sgPerms.Equal(NewIPPermissionSet(sg.IpPermissions...).Ungroup()) {
				if err := c.updateInstanceSecurityGroupForNLBMTU(sgID, sgPerms); err != nil {
					return err
				}
			}
		}
	}
	return nil
}

// updateInstanceSecurityGroupForNLBTraffic will manage permissions set(identified by ruleDesc) on securityGroup to match desired set(allow protocol traffic from ports/cidr).
// Note: sgPerms will be updated to reflect the current permission set on SG after update.
func (c *Cloud) updateInstanceSecurityGroupForNLBTraffic(sgID string, sgPerms IPPermissionSet, ruleDesc string, protocol string, ports sets.Int64, cidrs []string) error {
	desiredPerms := NewIPPermissionSet()
	for port := range ports {
		for _, cidr := range cidrs {
			desiredPerms.Insert(&ec2.IpPermission{
				IpProtocol: aws.String(protocol),
				FromPort:   aws.Int64(port),
				ToPort:     aws.Int64(port),
				IpRanges: []*ec2.IpRange{
					{
						CidrIp:      aws.String(cidr),
						Description: aws.String(ruleDesc),
					},
				},
			})
		}
	}

	permsToGrant := desiredPerms.Difference(sgPerms)
	permsToRevoke := sgPerms.Difference(desiredPerms)
	permsToRevoke.DeleteIf(IPPermissionNotMatch{IPPermissionMatchDesc{ruleDesc}})
	if len(permsToRevoke) > 0 {
		permsToRevokeList := permsToRevoke.List()
		changed, err := c.removeSecurityGroupIngress(sgID, permsToRevokeList)
		if err != nil {
			klog.Warningf("Error remove traffic permission from security group: %q", err)
			return err
		}
		if !changed {
			klog.Warning("Revoking ingress was not needed; concurrent change? groupId=", sgID)
		}
		sgPerms.Delete(permsToRevokeList...)
	}
	if len(permsToGrant) > 0 {
		permsToGrantList := permsToGrant.List()
		changed, err := c.addSecurityGroupIngress(sgID, permsToGrantList)
		if err != nil {
			klog.Warningf("Error add traffic permission to security group: %q", err)
			return err
		}
		if !changed {
			klog.Warning("Allowing ingress was not needed; concurrent change? groupId=", sgID)
		}
		sgPerms.Insert(permsToGrantList...)
	}
	return nil
}

// Note: sgPerms will be updated to reflect the current permission set on SG after update.
func (c *Cloud) updateInstanceSecurityGroupForNLBMTU(sgID string, sgPerms IPPermissionSet) error {
	desiredPerms := NewIPPermissionSet()
	for _, perm := range sgPerms {
		for _, ipRange := range perm.IpRanges {
			if strings.Contains(aws.StringValue(ipRange.Description), NLBClientRuleDescription) {
				desiredPerms.Insert(&ec2.IpPermission{
					IpProtocol: aws.String("icmp"),
					FromPort:   aws.Int64(3),
					ToPort:     aws.Int64(4),
					IpRanges: []*ec2.IpRange{
						{
							CidrIp:      ipRange.CidrIp,
							Description: aws.String(NLBMtuDiscoveryRuleDescription),
						},
					},
				})
			}
		}
	}

	permsToGrant := desiredPerms.Difference(sgPerms)
	permsToRevoke := sgPerms.Difference(desiredPerms)
	permsToRevoke.DeleteIf(IPPermissionNotMatch{IPPermissionMatchDesc{NLBMtuDiscoveryRuleDescription}})
	if len(permsToRevoke) > 0 {
		permsToRevokeList := permsToRevoke.List()
		changed, err := c.removeSecurityGroupIngress(sgID, permsToRevokeList)
		if err != nil {
			klog.Warningf("Error remove MTU permission from security group: %q", err)
			return err
		}
		if !changed {
			klog.Warning("Revoking ingress was not needed; concurrent change? groupId=", sgID)
		}

		sgPerms.Delete(permsToRevokeList...)
	}
	if len(permsToGrant) > 0 {
		permsToGrantList := permsToGrant.List()
		changed, err := c.addSecurityGroupIngress(sgID, permsToGrantList)
		if err != nil {
			klog.Warningf("Error add MTU permission to security group: %q", err)
			return err
		}
		if !changed {
			klog.Warning("Allowing ingress was not needed; concurrent change? groupId=", sgID)
		}
		sgPerms.Insert(permsToGrantList...)
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
		tags := getKeyValuePropertiesFromAnnotation(annotations, ServiceAnnotationLoadBalancerAdditionalTags)

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
			additions, removals := syncElbListeners(loadBalancerName, listeners, loadBalancer.ListenerDescriptions)

			if len(removals) != 0 {
				request := &elb.DeleteLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.LoadBalancerPorts = removals
				klog.V(2).Info("Deleting removed load balancer listeners")
				if _, err := c.elb.DeleteLoadBalancerListeners(request); err != nil {
					return nil, fmt.Errorf("error deleting AWS loadbalancer listeners: %q", err)
				}
				dirty = true
			}

			if len(additions) != 0 {
				request := &elb.CreateLoadBalancerListenersInput{}
				request.LoadBalancerName = aws.String(loadBalancerName)
				request.Listeners = additions
				klog.V(2).Info("Creating added load balancer listeners")
				if _, err := c.elb.CreateLoadBalancerListeners(request); err != nil {
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
			tags := getKeyValuePropertiesFromAnnotation(annotations, ServiceAnnotationLoadBalancerAdditionalTags)
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

// syncElbListeners computes a plan to reconcile the desired vs actual state of the listeners on an ELB
// NOTE: there exists an O(nlgn) implementation for this function. However, as the default limit of
//       listeners per elb is 100, this implementation is reduced from O(m*n) => O(n).
func syncElbListeners(loadBalancerName string, listeners []*elb.Listener, listenerDescriptions []*elb.ListenerDescription) ([]*elb.Listener, []*int64) {
	foundSet := make(map[int]bool)
	removals := []*int64{}
	additions := []*elb.Listener{}

	for _, listenerDescription := range listenerDescriptions {
		actual := listenerDescription.Listener
		if actual == nil {
			klog.Warning("Ignoring empty listener in AWS loadbalancer: ", loadBalancerName)
			continue
		}

		found := false
		for i, expected := range listeners {
			if expected == nil {
				klog.Warning("Ignoring empty desired listener for loadbalancer: ", loadBalancerName)
				continue
			}
			if elbListenersAreEqual(actual, expected) {
				// The current listener on the actual
				// elb is in the set of desired listeners.
				foundSet[i] = true
				found = true
				break
			}
		}
		if !found {
			removals = append(removals, actual.LoadBalancerPort)
		}
	}

	for i := range listeners {
		if !foundSet[i] {
			additions = append(additions, listeners[i])
		}
	}

	return additions, removals
}

func elbListenersAreEqual(actual, expected *elb.Listener) bool {
	if !elbProtocolsAreEqual(actual.Protocol, expected.Protocol) {
		return false
	}
	if !elbProtocolsAreEqual(actual.InstanceProtocol, expected.InstanceProtocol) {
		return false
	}
	if aws.Int64Value(actual.InstancePort) != aws.Int64Value(expected.InstancePort) {
		return false
	}
	if aws.Int64Value(actual.LoadBalancerPort) != aws.Int64Value(expected.LoadBalancerPort) {
		return false
	}
	if !awsArnEquals(actual.SSLCertificateId, expected.SSLCertificateId) {
		return false
	}
	return true
}

func createSubnetMappings(subnetIDs []string, allocationIDs []string) []*elbv2.SubnetMapping {
	response := []*elbv2.SubnetMapping{}

	for index, id := range subnetIDs {
		sm := &elbv2.SubnetMapping{SubnetId: aws.String(id)}
		if len(allocationIDs) > 0 {
			sm.AllocationId = aws.String(allocationIDs[index])
		}
		response = append(response, sm)
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
	healthcheck.HealthyThreshold, err = getOrDefault(ServiceAnnotationLoadBalancerHCHealthyThreshold, defaultElbHCHealthyThreshold)
	if err != nil {
		return nil, err
	}
	healthcheck.UnhealthyThreshold, err = getOrDefault(ServiceAnnotationLoadBalancerHCUnhealthyThreshold, defaultElbHCUnhealthyThreshold)
	if err != nil {
		return nil, err
	}
	healthcheck.Timeout, err = getOrDefault(ServiceAnnotationLoadBalancerHCTimeout, defaultElbHCTimeout)
	if err != nil {
		return nil, err
	}
	healthcheck.Interval, err = getOrDefault(ServiceAnnotationLoadBalancerHCInterval, defaultElbHCInterval)
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
	// Override healthcheck protocol, port and path based on annotations
	if s, ok := annotations[ServiceAnnotationLoadBalancerHealthCheckProtocol]; ok {
		protocol = s
	}
	if s, ok := annotations[ServiceAnnotationLoadBalancerHealthCheckPort]; ok && s != defaultHealthCheckPort {
		p, err := strconv.ParseInt(s, 10, 0)
		if err != nil {
			return err
		}
		port = int32(p)
	}
	switch strings.ToUpper(protocol) {
	case "HTTP", "HTTPS":
		if path == "" {
			path = defaultHealthCheckPath
		}
		if s := annotations[ServiceAnnotationLoadBalancerHealthCheckPath]; s != "" {
			path = s
		}
	default:
		path = ""
	}
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
func (c *Cloud) findInstancesForELB(nodes []*v1.Node, annotations map[string]string) (map[InstanceID]*ec2.Instance, error) {

	targetNodes := filterTargetNodes(nodes, annotations)

	// Map to instance ids ignoring Nodes where we cannot find the id (but logging)
	instanceIDs := mapToAWSInstanceIDsTolerant(targetNodes)

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

// filterTargetNodes uses node labels to filter the nodes that should be targeted by the ELB,
// checking if all the labels provided in an annotation are present in the nodes
func filterTargetNodes(nodes []*v1.Node, annotations map[string]string) []*v1.Node {

	targetNodeLabels := getKeyValuePropertiesFromAnnotation(annotations, ServiceAnnotationLoadBalancerTargetNodeLabels)

	if len(targetNodeLabels) == 0 {
		return nodes
	}

	targetNodes := make([]*v1.Node, 0, len(nodes))

	for _, node := range nodes {
		if node.Labels != nil && len(node.Labels) > 0 {
			allFiltersMatch := true

			for targetLabelKey, targetLabelValue := range targetNodeLabels {
				if nodeLabelValue, ok := node.Labels[targetLabelKey]; !ok || (nodeLabelValue != targetLabelValue && targetLabelValue != "") {
					allFiltersMatch = false
					break
				}
			}

			if allFiltersMatch {
				targetNodes = append(targetNodes, node)
			}
		}
	}

	return targetNodes
}
