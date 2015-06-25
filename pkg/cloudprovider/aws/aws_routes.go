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
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/service/ec2"
)

func (s *AWSCloud) findRouteTable(clusterName string) (*ec2.RouteTable, error) {
	// This should be unnecessary (we already filter on TagNameKubernetesCluster,
	// and something is broken if cluster name doesn't match, but anyway...
	// TODO: All clouds should be cluster-aware by default
	filters := []*ec2.Filter{newEc2Filter("tag:"+TagNameKubernetesCluster, clusterName)}
	request := &ec2.DescribeRouteTablesInput{Filters: s.addFilters(filters)}

	tables, err := s.ec2.DescribeRouteTables(request)
	if err != nil {
		return nil, err
	}

	if len(tables) == 0 {
		return nil, fmt.Errorf("unable to find route table for AWS cluster: %s", clusterName)
	}

	if len(tables) != 1 {
		return nil, fmt.Errorf("found multiple matching AWS route tables for AWS cluster: %s", clusterName)
	}
	return tables[0], nil
}

// ListRoutes implements Routes.ListRoutes
// List all routes that match the filter
func (s *AWSCloud) ListRoutes(clusterName string) ([]*cloudprovider.Route, error) {
	table, err := s.findRouteTable(clusterName)
	if err != nil {
		return nil, err
	}

	var routes []*cloudprovider.Route
	for _, r := range table.Routes {
		instanceID := orEmpty(r.InstanceID)
		destinationCIDR := orEmpty(r.DestinationCIDRBlock)

		if instanceID == "" || destinationCIDR == "" {
			continue
		}

		routeName := clusterName + "-" + destinationCIDR
		routes = append(routes, &cloudprovider.Route{routeName, instanceID, destinationCIDR})
	}

	return routes, nil
}

// Sets the instance attribute "source-dest-check" to the specified value
func (s *AWSCloud) configureInstanceSourceDestCheck(instanceID string, sourceDestCheck bool) error {
	request := &ec2.ModifyInstanceAttributeInput{}
	request.InstanceID = aws.String(instanceID)
	request.SourceDestCheck = &ec2.AttributeBooleanValue{Value: aws.Boolean(sourceDestCheck)}

	_, err := s.ec2.ModifyInstanceAttribute(request)
	if err != nil {
		return fmt.Errorf("error configuring source-dest-check on instance %s: %v", instanceID, err)
	}
	return nil
}

// CreateRoute implements Routes.CreateRoute
// Create the described route
func (s *AWSCloud) CreateRoute(clusterName string, nameHint string, route *cloudprovider.Route) error {
	// In addition to configuring the route itself, we also need to configure the instance to accept that traffic
	// On AWS, this requires turning source-dest checks off
	err := s.configureInstanceSourceDestCheck(route.TargetInstance, false)
	if err != nil {
		return err
	}

	table, err := s.findRouteTable(clusterName)
	if err != nil {
		return err
	}

	request := &ec2.CreateRouteInput{}
	// TODO: use ClientToken for idempotency?
	request.DestinationCIDRBlock = aws.String(route.DestinationCIDR)
	request.InstanceID = aws.String(route.TargetInstance)
	request.RouteTableID = table.RouteTableID

	_, err = s.ec2.CreateRoute(request)
	if err != nil {
		return fmt.Errorf("error creating AWS route (%s): %v", route.DestinationCIDR, err)
	}

	return nil
}

// DeleteRoute implements Routes.DeleteRoute
// Delete the specified route
func (s *AWSCloud) DeleteRoute(clusterName string, route *cloudprovider.Route) error {
	table, err := s.findRouteTable(clusterName)
	if err != nil {
		return err
	}

	request := &ec2.DeleteRouteInput{}
	request.DestinationCIDRBlock = aws.String(route.DestinationCIDR)
	request.RouteTableID = table.RouteTableID

	_, err = s.ec2.DeleteRoute(request)
	if err != nil {
		return fmt.Errorf("error deleting AWS route (%s): %v", route.DestinationCIDR, err)
	}

	return nil
}
