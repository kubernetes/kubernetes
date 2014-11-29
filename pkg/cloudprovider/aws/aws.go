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

package aws_cloud

import (
	"fmt"
	"io"
	"net"
	"regexp"
	"strings"
	"sync"

	"code.google.com/p/gcfg"
	"github.com/mitchellh/goamz/aws"
	"github.com/mitchellh/goamz/ec2"
	"github.com/mitchellh/goamz/elb"
	"github.com/golang/glog"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/cloudprovider"
)

type EC2 interface {
	Instances(instIds []string, filter *ec2.Filter) (resp *ec2.InstancesResp, err error)
	CreateSecurityGroup(group ec2.SecurityGroup) (resp *ec2.CreateSecurityGroupResp, err error)
	CreateTags(resourceIds []string, tags []ec2.Tag) (resp *ec2.SimpleResp, err error)
	DescribeVpcs(vpcIds []string, filter *ec2.Filter) (resp *ec2.VpcsResp, err error)
	SecurityGroups(groups []ec2.SecurityGroup, filter *ec2.Filter) (resp *ec2.SecurityGroupsResp, err error)
	DescribeSubnets(subnetIds []string, filter *ec2.Filter) (resp *ec2.SubnetsResp, err error)
}

type AwsMetadata interface {
	GetInstanceAz() (string, error)
}

type Filter struct {
	predicates []*FilterPredicate
}

type FilterPredicate struct {
	Name   string
	Values []string
}

func NewFilter() (*Filter) {
	return &Filter{}
}

func (self*Filter) toAws() (filter *ec2.Filter) {
	awsFilter := ec2.NewFilter()
	for _, predicate := range self.predicates {
		awsFilter.Add(predicate.Name, predicate.Values...)
	}
	return awsFilter
}

func (self*Filter) Where(name string, value string) (*Filter) {
	values := []string{value}
	predicate := &FilterPredicate{Name: name, Values: values}
	self.predicates = append(self.predicates, predicate)
	return self
}

//func removeDuplicates(in []string) ([]string) {
//	out := []string{}
//	done := map[string]string{}
//	for _, s := range in {
//		if done[s] != "" {
//			continue
//		}
//		out = append(out, s)
//		done[s] = s
//	}
//	return out
//}

// AWSCloud is an implementation of Interface and Instances for Amazon Web Services.
type AWSCloud struct {
	auth             aws.Auth
	ec2              EC2
	metadata         AwsMetadata
	elbClients map[string]*elb.ELB
	cfg *AWSCloudConfig
	availabilityZone string
	region *aws.Region

	mutex sync.Mutex
}

// awsCloudLoadBalancer is an implementation of TCPLoadBalancer for Amazon Web Services.
type awsCloudLoadBalancer struct {
	awsCloud *AWSCloud
}

// awsZones is an implementation of Zones for Amazon Web Services.
type awsZones struct {
	awsCloud *AWSCloud
}

type AWSCloudConfig struct {
	Global struct {
		Region string
	}
}

type AuthFunc func() (auth aws.Auth, err error)

func init() {
	cloudprovider.RegisterCloudProvider("aws", func(config io.Reader) (cloudprovider.Interface, error) {
			return newAWSCloud(config, &instanceMetadata{}, getAuth)
		})
}

func getAuth() (auth aws.Auth, err error) {
	return aws.GetAuth("", "")
}

// readAWSCloudConfig reads an instance of AWSCloudConfig from config reader.
func readAWSCloudConfig(config io.Reader) (*AWSCloudConfig, error) {
	if config == nil {
		return nil, fmt.Errorf("no AWS cloud provider config file given")
	}

	var cfg AWSCloudConfig
	err := gcfg.ReadInto(&cfg, config)
	if err != nil {
		return nil, err
	}

	if cfg.Global.Region == "" {
		return nil, fmt.Errorf("no region specified in configuration file")
	}

	return &cfg, nil
}

// newAWSCloud creates a new instance of AWSCloud.
func newAWSCloud(config io.Reader, metadata AwsMetadata, authFunc AuthFunc) (*AWSCloud, error) {
	cfg, err := readAWSCloudConfig(config)
	if err != nil {
		return nil, fmt.Errorf("unable to read AWS cloud provider config file: %v", err)
	}

	auth, err := authFunc()
	if err != nil {
		return nil, err
	}

	availabilityZone, err := metadata.GetInstanceAz()
	if err != nil {
		return nil, err
	}

	// TODO: We should just extract this from availabilityZone
	// We can strip the last character/prefix match to get the region
	region, ok := aws.Regions[cfg.Global.Region]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS region: %s", cfg.Global.Region)
	}

	ec2 := ec2.New(auth, region)
	return &AWSCloud{
		auth: auth,
		metadata: &instanceMetadata{},
		availabilityZone: availabilityZone,
		region: &region,
		ec2: ec2,
		cfg: cfg,
		elbClients: map[string]*elb.ELB{},
	}, nil
}

func (aws *AWSCloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// TCPLoadBalancer returns an implementation of TCPLoadBalancer for Amazon Web Services.
func (aws *AWSCloud) TCPLoadBalancer() (cloudprovider.TCPLoadBalancer, bool) {
	lb := &awsCloudLoadBalancer{}
	lb.awsCloud = aws
	return lb, true
}

// Instances returns an implementation of Instances for Amazon Web Services.
func (aws *AWSCloud) Instances() (cloudprovider.Instances, bool) {
	return aws, true
}

// Zones returns an implementation of Zones for Amazon Web Services.
func (aws *AWSCloud) Zones() (cloudprovider.Zones, bool) {
	zones := &awsZones{}
	zones.awsCloud = aws
	return zones, true
}

// IPAddress is an implementation of Instances.IPAddress.
func (aws *AWSCloud) IPAddress(name string) (net.IP, error) {
	f := ec2.NewFilter()
	f.Add("private-dns-name", name)

	resp, err := aws.ec2.Instances(nil, f)
	if err != nil {
		return nil, err
	}
	if len(resp.Reservations) == 0 {
		return nil, fmt.Errorf("no reservations found for host: %s", name)
	}
	if len(resp.Reservations) > 1 {
		return nil, fmt.Errorf("multiple reservations found for host: %s", name)
	}
	if len(resp.Reservations[0].Instances) == 0 {
		return nil, fmt.Errorf("no instances found for host: %s", name)
	}
	if len(resp.Reservations[0].Instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for host: %s", name)
	}

	ipAddress := resp.Reservations[0].Instances[0].PrivateIpAddress
	ip := net.ParseIP(ipAddress)
	if ip == nil {
		return nil, fmt.Errorf("invalid network IP: %s", ipAddress)
	}
	return ip, nil
}

// Return a list of instances matching regex string.
func (aws *AWSCloud) getInstancesByRegex(regex string) ([]string, error) {
	resp, err := aws.ec2.Instances(nil, nil)
	if err != nil {
		return []string{}, err
	}
	if resp == nil {
		return []string{}, fmt.Errorf("no InstanceResp returned")
	}

	re, err := regexp.Compile(regex)
	if err != nil {
		return []string{}, err
	}

	instances := []string{}
	for _, reservation := range resp.Reservations {
		for _, instance := range reservation.Instances {
			for _, tag := range instance.Tags {
				if tag.Key == "Name" && re.MatchString(tag.Value) {
					instances = append(instances, instance.PrivateDNSName)
					break
				}
			}
		}
	}
	return instances, nil
}

// List is an implementation of Instances.List.
func (aws *AWSCloud) List(filter string) ([]string, error) {
	// TODO: Should really use tag query. No need to go regexp.
	return aws.getInstancesByRegex(filter)
}

func (v *AWSCloud) GetNodeResources(name string) (*api.NodeResources, error) {
	return nil, nil
}

// Builds an ELB client
func (self *AWSCloud) getElbClient(regionName string) (*elb.ELB, error) {
	self.mutex.Lock()
	defer self.mutex.Unlock()

	region, ok := aws.Regions[regionName]
	if !ok {
		return nil, fmt.Errorf("not a valid AWS region: %s", regionName)
	}
	elbClient, found := self.elbClients[region.Name]
	if !found {
		elbClient = elb.New(self.auth, region)
		self.elbClients[region.Name] = elbClient
	}
	return elbClient, nil
}

// Find the kubernetes vpc
func (self *AWSCloud) findVpc() (*ec2.VPC, error) {
	client := self.ec2

	// TODO: How do we want to identify our VPC?
	filter := NewFilter().Where("tag:Name", "kubernetes-vpc")

	ids := []string{}
	response, err := client.DescribeVpcs(ids, filter.toAws())
	if err != nil {
		glog.Error("error listing VPCs", err)
		return nil, err
	}

	vpcs := response.VPCs
	if len(vpcs) == 0 {
		return nil, nil
	}
	if len(vpcs) == 1 {
		return &vpcs[0], nil
	}

	glog.Warning("Found multiple VPCs; picking arbitrarily", vpcs)
	return &vpcs[0], nil
}

// Find the EC2 instance for a given k8s name
// TODO: This is a _bad_ idea.  The internal ID can be recycled, for one.
// We should the AWS instance id
func (self *AWSCloud) findInstance(privateDnsName string) (*ec2.Instance, error) {
	client := self.ec2

	filter := NewFilter().Where("private-dns-name", privateDnsName)

	response, err := client.Instances(nil, filter.toAws())
	if err != nil {
		glog.Error("error listing instances", err)
		return nil, err
	}

	for _, reservation := range response.Reservations {
		for _, instance := range reservation.Instances {
			return &instance, nil
		}
	}
	return nil, nil
}

func (self *AWSCloud) describeSubnets(subnetIds []string, filter *Filter) (*ec2.SubnetsResp, error) {
	client := self.ec2

	subnets, err := client.DescribeSubnets(subnetIds, filter.toAws())
	if err != nil {
		glog.Error("error listing subnets", err)
		return nil, err
	}

	return subnets, nil
}

// Builds an ELB client
func (self *awsCloudLoadBalancer) getElbClient(region string) (*elb.ELB, error) {
	return self.awsCloud.getElbClient(region)
}

// Gets the current load balancer state
func (self *awsCloudLoadBalancer) describeLoadBalancer(client *elb.ELB, name string) (*elb.LoadBalancer, error) {
	request := &elb.DescribeLoadBalancer{}
	request.Names = []string { name }
	response, err := client.DescribeLoadBalancers(request)
	if err != nil {
		return nil, err
	}

	for _, loadBalancer := range response.LoadBalancers {
		return &loadBalancer, nil
	}
	return nil, nil
}

// TCPLoadBalancerExists is an implementation of TCPLoadBalancer.TCPLoadBalancerExists.
func (self *awsCloudLoadBalancer) TCPLoadBalancerExists(name, region string) (bool, error) {
	client, err := self.getElbClient(region)
	if err != nil {
		return false, err
	}

	lb, err := self.describeLoadBalancer(client, name)
	if err != nil {
		return false, err
	}

	if lb != nil {
		return true, nil
	}
	return false, nil
}

// Create a security group that will be used with the load balancer
func (self *AWSCloud) createSecurityGroup(vpcId, name, description string) (string, error) {
	client := self.ec2

	request := ec2.SecurityGroup{}
	request.VpcId = vpcId
	request.Name = name
	request.Description = description
	response, err := client.CreateSecurityGroup(request)
	if err != nil {
		return "", err
	}

	return response.Id, nil
}

// Finds security groups
func (self *AWSCloud) findSecurityGroups(filter *Filter) ([]ec2.SecurityGroupInfo, error) {
	client := self.ec2
	response, err := client.SecurityGroups([]ec2.SecurityGroup{}, filter.toAws())
	if err != nil {
		return nil, err
	}

	return response.Groups, nil
}

func (self *AWSCloud) createTags(resourceId string, tags []ec2.Tag) (error) {
	client := self.ec2

	resourceIds := []string { resourceId }

	_, err := client.CreateTags(resourceIds, tags)
	if err != nil {
		return err
	}

	return nil
}

func (self*awsCloudLoadBalancer) hostsToInstances(hosts []string) ([]*ec2.Instance, error) {
	instances := []*ec2.Instance{}
	for _, host := range hosts {
		instance, err := self.awsCloud.findInstance(host)
		if err != nil {
			return nil, err
		}
		if instance == nil {
			return nil, fmt.Errorf("unable to find instance "+host)
		}
		instances = append(instances, instance)
	}
	return instances, nil
}

func mapToInstanceIds(instances []*ec2.Instance) ([]string) {
	ids := []string{}
	for _, instance := range instances {
		ids = append(ids, instance.InstanceId)
	}
	return ids
}

// CreateTCPLoadBalancer is an implementation of TCPLoadBalancer.CreateTCPLoadBalancer.
func (self *awsCloudLoadBalancer) CreateTCPLoadBalancer(name, region string, externalIP net.IP, port int, hosts []string) (string, error) {
	instances, err := self.hostsToInstances(hosts)
	if err != nil {
		return "", err
	}

	client, err := self.getElbClient(region)
	if err != nil {
		return "", err
	}

	vpc, err := self.awsCloud.findVpc()
	if err != nil {
		glog.Error("error finding vpc", err)
		return "", err
	}

	if vpc == nil {
		return "", fmt.Errorf("Unable to find vpc")
	}


	subnetIds := []string{}
	{
		subnets, err := self.awsCloud.describeSubnets(nil, NewFilter().Where("vpc-id", vpc.VpcId))
		if err != nil {
			glog.Error("error listing subnets", err)
			return "", err
		}

		//	zones := []string{}
		for _, subnet := range subnets.Subnets {
			subnetIds = append(subnetIds, subnet.SubnetId)
			if !strings.HasPrefix(subnet.AvailabilityZone, region) {
				glog.Error("found AZ that did not match region", subnet.AvailabilityZone, " vs ", region)
				return "", fmt.Errorf("invalid AZ for region")
			}
			//		zones = append(zones, subnet.AvailabilityZone)
		}
	}

	var loadBalancerName, dnsName string
	{
		request := &elb.DescribeLoadBalancer{}
		request.Names = []string{ name }
		response, err := client.DescribeLoadBalancers(request)
		if err != nil {
			glog.Error("error finding load balancer", err)
			return "", err
		}

		var loadBalancer *elb.LoadBalancer
		for _, lb := range response.LoadBalancers {
			//assert loadBalancer == nil
			loadBalancer = &lb
		}

		if loadBalancer == nil {
			createRequest := &elb.CreateLoadBalancer{}
			createRequest.LoadBalancerName = name

			listener := elb.Listener{}
			listener.InstancePort = int64(port)
			listener.LoadBalancerPort = int64(port)
			listener.Protocol = "tcp"
			listener.InstanceProtocol = "tcp"
			createRequest.Listeners = []elb.Listener{ listener }
			//	nameTag := &elb.Tag{ Key: "Name", Value: name}
			//	createRequest.Tags = []Tag { nameTag }

			//	zones := []string{"us-east-1a"}
			//	createRequest.AvailZone = removeDuplicates(zones)

			// We are supposed to specify one subnet per AZ.
			// TODO: What happens if we have more than one subnet per AZ?
			createRequest.Subnets = subnetIds

			sgName := "k8s-elb-" + name
			sgDescription := "Security group for Kubernetes ELB " + name

			{
				filter := NewFilter().Where("vpc-id", vpc.VpcId).Where("group-name", sgName)
				// TODO: Should we do something more reliable ?? .Where("tag:kubernetes-id", kubernetesId)
				securityGroups, err := self.awsCloud.findSecurityGroups(filter)
				if err != nil {
					return "", err
				}
				var securityGroupId string
				for _, securityGroup := range securityGroups {
					securityGroupId = securityGroup.Id
				}
				if securityGroupId == "" {
					securityGroupId, err = self.awsCloud.createSecurityGroup(vpc.VpcId, sgName, sgDescription)
					if err != nil {
						return "", err
					}
				}
				createRequest.SecurityGroups = []string { securityGroupId }
			}

			if len(externalIP) > 0 {
				return "", fmt.Errorf("External IP cannot be specified for AWS ELB")
			}

			createResponse, err := client.CreateLoadBalancer(createRequest)
			if err != nil {
				return "", err
			}

			dnsName = createResponse.DNSName
			loadBalancerName = name
		} else {
			// TODO: Verify that load balancer configuration matches?

			loadBalancerName = loadBalancer.LoadBalancerName
			dnsName = loadBalancer.DNSName
		}
	}

	registerRequest := &elb.RegisterInstancesWithLoadBalancer{}
	registerRequest.LoadBalancerName = loadBalancerName
	registerRequest.Instances = mapToInstanceIds(instances)

	registerResponse, err := client.RegisterInstancesWithLoadBalancer(registerRequest)
	if err != nil {
		return "", err
	}

	glog.V(1).Info("Updated instances registered with load-balancer", name, registerResponse.Instances)

	// TODO: Wait for creation?

	return dnsName, nil
}

// UpdateTCPLoadBalancer is an implementation of TCPLoadBalancer.UpdateTCPLoadBalancer.
func (self *awsCloudLoadBalancer) UpdateTCPLoadBalancer(name, region string, hosts []string) error {
	instances, err := self.hostsToInstances(hosts)
	if err != nil {
		return err
	}

	client, err := self.getElbClient(region)
	if err != nil {
		return err
	}

	lb, err := self.describeLoadBalancer(client, name)
	if err != nil {
		return err
	}

	if lb == nil {
		return fmt.Errorf("Load balancer not found")
	}

	existingInstances := map[string]*elb.Instance{}
	for _, instance := range lb.Instances {
		existingInstances[instance.InstanceId] = &instance
	}

	wantInstances := map[string]*ec2.Instance{}
	for _, instance := range instances {
		wantInstances[instance.InstanceId] = instance
	}

	addInstances := []string{}
	for key, _ := range wantInstances {
		_, found := existingInstances[key]
		if !found {
			addInstances = append(addInstances, key)
		}
	}

	removeInstances := []string{}
	for key, _ := range existingInstances {
		_, found := wantInstances[key]
		if !found {
			removeInstances = append(removeInstances, key)
		}
	}

	if len(addInstances) > 0 {
		registerRequest := &elb.RegisterInstancesWithLoadBalancer{}
		registerRequest.Instances = addInstances
		registerRequest.LoadBalancerName = lb.LoadBalancerName
		_, err = client.RegisterInstancesWithLoadBalancer(registerRequest)
		if err != nil {
			return err
		}
	}

	if len(removeInstances) > 0 {
		deregisterRequest := &elb.DeregisterInstancesFromLoadBalancer{}
		deregisterRequest.Instances = removeInstances
		deregisterRequest.LoadBalancerName = lb.LoadBalancerName
		_, err = client.DeregisterInstancesFromLoadBalancer(deregisterRequest)
		if err != nil {
			return err
		}
	}

	return nil
}

// DeleteTCPLoadBalancer is an implementation of TCPLoadBalancer.DeleteTCPLoadBalancer.
func (self *awsCloudLoadBalancer) DeleteTCPLoadBalancer(name, region string) error {
	client, err := self.getElbClient(region)
	if err != nil {
		return err
	}

	request := &elb.DeleteLoadBalancer{}
	request.LoadBalancerName = name
	_, err = client.DeleteLoadBalancer(request)
	if err != nil {
		return err
	}
	return nil
}

type instanceMetadata struct {
}

func (self*instanceMetadata) GetInstanceAz() (az string, err error) {
	path := "placement/availability-zone"

	data, err := aws.GetMetaData(path)
	if err != nil {
		glog.Error("unable to fetch az from EC2 metadata", err)
		return "", err
	}

	return string(data), nil
}


func (self *awsZones) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: self.awsCloud.availabilityZone,
		Region:        self.awsCloud.region.Name,
	}, nil
}
