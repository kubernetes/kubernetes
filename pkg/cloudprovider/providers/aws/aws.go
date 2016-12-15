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
	"errors"
	"fmt"
	"io"
	"net"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"time"

	"gopkg.in/gcfg.v1"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/golang/glog"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/service"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/cloudprovider"
	aws_credentials "k8s.io/kubernetes/pkg/credentialprovider/aws"
	"k8s.io/kubernetes/pkg/types"
	"k8s.io/kubernetes/pkg/util/sets"
	"k8s.io/kubernetes/pkg/volume"
)

// ProviderName is the name of this cloud provider.
const ProviderName = "aws"

// TagNameKubernetesCluster is the tag name we use to differentiate multiple
// logically independent clusters running in the same AZ
const TagNameKubernetesCluster = "KubernetesCluster"

// TagNameKubernetesService is the tag name we use to differentiate multiple
// services. Used currently for ELBs only.
const TagNameKubernetesService = "kubernetes.io/service-name"

// TagNameSubnetInternalELB is the tag name used on a subnet to designate that
// it should be used for internal ELBs
const TagNameSubnetInternalELB = "kubernetes.io/role/internal-elb"

// TagNameSubnetPublicELB is the tag name used on a subnet to designate that
// it should be used for internet ELBs
const TagNameSubnetPublicELB = "kubernetes.io/role/elb"

// ServiceAnnotationLoadBalancerInternal is the annotation used on the service
// to indicate that we want an internal ELB.
// Currently we accept only the value "0.0.0.0/0" - other values are an error.
// This lets us define more advanced semantics in future.
const ServiceAnnotationLoadBalancerInternal = "service.beta.kubernetes.io/aws-load-balancer-internal"

// ServiceAnnotationLoadBalancerProxyProtocol is the annotation used on the
// service to enable the proxy protocol on an ELB. Right now we only accept the
// value "*" which means enable the proxy protocol on all ELB backends. In the
// future we could adjust this to allow setting the proxy protocol only on
// certain backends.
const ServiceAnnotationLoadBalancerProxyProtocol = "service.beta.kubernetes.io/aws-load-balancer-proxy-protocol"

// ServiceAnnotationLoadBalancerAccessLogEmitInterval is the annotation used to
// specify access log emit interval.
const ServiceAnnotationLoadBalancerAccessLogEmitInterval = "service.beta.kubernetes.io/aws-load-balancer-access-log-emit-interval"

// ServiceAnnotationLoadBalancerAccessLogEnabled is the annotation used on the
// service to enable or disable access logs.
const ServiceAnnotationLoadBalancerAccessLogEnabled = "service.beta.kubernetes.io/aws-load-balancer-access-log-enabled"

// ServiceAnnotationLoadBalancerAccessLogS3BucketName is the annotation used to
// specify access log s3 bucket name.
const ServiceAnnotationLoadBalancerAccessLogS3BucketName = "service.beta.kubernetes.io/aws-load-balancer-access-log-s3-bucket-name"

// ServiceAnnotationLoadBalancerAccessLogS3BucketPrefix is the annotation used
// to specify access log s3 bucket prefix.
const ServiceAnnotationLoadBalancerAccessLogS3BucketPrefix = "service.beta.kubernetes.io/aws-load-balancer-access-log-s3-bucket-prefix"

// ServiceAnnotationLoadBalancerConnectionDrainingEnabled is the annnotation
// used on the service to enable or disable connection draining.
const ServiceAnnotationLoadBalancerConnectionDrainingEnabled = "service.beta.kubernetes.io/aws-load-balancer-connection-draining-enabled"

// ServiceAnnotationLoadBalancerConnectionDrainingTimeout is the annotation
// used on the service to specify a connection draining timeout.
const ServiceAnnotationLoadBalancerConnectionDrainingTimeout = "service.beta.kubernetes.io/aws-load-balancer-connection-draining-timeout"

// ServiceAnnotationLoadBalancerConnectionIdleTimeout is the annotation used
// on the service to specify the idle connection timeout.
const ServiceAnnotationLoadBalancerConnectionIdleTimeout = "service.beta.kubernetes.io/aws-load-balancer-connection-idle-timeout"

// ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled is the annotation
// used on the service to enable or disable cross-zone load balancing.
const ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled = "service.beta.kubernetes.io/aws-load-balancer-cross-zone-load-balancing-enabled"

// ServiceAnnotationLoadBalancerCertificate is the annotation used on the
// service to request a secure listener. Value is a valid certificate ARN.
// For more, see http://docs.aws.amazon.com/ElasticLoadBalancing/latest/DeveloperGuide/elb-listener-config.html
// CertARN is an IAM or CM certificate ARN, e.g. arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012
const ServiceAnnotationLoadBalancerCertificate = "service.beta.kubernetes.io/aws-load-balancer-ssl-cert"

// ServiceAnnotationLoadBalancerSSLPorts is the annotation used on the service
// to specify a comma-separated list of ports that will use SSL/HTTPS
// listeners. Defaults to '*' (all).
const ServiceAnnotationLoadBalancerSSLPorts = "service.beta.kubernetes.io/aws-load-balancer-ssl-ports"

// ServiceAnnotationLoadBalancerBEProtocol is the annotation used on the service
// to specify the protocol spoken by the backend (pod) behind a listener.
// If `http` (default) or `https`, an HTTPS listener that terminates the
//  connection and parses headers is created.
// If set to `ssl` or `tcp`, a "raw" SSL listener is used.
// If set to `http` and `aws-load-balancer-ssl-cert` is not used then
// a HTTP listener is used.
const ServiceAnnotationLoadBalancerBEProtocol = "service.beta.kubernetes.io/aws-load-balancer-backend-protocol"

const (
	// volumeAttachmentStatusTimeout is the maximum time to wait for a volume attach/detach to complete
	volumeAttachmentStatusTimeout = 30 * time.Minute
	// volumeAttachmentConsecutiveErrorLimit is the number of consecutive errors we will ignore when waiting for a volume to attach/detach
	volumeAttachmentStatusConsecutiveErrorLimit = 10
	// volumeAttachmentErrorDelay is the amount of time we wait before retrying after encountering an error,
	// while waiting for a volume attach/detach to complete
	volumeAttachmentStatusErrorDelay = 20 * time.Second
	// volumeAttachmentStatusPollInterval is the interval at which we poll the volume,
	// while waiting for a volume attach/detach to complete
	volumeAttachmentStatusPollInterval = 10 * time.Second
)

// Maps from backend protocol to ELB protocol
var backendProtocolMapping = map[string]string{
	"https": "https",
	"http":  "https",
	"ssl":   "ssl",
	"tcp":   "ssl",
}

// MaxReadThenCreateRetries sets the maximum number of attempts we will make when
// we read to see if something exists and then try to create it if we didn't find it.
// This can fail once in a consistent system if done in parallel
// In an eventually consistent system, it could fail unboundedly
const MaxReadThenCreateRetries = 30

// DefaultVolumeType specifies which storage to use for newly created Volumes
// TODO: Remove when user/admin can configure volume types and thus we don't
// need hardcoded defaults.
const DefaultVolumeType = "gp2"

// DefaultMaxEBSVolumes is the limit for volumes attached to an instance.
// Amazon recommends no more than 40; the system root volume uses at least one.
// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/volume_limits.html#linux-specific-volume-limits
const DefaultMaxEBSVolumes = 39

// Used to call aws_credentials.Init() just once
var once sync.Once

// Services is an abstraction over AWS, to allow mocking/other implementations
type Services interface {
	Compute(region string) (EC2, error)
	LoadBalancing(region string) (ELB, error)
	Autoscaling(region string) (ASG, error)
	Metadata() (EC2Metadata, error)
}

// EC2 is an abstraction over AWS', to allow mocking/other implementations
// Note that the DescribeX functions return a list, so callers don't need to deal with paging
// TODO: Should we rename this to AWS (EBS & ELB are not technically part of EC2)
type EC2 interface {
	// Query EC2 for instances matching the filter
	DescribeInstances(request *ec2.DescribeInstancesInput) ([]*ec2.Instance, error)

	// Attach a volume to an instance
	AttachVolume(*ec2.AttachVolumeInput) (*ec2.VolumeAttachment, error)
	// Detach a volume from an instance it is attached to
	DetachVolume(request *ec2.DetachVolumeInput) (resp *ec2.VolumeAttachment, err error)
	// Lists volumes
	DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error)
	// Create an EBS volume
	CreateVolume(request *ec2.CreateVolumeInput) (resp *ec2.Volume, err error)
	// Delete an EBS volume
	DeleteVolume(*ec2.DeleteVolumeInput) (*ec2.DeleteVolumeOutput, error)

	DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error)

	CreateSecurityGroup(*ec2.CreateSecurityGroupInput) (*ec2.CreateSecurityGroupOutput, error)
	DeleteSecurityGroup(request *ec2.DeleteSecurityGroupInput) (*ec2.DeleteSecurityGroupOutput, error)

	AuthorizeSecurityGroupIngress(*ec2.AuthorizeSecurityGroupIngressInput) (*ec2.AuthorizeSecurityGroupIngressOutput, error)
	RevokeSecurityGroupIngress(*ec2.RevokeSecurityGroupIngressInput) (*ec2.RevokeSecurityGroupIngressOutput, error)

	DescribeSubnets(*ec2.DescribeSubnetsInput) ([]*ec2.Subnet, error)

	CreateTags(*ec2.CreateTagsInput) (*ec2.CreateTagsOutput, error)

	DescribeRouteTables(request *ec2.DescribeRouteTablesInput) ([]*ec2.RouteTable, error)
	CreateRoute(request *ec2.CreateRouteInput) (*ec2.CreateRouteOutput, error)
	DeleteRoute(request *ec2.DeleteRouteInput) (*ec2.DeleteRouteOutput, error)

	ModifyInstanceAttribute(request *ec2.ModifyInstanceAttributeInput) (*ec2.ModifyInstanceAttributeOutput, error)
}

// ELB is a simple pass-through of AWS' ELB client interface, which allows for testing
type ELB interface {
	CreateLoadBalancer(*elb.CreateLoadBalancerInput) (*elb.CreateLoadBalancerOutput, error)
	DeleteLoadBalancer(*elb.DeleteLoadBalancerInput) (*elb.DeleteLoadBalancerOutput, error)
	DescribeLoadBalancers(*elb.DescribeLoadBalancersInput) (*elb.DescribeLoadBalancersOutput, error)
	RegisterInstancesWithLoadBalancer(*elb.RegisterInstancesWithLoadBalancerInput) (*elb.RegisterInstancesWithLoadBalancerOutput, error)
	DeregisterInstancesFromLoadBalancer(*elb.DeregisterInstancesFromLoadBalancerInput) (*elb.DeregisterInstancesFromLoadBalancerOutput, error)
	CreateLoadBalancerPolicy(*elb.CreateLoadBalancerPolicyInput) (*elb.CreateLoadBalancerPolicyOutput, error)
	SetLoadBalancerPoliciesForBackendServer(*elb.SetLoadBalancerPoliciesForBackendServerInput) (*elb.SetLoadBalancerPoliciesForBackendServerOutput, error)

	DetachLoadBalancerFromSubnets(*elb.DetachLoadBalancerFromSubnetsInput) (*elb.DetachLoadBalancerFromSubnetsOutput, error)
	AttachLoadBalancerToSubnets(*elb.AttachLoadBalancerToSubnetsInput) (*elb.AttachLoadBalancerToSubnetsOutput, error)

	CreateLoadBalancerListeners(*elb.CreateLoadBalancerListenersInput) (*elb.CreateLoadBalancerListenersOutput, error)
	DeleteLoadBalancerListeners(*elb.DeleteLoadBalancerListenersInput) (*elb.DeleteLoadBalancerListenersOutput, error)

	ApplySecurityGroupsToLoadBalancer(*elb.ApplySecurityGroupsToLoadBalancerInput) (*elb.ApplySecurityGroupsToLoadBalancerOutput, error)

	ConfigureHealthCheck(*elb.ConfigureHealthCheckInput) (*elb.ConfigureHealthCheckOutput, error)

	DescribeLoadBalancerAttributes(*elb.DescribeLoadBalancerAttributesInput) (*elb.DescribeLoadBalancerAttributesOutput, error)
	ModifyLoadBalancerAttributes(*elb.ModifyLoadBalancerAttributesInput) (*elb.ModifyLoadBalancerAttributesOutput, error)
}

// ASG is a simple pass-through of the Autoscaling client interface, which
// allows for testing.
type ASG interface {
	UpdateAutoScalingGroup(*autoscaling.UpdateAutoScalingGroupInput) (*autoscaling.UpdateAutoScalingGroupOutput, error)
	DescribeAutoScalingGroups(*autoscaling.DescribeAutoScalingGroupsInput) (*autoscaling.DescribeAutoScalingGroupsOutput, error)
}

// EC2Metadata is an abstraction over the AWS metadata service.
type EC2Metadata interface {
	// Query the EC2 metadata service (used to discover instance-id etc)
	GetMetadata(path string) (string, error)
}

// AWS volume types
const (
	// Provisioned IOPS SSD
	VolumeTypeIO1 = "io1"
	// General Purpose SSD
	VolumeTypeGP2 = "gp2"
	// Cold HDD (sc1)
	VolumeTypeSC1 = "sc1"
	// Throughput Optimized HDD
	VolumeTypeST1 = "st1"
)

// AWS provisioning limits.
// Source: http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/EBSVolumeTypes.html
const (
	MinTotalIOPS = 100
	MaxTotalIOPS = 20000
)

// VolumeOptions specifies capacity and tags for a volume.
type VolumeOptions struct {
	CapacityGB       int
	Tags             map[string]string
	PVCName          string
	VolumeType       string
	AvailabilityZone string
	// IOPSPerGB x CapacityGB will give total IOPS of the volume to create.
	// Calculated total IOPS will be capped at MaxTotalIOPS.
	IOPSPerGB int
	Encrypted bool
	// fully qualified resource name to the key to use for encryption.
	// example: arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123-456a-a12b-a123b4cd56ef
	KmsKeyId string
}

// Volumes is an interface for managing cloud-provisioned volumes
// TODO: Allow other clouds to implement this
type Volumes interface {
	// Attach the disk to the node with the specified NodeName
	// nodeName can be empty to mean "the instance on which we are running"
	// Returns the device (e.g. /dev/xvdf) where we attached the volume
	AttachDisk(diskName KubernetesVolumeID, nodeName types.NodeName, readOnly bool) (string, error)
	// Detach the disk from the node with the specified NodeName
	// nodeName can be empty to mean "the instance on which we are running"
	// Returns the device where the volume was attached
	DetachDisk(diskName KubernetesVolumeID, nodeName types.NodeName) (string, error)

	// Create a volume with the specified options
	CreateDisk(volumeOptions *VolumeOptions) (volumeName KubernetesVolumeID, err error)
	// Delete the specified volume
	// Returns true iff the volume was deleted
	// If the was not found, returns (false, nil)
	DeleteDisk(volumeName KubernetesVolumeID) (bool, error)

	// Get labels to apply to volume on creation
	GetVolumeLabels(volumeName KubernetesVolumeID) (map[string]string, error)

	// Get volume's disk path from volume name
	// return the device path where the volume is attached
	GetDiskPath(volumeName KubernetesVolumeID) (string, error)

	// Check if the volume is already attached to the node with the specified NodeName
	DiskIsAttached(diskName KubernetesVolumeID, nodeName types.NodeName) (bool, error)

	// Check if a list of volumes are attached to the node with the specified NodeName
	DisksAreAttached(diskNames []KubernetesVolumeID, nodeName types.NodeName) (map[KubernetesVolumeID]bool, error)
}

// InstanceGroups is an interface for managing cloud-managed instance groups / autoscaling instance groups
// TODO: Allow other clouds to implement this
type InstanceGroups interface {
	// Set the size to the fixed size
	ResizeInstanceGroup(instanceGroupName string, size int) error
	// Queries the cloud provider for information about the specified instance group
	DescribeInstanceGroup(instanceGroupName string) (InstanceGroupInfo, error)
}

// InstanceGroupInfo is returned by InstanceGroups.Describe, and exposes information about the group.
type InstanceGroupInfo interface {
	// The number of instances currently running under control of this group
	CurrentSize() (int, error)
}

// Cloud is an implementation of Interface, LoadBalancer and Instances for Amazon Web Services.
type Cloud struct {
	ec2      EC2
	elb      ELB
	asg      ASG
	metadata EC2Metadata
	cfg      *CloudConfig
	region   string
	vpcID    string

	filterTags map[string]string

	// The AWS instance that we are running on
	// Note that we cache some state in awsInstance (mountpoints), so we must preserve the instance
	selfAWSInstance *awsInstance

	mutex                    sync.Mutex
	lastNodeNames            sets.String
	lastInstancesByNodeNames []*ec2.Instance

	// We keep an active list of devices we have assigned but not yet
	// attached, to avoid a race condition where we assign a device mapping
	// and then get a second request before we attach the volume
	attachingMutex sync.Mutex
	attaching      map[types.NodeName]map[mountDevice]awsVolumeID

	// state of our device allocator for each node
	deviceAllocators map[types.NodeName]DeviceAllocator
}

var _ Volumes = &Cloud{}

// CloudConfig wraps the settings for the AWS cloud provider.
type CloudConfig struct {
	Global struct {
		// TODO: Is there any use for this?  We can get it from the instance metadata service
		// Maybe if we're not running on AWS, e.g. bootstrap; for now it is not very useful
		Zone string

		KubernetesClusterTag string

		//The aws provider creates an inbound rule per load balancer on the node security
		//group. However, this can run into the AWS security group rule limit of 50 if
		//many LoadBalancers are created.
		//
		//This flag disables the automatic ingress creation. It requires that the user
		//has setup a rule that allows inbound traffic on kubelet ports from the
		//local VPC subnet (so load balancers can access it). E.g. 10.82.0.0/16 30000-32000.
		DisableSecurityGroupIngress bool

		//During the instantiation of an new AWS cloud provider, the detected region
		//is validated against a known set of regions.
		//
		//In a non-standard, AWS like environment (e.g. Eucalyptus), this check may
		//be undesirable.  Setting this to true will disable the check and provide
		//a warning that the check was skipped.  Please note that this is an
		//experimental feature and work-in-progress for the moment.  If you find
		//yourself in an non-AWS cloud and open an issue, please indicate that in the
		//issue body.
		DisableStrictZoneCheck bool
	}
}

// awsSdkEC2 is an implementation of the EC2 interface, backed by aws-sdk-go
type awsSdkEC2 struct {
	ec2 *ec2.EC2
}

type awsSDKProvider struct {
	creds *credentials.Credentials

	mutex          sync.Mutex
	regionDelayers map[string]*CrossRequestRetryDelay
}

func newAWSSDKProvider(creds *credentials.Credentials) *awsSDKProvider {
	return &awsSDKProvider{
		creds:          creds,
		regionDelayers: make(map[string]*CrossRequestRetryDelay),
	}
}

func (p *awsSDKProvider) addHandlers(regionName string, h *request.Handlers) {
	h.Sign.PushFrontNamed(request.NamedHandler{
		Name: "k8s/logger",
		Fn:   awsHandlerLogger,
	})

	delayer := p.getCrossRequestRetryDelay(regionName)
	if delayer != nil {
		h.Sign.PushFrontNamed(request.NamedHandler{
			Name: "k8s/delay-presign",
			Fn:   delayer.BeforeSign,
		})

		h.AfterRetry.PushFrontNamed(request.NamedHandler{
			Name: "k8s/delay-afterretry",
			Fn:   delayer.AfterRetry,
		})
	}
}

// Get a CrossRequestRetryDelay, scoped to the region, not to the request.
// This means that when we hit a limit on a call, we will delay _all_ calls to the API.
// We do this to protect the AWS account from becoming overloaded and effectively locked.
// We also log when we hit request limits.
// Note that this delays the current goroutine; this is bad behaviour and will
// likely cause k8s to become slow or unresponsive for cloud operations.
// However, this throttle is intended only as a last resort.  When we observe
// this throttling, we need to address the root cause (e.g. add a delay to a
// controller retry loop)
func (p *awsSDKProvider) getCrossRequestRetryDelay(regionName string) *CrossRequestRetryDelay {
	p.mutex.Lock()
	defer p.mutex.Unlock()

	delayer, found := p.regionDelayers[regionName]
	if !found {
		delayer = NewCrossRequestRetryDelay()
		p.regionDelayers[regionName] = delayer
	}
	return delayer
}

func (p *awsSDKProvider) Compute(regionName string) (EC2, error) {
	service := ec2.New(session.New(&aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}))

	p.addHandlers(regionName, &service.Handlers)

	ec2 := &awsSdkEC2{
		ec2: service,
	}
	return ec2, nil
}

func (p *awsSDKProvider) LoadBalancing(regionName string) (ELB, error) {
	elbClient := elb.New(session.New(&aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}))

	p.addHandlers(regionName, &elbClient.Handlers)

	return elbClient, nil
}

func (p *awsSDKProvider) Autoscaling(regionName string) (ASG, error) {
	client := autoscaling.New(session.New(&aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}))

	p.addHandlers(regionName, &client.Handlers)

	return client, nil
}

func (p *awsSDKProvider) Metadata() (EC2Metadata, error) {
	client := ec2metadata.New(session.New(&aws.Config{}))
	return client, nil
}

// stringPointerArray creates a slice of string pointers from a slice of strings
// Deprecated: consider using aws.StringSlice - but note the slightly different behaviour with a nil input
func stringPointerArray(orig []string) []*string {
	if orig == nil {
		return nil
	}
	return aws.StringSlice(orig)
}

// isNilOrEmpty returns true if the value is nil or ""
// Deprecated: prefer aws.StringValue(x) == "" (and elimination of this check altogether whrere possible)
func isNilOrEmpty(s *string) bool {
	return s == nil || *s == ""
}

// orEmpty returns the string value, or "" if the pointer is nil
// Deprecated: prefer aws.StringValue
func orEmpty(s *string) string {
	return aws.StringValue(s)
}

func newEc2Filter(name string, value string) *ec2.Filter {
	filter := &ec2.Filter{
		Name: aws.String(name),
		Values: []*string{
			aws.String(value),
		},
	}
	return filter
}

// AddSSHKeyToAllInstances is currently not implemented.
func (c *Cloud) AddSSHKeyToAllInstances(user string, keyData []byte) error {
	return errors.New("unimplemented")
}

// CurrentNodeName returns the name of the current node
func (c *Cloud) CurrentNodeName(hostname string) (types.NodeName, error) {
	return c.selfAWSInstance.nodeName, nil
}

// Implementation of EC2.Instances
func (s *awsSdkEC2) DescribeInstances(request *ec2.DescribeInstancesInput) ([]*ec2.Instance, error) {
	// Instances are paged
	results := []*ec2.Instance{}
	var nextToken *string

	for {
		response, err := s.ec2.DescribeInstances(request)
		if err != nil {
			return nil, fmt.Errorf("error listing AWS instances: %v", err)
		}

		for _, reservation := range response.Reservations {
			results = append(results, reservation.Instances...)
		}

		nextToken = response.NextToken
		if isNilOrEmpty(nextToken) {
			break
		}
		request.NextToken = nextToken
	}

	return results, nil
}

// Implements EC2.DescribeSecurityGroups
func (s *awsSdkEC2) DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error) {
	// Security groups are not paged
	response, err := s.ec2.DescribeSecurityGroups(request)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS security groups: %v", err)
	}
	return response.SecurityGroups, nil
}

func (s *awsSdkEC2) AttachVolume(request *ec2.AttachVolumeInput) (*ec2.VolumeAttachment, error) {
	return s.ec2.AttachVolume(request)
}

func (s *awsSdkEC2) DetachVolume(request *ec2.DetachVolumeInput) (*ec2.VolumeAttachment, error) {
	return s.ec2.DetachVolume(request)
}

func (s *awsSdkEC2) DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error) {
	// Volumes are paged
	results := []*ec2.Volume{}
	var nextToken *string

	for {
		response, err := s.ec2.DescribeVolumes(request)

		if err != nil {
			return nil, fmt.Errorf("error listing AWS volumes: %v", err)
		}

		results = append(results, response.Volumes...)

		nextToken = response.NextToken
		if isNilOrEmpty(nextToken) {
			break
		}
		request.NextToken = nextToken
	}

	return results, nil
}

func (s *awsSdkEC2) CreateVolume(request *ec2.CreateVolumeInput) (resp *ec2.Volume, err error) {
	return s.ec2.CreateVolume(request)
}

func (s *awsSdkEC2) DeleteVolume(request *ec2.DeleteVolumeInput) (*ec2.DeleteVolumeOutput, error) {
	return s.ec2.DeleteVolume(request)
}

func (s *awsSdkEC2) DescribeSubnets(request *ec2.DescribeSubnetsInput) ([]*ec2.Subnet, error) {
	// Subnets are not paged
	response, err := s.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS subnets: %v", err)
	}
	return response.Subnets, nil
}

func (s *awsSdkEC2) CreateSecurityGroup(request *ec2.CreateSecurityGroupInput) (*ec2.CreateSecurityGroupOutput, error) {
	return s.ec2.CreateSecurityGroup(request)
}

func (s *awsSdkEC2) DeleteSecurityGroup(request *ec2.DeleteSecurityGroupInput) (*ec2.DeleteSecurityGroupOutput, error) {
	return s.ec2.DeleteSecurityGroup(request)
}

func (s *awsSdkEC2) AuthorizeSecurityGroupIngress(request *ec2.AuthorizeSecurityGroupIngressInput) (*ec2.AuthorizeSecurityGroupIngressOutput, error) {
	return s.ec2.AuthorizeSecurityGroupIngress(request)
}

func (s *awsSdkEC2) RevokeSecurityGroupIngress(request *ec2.RevokeSecurityGroupIngressInput) (*ec2.RevokeSecurityGroupIngressOutput, error) {
	return s.ec2.RevokeSecurityGroupIngress(request)
}

func (s *awsSdkEC2) CreateTags(request *ec2.CreateTagsInput) (*ec2.CreateTagsOutput, error) {
	return s.ec2.CreateTags(request)
}

func (s *awsSdkEC2) DescribeRouteTables(request *ec2.DescribeRouteTablesInput) ([]*ec2.RouteTable, error) {
	// Not paged
	response, err := s.ec2.DescribeRouteTables(request)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS route tables: %v", err)
	}
	return response.RouteTables, nil
}

func (s *awsSdkEC2) CreateRoute(request *ec2.CreateRouteInput) (*ec2.CreateRouteOutput, error) {
	return s.ec2.CreateRoute(request)
}

func (s *awsSdkEC2) DeleteRoute(request *ec2.DeleteRouteInput) (*ec2.DeleteRouteOutput, error) {
	return s.ec2.DeleteRoute(request)
}

func (s *awsSdkEC2) ModifyInstanceAttribute(request *ec2.ModifyInstanceAttributeInput) (*ec2.ModifyInstanceAttributeOutput, error) {
	return s.ec2.ModifyInstanceAttribute(request)
}

func init() {
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		creds := credentials.NewChainCredentials(
			[]credentials.Provider{
				&credentials.EnvProvider{},
				&ec2rolecreds.EC2RoleProvider{
					Client: ec2metadata.New(session.New(&aws.Config{})),
				},
				&credentials.SharedCredentialsProvider{},
			})
		aws := newAWSSDKProvider(creds)
		return newAWSCloud(config, aws)
	})
}

// readAWSCloudConfig reads an instance of AWSCloudConfig from config reader.
func readAWSCloudConfig(config io.Reader, metadata EC2Metadata) (*CloudConfig, error) {
	var cfg CloudConfig
	var err error

	if config != nil {
		err = gcfg.ReadInto(&cfg, config)
		if err != nil {
			return nil, err
		}
	}

	if cfg.Global.Zone == "" {
		if metadata != nil {
			glog.Info("Zone not specified in configuration file; querying AWS metadata service")
			cfg.Global.Zone, err = getAvailabilityZone(metadata)
			if err != nil {
				return nil, err
			}
		}
		if cfg.Global.Zone == "" {
			return nil, fmt.Errorf("no zone specified in configuration file")
		}
	}

	return &cfg, nil
}

func getInstanceType(metadata EC2Metadata) (string, error) {
	return metadata.GetMetadata("instance-type")
}

func getAvailabilityZone(metadata EC2Metadata) (string, error) {
	return metadata.GetMetadata("placement/availability-zone")
}

func isRegionValid(region string) bool {
	for _, r := range aws_credentials.AWSRegions {
		if r == region {
			return true
		}
	}
	return false
}

// Derives the region from a valid az name.
// Returns an error if the az is known invalid (empty)
func azToRegion(az string) (string, error) {
	if len(az) < 1 {
		return "", fmt.Errorf("invalid (empty) AZ")
	}
	region := az[:len(az)-1]
	return region, nil
}

// newAWSCloud creates a new instance of AWSCloud.
// AWSProvider and instanceId are primarily for tests
func newAWSCloud(config io.Reader, awsServices Services) (*Cloud, error) {
	// We have some state in the Cloud object - in particular the attaching map
	// Log so that if we are building multiple Cloud objects, it is obvious!
	glog.Infof("Building AWS cloudprovider")

	metadata, err := awsServices.Metadata()
	if err != nil {
		return nil, fmt.Errorf("error creating AWS metadata client: %v", err)
	}

	cfg, err := readAWSCloudConfig(config, metadata)
	if err != nil {
		return nil, fmt.Errorf("unable to read AWS cloud provider config file: %v", err)
	}

	zone := cfg.Global.Zone
	if len(zone) <= 1 {
		return nil, fmt.Errorf("invalid AWS zone in config file: %s", zone)
	}
	regionName, err := azToRegion(zone)
	if err != nil {
		return nil, err
	}

	if !cfg.Global.DisableStrictZoneCheck {
		valid := isRegionValid(regionName)
		if !valid {
			return nil, fmt.Errorf("not a valid AWS zone (unknown region): %s", zone)
		}
	} else {
		glog.Warningf("Strict AWS zone checking is disabled.  Proceeding with zone: %s", zone)
	}

	ec2, err := awsServices.Compute(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS EC2 client: %v", err)
	}

	elb, err := awsServices.LoadBalancing(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS ELB client: %v", err)
	}

	asg, err := awsServices.Autoscaling(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS autoscaling client: %v", err)
	}

	awsCloud := &Cloud{
		ec2:      ec2,
		elb:      elb,
		asg:      asg,
		metadata: metadata,
		cfg:      cfg,
		region:   regionName,

		attaching:        make(map[types.NodeName]map[mountDevice]awsVolumeID),
		deviceAllocators: make(map[types.NodeName]DeviceAllocator),
	}

	selfAWSInstance, err := awsCloud.buildSelfAWSInstance()
	if err != nil {
		return nil, err
	}

	awsCloud.selfAWSInstance = selfAWSInstance
	awsCloud.vpcID = selfAWSInstance.vpcID

	filterTags := map[string]string{}
	if cfg.Global.KubernetesClusterTag != "" {
		filterTags[TagNameKubernetesCluster] = cfg.Global.KubernetesClusterTag
	} else {
		// TODO: Clean up double-API query
		info, err := selfAWSInstance.describeInstance()
		if err != nil {
			return nil, err
		}
		for _, tag := range info.Tags {
			if orEmpty(tag.Key) == TagNameKubernetesCluster {
				filterTags[TagNameKubernetesCluster] = orEmpty(tag.Value)
			}
		}
	}

	if filterTags[TagNameKubernetesCluster] == "" {
		glog.Errorf("Tag %q not found; Kubernetes may behave unexpectedly.", TagNameKubernetesCluster)
	}

	awsCloud.filterTags = filterTags
	if len(filterTags) > 0 {
		glog.Infof("AWS cloud filtering on tags: %v", filterTags)
	} else {
		glog.Infof("AWS cloud - no tag filtering")
	}

	// Register handler for ECR credentials
	once.Do(func() {
		aws_credentials.Init()
	})

	return awsCloud, nil
}

// Clusters returns the list of clusters.
func (c *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (c *Cloud) ProviderName() string {
	return ProviderName
}

// ScrubDNS filters DNS settings for pods.
func (c *Cloud) ScrubDNS(nameservers, searches []string) (nsOut, srchOut []string) {
	return nameservers, searches
}

// LoadBalancer returns an implementation of LoadBalancer for Amazon Web Services.
func (c *Cloud) LoadBalancer() (cloudprovider.LoadBalancer, bool) {
	return c, true
}

// Instances returns an implementation of Instances for Amazon Web Services.
func (c *Cloud) Instances() (cloudprovider.Instances, bool) {
	return c, true
}

// Zones returns an implementation of Zones for Amazon Web Services.
func (c *Cloud) Zones() (cloudprovider.Zones, bool) {
	return c, true
}

// Routes returns an implementation of Routes for Amazon Web Services.
func (c *Cloud) Routes() (cloudprovider.Routes, bool) {
	return c, true
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (c *Cloud) NodeAddresses(name types.NodeName) ([]api.NodeAddress, error) {
	if c.selfAWSInstance.nodeName == name || len(name) == 0 {
		addresses := []api.NodeAddress{}

		internalIP, err := c.metadata.GetMetadata("local-ipv4")
		if err != nil {
			return nil, err
		}
		addresses = append(addresses, api.NodeAddress{Type: api.NodeInternalIP, Address: internalIP})
		// Legacy compatibility: the private ip was the legacy host ip
		addresses = append(addresses, api.NodeAddress{Type: api.NodeLegacyHostIP, Address: internalIP})

		externalIP, err := c.metadata.GetMetadata("public-ipv4")
		if err != nil {
			//TODO: It would be nice to be able to determine the reason for the failure,
			// but the AWS client masks all failures with the same error description.
			glog.V(2).Info("Could not determine public IP from AWS metadata.")
		} else {
			addresses = append(addresses, api.NodeAddress{Type: api.NodeExternalIP, Address: externalIP})
		}

		return addresses, nil
	}
	instance, err := c.getInstanceByNodeName(name)
	if err != nil {
		return nil, fmt.Errorf("getInstanceByNodeName failed for %q with %v", name, err)
	}

	addresses := []api.NodeAddress{}

	if !isNilOrEmpty(instance.PrivateIpAddress) {
		ipAddress := *instance.PrivateIpAddress
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("EC2 instance had invalid private address: %s (%s)", orEmpty(instance.InstanceId), ipAddress)
		}
		addresses = append(addresses, api.NodeAddress{Type: api.NodeInternalIP, Address: ip.String()})

		// Legacy compatibility: the private ip was the legacy host ip
		addresses = append(addresses, api.NodeAddress{Type: api.NodeLegacyHostIP, Address: ip.String()})
	}

	// TODO: Other IP addresses (multiple ips)?
	if !isNilOrEmpty(instance.PublicIpAddress) {
		ipAddress := *instance.PublicIpAddress
		ip := net.ParseIP(ipAddress)
		if ip == nil {
			return nil, fmt.Errorf("EC2 instance had invalid public address: %s (%s)", orEmpty(instance.InstanceId), ipAddress)
		}
		addresses = append(addresses, api.NodeAddress{Type: api.NodeExternalIP, Address: ip.String()})
	}

	return addresses, nil
}

// ExternalID returns the cloud provider ID of the node with the specified nodeName (deprecated).
func (c *Cloud) ExternalID(nodeName types.NodeName) (string, error) {
	if c.selfAWSInstance.nodeName == nodeName {
		// We assume that if this is run on the instance itself, the instance exists and is alive
		return c.selfAWSInstance.awsID, nil
	}
	// We must verify that the instance still exists
	// Note that if the instance does not exist or is no longer running, we must return ("", cloudprovider.InstanceNotFound)
	instance, err := c.findInstanceByNodeName(nodeName)
	if err != nil {
		return "", err
	}
	if instance == nil {
		return "", cloudprovider.InstanceNotFound
	}
	return orEmpty(instance.InstanceId), nil
}

// InstanceID returns the cloud provider ID of the node with the specified nodeName.
func (c *Cloud) InstanceID(nodeName types.NodeName) (string, error) {
	// In the future it is possible to also return an endpoint as:
	// <endpoint>/<zone>/<instanceid>
	if c.selfAWSInstance.nodeName == nodeName {
		return "/" + c.selfAWSInstance.availabilityZone + "/" + c.selfAWSInstance.awsID, nil
	}
	inst, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		return "", fmt.Errorf("getInstanceByNodeName failed for %q with %v", nodeName, err)
	}
	return "/" + orEmpty(inst.Placement.AvailabilityZone) + "/" + orEmpty(inst.InstanceId), nil
}

// InstanceType returns the type of the node with the specified nodeName.
func (c *Cloud) InstanceType(nodeName types.NodeName) (string, error) {
	if c.selfAWSInstance.nodeName == nodeName {
		return c.selfAWSInstance.instanceType, nil
	}
	inst, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		return "", fmt.Errorf("getInstanceByNodeName failed for %q with %v", nodeName, err)
	}
	return orEmpty(inst.InstanceType), nil
}

// Return a list of instances matching regex string.
func (c *Cloud) getInstancesByRegex(regex string) ([]types.NodeName, error) {
	filters := []*ec2.Filter{newEc2Filter("instance-state-name", "running")}
	filters = c.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return []types.NodeName{}, err
	}
	if len(instances) == 0 {
		return []types.NodeName{}, fmt.Errorf("no instances returned")
	}

	if strings.HasPrefix(regex, "'") && strings.HasSuffix(regex, "'") {
		glog.Infof("Stripping quotes around regex (%s)", regex)
		regex = regex[1 : len(regex)-1]
	}

	re, err := regexp.Compile(regex)
	if err != nil {
		return []types.NodeName{}, err
	}

	matchingInstances := []types.NodeName{}
	for _, instance := range instances {
		// Only return fully-ready instances when listing instances
		// (vs a query by name, where we will return it if we find it)
		if orEmpty(instance.State.Name) == "pending" {
			glog.V(2).Infof("Skipping EC2 instance (pending): %s", *instance.InstanceId)
			continue
		}

		nodeName := mapInstanceToNodeName(instance)
		if nodeName == "" {
			glog.V(2).Infof("Skipping EC2 instance (no PrivateDNSName): %s",
				aws.StringValue(instance.InstanceId))
			continue
		}

		for _, tag := range instance.Tags {
			if orEmpty(tag.Key) == "Name" && re.MatchString(orEmpty(tag.Value)) {
				matchingInstances = append(matchingInstances, nodeName)
				break
			}
		}
	}
	glog.V(2).Infof("Matched EC2 instances: %s", matchingInstances)
	return matchingInstances, nil
}

// List is an implementation of Instances.List.
func (c *Cloud) List(filter string) ([]types.NodeName, error) {
	// TODO: Should really use tag query. No need to go regexp.
	return c.getInstancesByRegex(filter)
}

// getAllZones retrieves  a list of all the zones in which nodes are running
// It currently involves querying all instances
func (c *Cloud) getAllZones() (sets.String, error) {
	// We don't currently cache this; it is currently used only in volume
	// creation which is expected to be a comparatively rare occurrence.

	// TODO: Caching / expose api.Nodes to the cloud provider?
	// TODO: We could also query for subnets, I think

	filters := []*ec2.Filter{newEc2Filter("instance-state-name", "running")}
	filters = c.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances returned")
	}

	zones := sets.NewString()

	for _, instance := range instances {
		// Only return fully-ready instances when listing instances
		// (vs a query by name, where we will return it if we find it)
		if orEmpty(instance.State.Name) == "pending" {
			glog.V(2).Infof("Skipping EC2 instance (pending): %s", *instance.InstanceId)
			continue
		}

		if instance.Placement != nil {
			zone := aws.StringValue(instance.Placement.AvailabilityZone)
			zones.Insert(zone)
		}
	}

	glog.V(2).Infof("Found instances in zones %s", zones)
	return zones, nil
}

// GetZone implements Zones.GetZone
func (c *Cloud) GetZone() (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: c.selfAWSInstance.availabilityZone,
		Region:        c.region,
	}, nil
}

// Abstraction around AWS Instance Types
// There isn't an API to get information for a particular instance type (that I know of)
type awsInstanceType struct {
}

// Used to represent a mount device for attaching an EBS volume
// This should be stored as a single letter (i.e. c, not sdc or /dev/sdc)
type mountDevice string

type awsInstance struct {
	ec2 EC2

	// id in AWS
	awsID string

	// node name in k8s
	nodeName types.NodeName

	// availability zone the instance resides in
	availabilityZone string

	// ID of VPC the instance resides in
	vpcID string

	// ID of subnet the instance resides in
	subnetID string

	// instance type
	instanceType string
}

// newAWSInstance creates a new awsInstance object
func newAWSInstance(ec2Service EC2, instance *ec2.Instance) *awsInstance {
	az := ""
	if instance.Placement != nil {
		az = aws.StringValue(instance.Placement.AvailabilityZone)
	}
	self := &awsInstance{
		ec2:              ec2Service,
		awsID:            aws.StringValue(instance.InstanceId),
		nodeName:         mapInstanceToNodeName(instance),
		availabilityZone: az,
		instanceType:     aws.StringValue(instance.InstanceType),
		vpcID:            aws.StringValue(instance.VpcId),
		subnetID:         aws.StringValue(instance.SubnetId),
	}

	return self
}

// Gets the awsInstanceType that models the instance type of this instance
func (i *awsInstance) getInstanceType() *awsInstanceType {
	// TODO: Make this real
	awsInstanceType := &awsInstanceType{}
	return awsInstanceType
}

// Gets the full information about this instance from the EC2 API
func (i *awsInstance) describeInstance() (*ec2.Instance, error) {
	instanceID := i.awsID
	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{&instanceID},
	}

	instances, err := i.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", i.awsID)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", i.awsID)
	}
	return instances[0], nil
}

// Gets the mountDevice already assigned to the volume, or assigns an unused mountDevice.
// If the volume is already assigned, this will return the existing mountDevice with alreadyAttached=true.
// Otherwise the mountDevice is assigned by finding the first available mountDevice, and it is returned with alreadyAttached=false.
func (c *Cloud) getMountDevice(i *awsInstance, volumeID awsVolumeID, assign bool) (assigned mountDevice, alreadyAttached bool, err error) {
	instanceType := i.getInstanceType()
	if instanceType == nil {
		return "", false, fmt.Errorf("could not get instance type for instance: %s", i.awsID)
	}

	info, err := i.describeInstance()
	if err != nil {
		return "", false, err
	}
	deviceMappings := map[mountDevice]awsVolumeID{}
	for _, blockDevice := range info.BlockDeviceMappings {
		name := aws.StringValue(blockDevice.DeviceName)
		if strings.HasPrefix(name, "/dev/sd") {
			name = name[7:]
		}
		if strings.HasPrefix(name, "/dev/xvd") {
			name = name[8:]
		}
		if len(name) < 1 || len(name) > 2 {
			glog.Warningf("Unexpected EBS DeviceName: %q", aws.StringValue(blockDevice.DeviceName))
		}
		deviceMappings[mountDevice(name)] = awsVolumeID(aws.StringValue(blockDevice.Ebs.VolumeId))
	}

	// We lock to prevent concurrent mounts from conflicting
	// We may still conflict if someone calls the API concurrently,
	// but the AWS API will then fail one of the two attach operations
	c.attachingMutex.Lock()
	defer c.attachingMutex.Unlock()

	for mountDevice, volume := range c.attaching[i.nodeName] {
		deviceMappings[mountDevice] = volume
	}

	// Check to see if this volume is already assigned a device on this machine
	for mountDevice, mappingVolumeID := range deviceMappings {
		if volumeID == mappingVolumeID {
			if assign {
				glog.Warningf("Got assignment call for already-assigned volume: %s@%s", mountDevice, mappingVolumeID)
			}
			return mountDevice, true, nil
		}
	}

	if !assign {
		return mountDevice(""), false, nil
	}

	// Find the next unused device name
	deviceAllocator := c.deviceAllocators[i.nodeName]
	if deviceAllocator == nil {
		// we want device names with two significant characters, starting with
		// /dev/xvdba (leaving xvda - xvdz and xvdaa-xvdaz to the system)
		deviceAllocator = NewDeviceAllocator(2, "ba")
		c.deviceAllocators[i.nodeName] = deviceAllocator
	}
	chosen, err := deviceAllocator.GetNext(deviceMappings)
	if err != nil {
		glog.Warningf("Could not assign a mount device.  mappings=%v, error: %v", deviceMappings, err)
		return "", false, fmt.Errorf("Too many EBS volumes attached to node %s.", i.nodeName)
	}

	attaching := c.attaching[i.nodeName]
	if attaching == nil {
		attaching = make(map[mountDevice]awsVolumeID)
		c.attaching[i.nodeName] = attaching
	}
	attaching[chosen] = volumeID
	glog.V(2).Infof("Assigned mount device %s -> volume %s", chosen, volumeID)

	return chosen, false, nil
}

// endAttaching removes the entry from the "attachments in progress" map
// It returns true if it was found (and removed), false otherwise
func (c *Cloud) endAttaching(i *awsInstance, volumeID awsVolumeID, mountDevice mountDevice) bool {
	c.attachingMutex.Lock()
	defer c.attachingMutex.Unlock()

	existingVolumeID, found := c.attaching[i.nodeName][mountDevice]
	if !found {
		return false
	}
	if volumeID != existingVolumeID {
		// This actually can happen, because getMountDevice combines the attaching map with the volumes
		// attached to the instance (as reported by the EC2 API).  So if endAttaching comes after
		// a 10 second poll delay, we might well have had a concurrent request to allocate a mountpoint,
		// which because we allocate sequentially is _very_ likely to get the immediately freed volume
		glog.Infof("endAttaching on device %q assigned to different volume: %q vs %q", mountDevice, volumeID, existingVolumeID)
		return false
	}
	glog.V(2).Infof("Releasing in-process attachment entry: %s -> volume %s", mountDevice, volumeID)
	delete(c.attaching[i.nodeName], mountDevice)
	return true
}

type awsDisk struct {
	ec2 EC2

	// Name in k8s
	name KubernetesVolumeID
	// id in AWS
	awsID awsVolumeID
}

func newAWSDisk(aws *Cloud, name KubernetesVolumeID) (*awsDisk, error) {
	awsID, err := name.mapToAWSVolumeID()
	if err != nil {
		return nil, err
	}
	disk := &awsDisk{ec2: aws.ec2, name: name, awsID: awsID}
	return disk, nil
}

// Gets the full information about this volume from the EC2 API
func (d *awsDisk) describeVolume() (*ec2.Volume, error) {
	volumeID := d.awsID

	request := &ec2.DescribeVolumesInput{
		VolumeIds: []*string{volumeID.awsString()},
	}

	volumes, err := d.ec2.DescribeVolumes(request)
	if err != nil {
		return nil, fmt.Errorf("error querying ec2 for volume %q: %v", volumeID, err)
	}
	if len(volumes) == 0 {
		return nil, fmt.Errorf("no volumes found for volume %q", volumeID)
	}
	if len(volumes) > 1 {
		return nil, fmt.Errorf("multiple volumes found for volume %q", volumeID)
	}
	return volumes[0], nil
}

// waitForAttachmentStatus polls until the attachment status is the expected value
// On success, it returns the last attachment state.
func (d *awsDisk) waitForAttachmentStatus(status string) (*ec2.VolumeAttachment, error) {
	// We wait up to 30 minutes for the attachment to complete.
	// This mirrors the GCE timeout.
	timeoutAt := time.Now().UTC().Add(volumeAttachmentStatusTimeout).Unix()

	// Because of rate limiting, we often see errors from describeVolume
	// So we tolerate a limited number of failures.
	// But once we see more than 10 errors in a row, we return the error
	describeErrorCount := 0

	for {
		info, err := d.describeVolume()
		if err != nil {
			describeErrorCount++
			if describeErrorCount > volumeAttachmentStatusConsecutiveErrorLimit {
				return nil, err
			} else {
				glog.Warningf("Ignoring error from describe volume; will retry: %q", err)
				time.Sleep(volumeAttachmentStatusErrorDelay)
				continue
			}
		} else {
			describeErrorCount = 0
		}
		if len(info.Attachments) > 1 {
			// Shouldn't happen; log so we know if it is
			glog.Warningf("Found multiple attachments for volume %q: %v", d.awsID, info)
		}
		var attachment *ec2.VolumeAttachment
		attachmentStatus := ""
		for _, a := range info.Attachments {
			if attachmentStatus != "" {
				// Shouldn't happen; log so we know if it is
				glog.Warningf("Found multiple attachments for volume %q: %v", d.awsID, info)
			}
			if a.State != nil {
				attachment = a
				attachmentStatus = *a.State
			} else {
				// Shouldn't happen; log so we know if it is
				glog.Warningf("Ignoring nil attachment state for volume %q: %v", d.awsID, a)
			}
		}
		if attachmentStatus == "" {
			attachmentStatus = "detached"
		}
		if attachmentStatus == status {
			return attachment, nil
		}

		if time.Now().Unix() > timeoutAt {
			glog.Warningf("Timeout waiting for volume %q state: actual=%s, desired=%s", d.awsID, attachmentStatus, status)
			return nil, fmt.Errorf("Timeout waiting for volume %q state: actual=%s, desired=%s", d.awsID, attachmentStatus, status)
		}

		glog.V(2).Infof("Waiting for volume %q state: actual=%s, desired=%s", d.awsID, attachmentStatus, status)

		time.Sleep(volumeAttachmentStatusPollInterval)
	}
}

// Deletes the EBS disk
func (d *awsDisk) deleteVolume() (bool, error) {
	request := &ec2.DeleteVolumeInput{VolumeId: d.awsID.awsString()}
	_, err := d.ec2.DeleteVolume(request)
	if err != nil {
		if awsError, ok := err.(awserr.Error); ok {
			if awsError.Code() == "InvalidVolume.NotFound" {
				return false, nil
			}
			if awsError.Code() == "VolumeInUse" {
				return false, volume.NewDeletedVolumeInUseError(err.Error())
			}
		}
		return false, fmt.Errorf("error deleting EBS volume %q: %v", d.awsID, err)
	}
	return true, nil
}

// Builds the awsInstance for the EC2 instance on which we are running.
// This is called when the AWSCloud is initialized, and should not be called otherwise (because the awsInstance for the local instance is a singleton with drive mapping state)
func (c *Cloud) buildSelfAWSInstance() (*awsInstance, error) {
	if c.selfAWSInstance != nil {
		panic("do not call buildSelfAWSInstance directly")
	}
	instanceID, err := c.metadata.GetMetadata("instance-id")
	if err != nil {
		return nil, fmt.Errorf("error fetching instance-id from ec2 metadata service: %v", err)
	}

	// We want to fetch the hostname via the EC2 metadata service
	// (`GetMetadata("local-hostname")`): But see #11543 - we need to use
	// the EC2 API to get the privateDnsName in case of a private DNS zone
	// e.g. mydomain.io, because the metadata service returns the wrong
	// hostname.  Once we're doing that, we might as well get all our
	// information from the instance returned by the EC2 API - it is a
	// single API call to get all the information, and it means we don't
	// have two code paths.
	instance, err := c.getInstanceByID(instanceID)
	if err != nil {
		return nil, fmt.Errorf("error finding instance %s: %v", instanceID, err)
	}
	return newAWSInstance(c.ec2, instance), nil
}

// Gets the awsInstance with for the node with the specified nodeName, or the 'self' instance if nodeName == ""
func (c *Cloud) getAwsInstance(nodeName types.NodeName) (*awsInstance, error) {
	var awsInstance *awsInstance
	if nodeName == "" {
		awsInstance = c.selfAWSInstance
	} else {
		instance, err := c.getInstanceByNodeName(nodeName)
		if err != nil {
			return nil, err
		}

		awsInstance = newAWSInstance(c.ec2, instance)
	}

	return awsInstance, nil
}

// AttachDisk implements Volumes.AttachDisk
func (c *Cloud) AttachDisk(diskName KubernetesVolumeID, nodeName types.NodeName, readOnly bool) (string, error) {
	disk, err := newAWSDisk(c, diskName)
	if err != nil {
		return "", err
	}

	awsInstance, err := c.getAwsInstance(nodeName)
	if err != nil {
		return "", fmt.Errorf("error finding instance %s: %v", nodeName, err)
	}

	if readOnly {
		// TODO: We could enforce this when we mount the volume (?)
		// TODO: We could also snapshot the volume and attach copies of it
		return "", errors.New("AWS volumes cannot be mounted read-only")
	}

	// mountDevice will hold the device where we should try to attach the disk
	var mountDevice mountDevice
	// alreadyAttached is true if we have already called AttachVolume on this disk
	var alreadyAttached bool

	// attachEnded is set to true if the attach operation completed
	// (successfully or not), and is thus no longer in progress
	attachEnded := false
	defer func() {
		if attachEnded {
			if !c.endAttaching(awsInstance, disk.awsID, mountDevice) {
				glog.Errorf("endAttaching called for disk %q when attach not in progress", disk.awsID)
			}
		}
	}()

	mountDevice, alreadyAttached, err = c.getMountDevice(awsInstance, disk.awsID, true)
	if err != nil {
		return "", err
	}

	// Inside the instance, the mountpoint always looks like /dev/xvdX (?)
	hostDevice := "/dev/xvd" + string(mountDevice)
	// We are using xvd names (so we are HVM only)
	// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/device_naming.html
	ec2Device := "/dev/xvd" + string(mountDevice)

	if !alreadyAttached {
		request := &ec2.AttachVolumeInput{
			Device:     aws.String(ec2Device),
			InstanceId: aws.String(awsInstance.awsID),
			VolumeId:   disk.awsID.awsString(),
		}

		attachResponse, err := c.ec2.AttachVolume(request)
		if err != nil {
			attachEnded = true
			// TODO: Check if the volume was concurrently attached?
			return "", fmt.Errorf("Error attaching EBS volume %q to instance %q: %v", disk.awsID, awsInstance.awsID, err)
		}

		glog.V(2).Infof("AttachVolume volume=%q instance=%q request returned %v", disk.awsID, awsInstance.awsID, attachResponse)
	}

	attachment, err := disk.waitForAttachmentStatus("attached")
	if err != nil {
		return "", err
	}

	// The attach operation has finished
	attachEnded = true

	// Double check the attachment to be 100% sure we attached the correct volume at the correct mountpoint
	// It could happen otherwise that we see the volume attached from a previous/separate AttachVolume call,
	// which could theoretically be against a different device (or even instance).
	if attachment == nil {
		// Impossible?
		return "", fmt.Errorf("unexpected state: attachment nil after attached %q to %q", diskName, nodeName)
	}
	if ec2Device != aws.StringValue(attachment.Device) {
		return "", fmt.Errorf("disk attachment of %q to %q failed: requested device %q but found %q", diskName, nodeName, ec2Device, aws.StringValue(attachment.Device))
	}
	if awsInstance.awsID != aws.StringValue(attachment.InstanceId) {
		return "", fmt.Errorf("disk attachment of %q to %q failed: requested instance %q but found %q", diskName, nodeName, awsInstance.awsID, aws.StringValue(attachment.InstanceId))
	}

	return hostDevice, nil
}

// DetachDisk implements Volumes.DetachDisk
func (c *Cloud) DetachDisk(diskName KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	disk, err := newAWSDisk(c, diskName)
	if err != nil {
		return "", err
	}

	awsInstance, err := c.getAwsInstance(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Instance %q does not exist. DetachDisk will assume disk %q is not attached to it.",
				nodeName,
				diskName)
			return "", nil
		}

		return "", err
	}

	mountDevice, alreadyAttached, err := c.getMountDevice(awsInstance, disk.awsID, false)
	if err != nil {
		return "", err
	}

	if !alreadyAttached {
		glog.Warningf("DetachDisk called on non-attached disk: %s", diskName)
		// TODO: Continue?  Tolerate non-attached error from the AWS DetachVolume call?
	}

	request := ec2.DetachVolumeInput{
		InstanceId: &awsInstance.awsID,
		VolumeId:   disk.awsID.awsString(),
	}

	response, err := c.ec2.DetachVolume(&request)
	if err != nil {
		return "", fmt.Errorf("error detaching EBS volume %q from %q: %v", disk.awsID, awsInstance.awsID, err)
	}
	if response == nil {
		return "", errors.New("no response from DetachVolume")
	}

	attachment, err := disk.waitForAttachmentStatus("detached")
	if err != nil {
		return "", err
	}
	if attachment != nil {
		// We expect it to be nil, it is (maybe) interesting if it is not
		glog.V(2).Infof("waitForAttachmentStatus returned non-nil attachment with state=detached: %v", attachment)
	}

	if mountDevice != "" {
		c.endAttaching(awsInstance, disk.awsID, mountDevice)
		// We don't check the return value - we don't really expect the attachment to have been
		// in progress, though it might have been
	}

	hostDevicePath := "/dev/xvd" + string(mountDevice)
	return hostDevicePath, err
}

// CreateDisk implements Volumes.CreateDisk
func (c *Cloud) CreateDisk(volumeOptions *VolumeOptions) (KubernetesVolumeID, error) {
	allZones, err := c.getAllZones()
	if err != nil {
		return "", fmt.Errorf("error querying for all zones: %v", err)
	}

	createAZ := volumeOptions.AvailabilityZone
	if createAZ == "" {
		createAZ = volume.ChooseZoneForVolume(allZones, volumeOptions.PVCName)
	}

	var createType string
	var iops int64
	switch volumeOptions.VolumeType {
	case VolumeTypeGP2, VolumeTypeSC1, VolumeTypeST1:
		createType = volumeOptions.VolumeType

	case VolumeTypeIO1:
		// See http://docs.aws.amazon.com/AWSEC2/latest/APIReference/API_CreateVolume.html
		// for IOPS constraints. AWS will throw an error if IOPS per GB gets out
		// of supported bounds, no need to check it here.
		createType = volumeOptions.VolumeType
		iops = int64(volumeOptions.CapacityGB * volumeOptions.IOPSPerGB)

		// Cap at min/max total IOPS, AWS would throw an error if it gets too
		// low/high.
		if iops < MinTotalIOPS {
			iops = MinTotalIOPS
		}
		if iops > MaxTotalIOPS {
			iops = MaxTotalIOPS
		}

	case "":
		createType = DefaultVolumeType

	default:
		return "", fmt.Errorf("invalid AWS VolumeType %q", volumeOptions.VolumeType)
	}

	// TODO: Should we tag this with the cluster id (so it gets deleted when the cluster does?)
	request := &ec2.CreateVolumeInput{}
	request.AvailabilityZone = aws.String(createAZ)
	request.Size = aws.Int64(int64(volumeOptions.CapacityGB))
	request.VolumeType = aws.String(createType)
	request.Encrypted = aws.Bool(volumeOptions.Encrypted)
	if len(volumeOptions.KmsKeyId) > 0 {
		request.KmsKeyId = aws.String(volumeOptions.KmsKeyId)
		request.Encrypted = aws.Bool(true)
	}
	if iops > 0 {
		request.Iops = aws.Int64(iops)
	}
	response, err := c.ec2.CreateVolume(request)
	if err != nil {
		return "", err
	}

	awsID := awsVolumeID(aws.StringValue(response.VolumeId))
	if awsID == "" {
		return "", fmt.Errorf("VolumeID was not returned by CreateVolume")
	}
	volumeName := KubernetesVolumeID("aws://" + aws.StringValue(response.AvailabilityZone) + "/" + string(awsID))

	// apply tags
	tags := make(map[string]string)
	for k, v := range volumeOptions.Tags {
		tags[k] = v
	}

	if c.getClusterName() != "" {
		tags[TagNameKubernetesCluster] = c.getClusterName()
	}

	if len(tags) != 0 {
		if err := c.createTags(string(awsID), tags); err != nil {
			// delete the volume and hope it succeeds
			_, delerr := c.DeleteDisk(volumeName)
			if delerr != nil {
				// delete did not succeed, we have a stray volume!
				return "", fmt.Errorf("error tagging volume %s, could not delete the volume: %v", volumeName, delerr)
			}
			return "", fmt.Errorf("error tagging volume %s: %v", volumeName, err)
		}
	}
	return volumeName, nil
}

// DeleteDisk implements Volumes.DeleteDisk
func (c *Cloud) DeleteDisk(volumeName KubernetesVolumeID) (bool, error) {
	awsDisk, err := newAWSDisk(c, volumeName)
	if err != nil {
		return false, err
	}
	return awsDisk.deleteVolume()
}

// GetVolumeLabels implements Volumes.GetVolumeLabels
func (c *Cloud) GetVolumeLabels(volumeName KubernetesVolumeID) (map[string]string, error) {
	awsDisk, err := newAWSDisk(c, volumeName)
	if err != nil {
		return nil, err
	}
	info, err := awsDisk.describeVolume()
	if err != nil {
		return nil, err
	}
	labels := make(map[string]string)
	az := aws.StringValue(info.AvailabilityZone)
	if az == "" {
		return nil, fmt.Errorf("volume did not have AZ information: %q", info.VolumeId)
	}

	labels[unversioned.LabelZoneFailureDomain] = az
	region, err := azToRegion(az)
	if err != nil {
		return nil, err
	}
	labels[unversioned.LabelZoneRegion] = region

	return labels, nil
}

// GetDiskPath implements Volumes.GetDiskPath
func (c *Cloud) GetDiskPath(volumeName KubernetesVolumeID) (string, error) {
	awsDisk, err := newAWSDisk(c, volumeName)
	if err != nil {
		return "", err
	}
	info, err := awsDisk.describeVolume()
	if err != nil {
		return "", err
	}
	if len(info.Attachments) == 0 {
		return "", fmt.Errorf("No attachment to volume %s", volumeName)
	}
	return aws.StringValue(info.Attachments[0].Device), nil
}

// DiskIsAttached implements Volumes.DiskIsAttached
func (c *Cloud) DiskIsAttached(diskName KubernetesVolumeID, nodeName types.NodeName) (bool, error) {
	awsInstance, err := c.getAwsInstance(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Instance %q does not exist. DiskIsAttached will assume disk %q is not attached to it.",
				nodeName,
				diskName)
			return false, nil
		}

		return false, err
	}

	diskID, err := diskName.mapToAWSVolumeID()
	if err != nil {
		return false, fmt.Errorf("error mapping volume spec %q to aws id: %v", diskName, err)
	}

	info, err := awsInstance.describeInstance()
	if err != nil {
		return false, err
	}
	for _, blockDevice := range info.BlockDeviceMappings {
		id := awsVolumeID(aws.StringValue(blockDevice.Ebs.VolumeId))
		if id == diskID {
			return true, nil
		}
	}
	return false, nil
}

func (c *Cloud) DisksAreAttached(diskNames []KubernetesVolumeID, nodeName types.NodeName) (map[KubernetesVolumeID]bool, error) {
	idToDiskName := make(map[awsVolumeID]KubernetesVolumeID)
	attached := make(map[KubernetesVolumeID]bool)
	for _, diskName := range diskNames {
		volumeID, err := diskName.mapToAWSVolumeID()
		if err != nil {
			return nil, fmt.Errorf("error mapping volume spec %q to aws id: %v", diskName, err)
		}
		idToDiskName[volumeID] = diskName
		attached[diskName] = false
	}
	awsInstance, err := c.getAwsInstance(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// If instance no longer exists, safe to assume volume is not attached.
			glog.Warningf(
				"Node %q does not exist. DisksAreAttached will assume disks %v are not attached to it.",
				nodeName,
				diskNames)
			return attached, nil
		}

		return attached, err
	}
	info, err := awsInstance.describeInstance()
	if err != nil {
		return attached, err
	}
	for _, blockDevice := range info.BlockDeviceMappings {
		volumeID := awsVolumeID(aws.StringValue(blockDevice.Ebs.VolumeId))
		diskName, found := idToDiskName[volumeID]
		if found {
			// Disk is still attached to node
			attached[diskName] = true
		}
	}

	return attached, nil
}

// Gets the current load balancer state
func (c *Cloud) describeLoadBalancer(name string) (*elb.LoadBalancerDescription, error) {
	request := &elb.DescribeLoadBalancersInput{}
	request.LoadBalancerNames = []*string{&name}

	response, err := c.elb.DescribeLoadBalancers(request)
	if err != nil {
		if awsError, ok := err.(awserr.Error); ok {
			if awsError.Code() == "LoadBalancerNotFound" {
				return nil, nil
			}
		}
		return nil, err
	}

	var ret *elb.LoadBalancerDescription
	for _, loadBalancer := range response.LoadBalancerDescriptions {
		if ret != nil {
			glog.Errorf("Found multiple load balancers with name: %s", name)
		}
		ret = loadBalancer
	}
	return ret, nil
}

// Retrieves instance's vpc id from metadata
func (c *Cloud) findVPCID() (string, error) {
	macs, err := c.metadata.GetMetadata("network/interfaces/macs/")
	if err != nil {
		return "", fmt.Errorf("Could not list interfaces of the instance: %v", err)
	}

	// loop over interfaces, first vpc id returned wins
	for _, macPath := range strings.Split(macs, "\n") {
		if len(macPath) == 0 {
			continue
		}
		url := fmt.Sprintf("network/interfaces/macs/%svpc-id", macPath)
		vpcID, err := c.metadata.GetMetadata(url)
		if err != nil {
			continue
		}
		return vpcID, nil
	}
	return "", fmt.Errorf("Could not find VPC ID in instance metadata")
}

// Retrieves the specified security group from the AWS API, or returns nil if not found
func (c *Cloud) findSecurityGroup(securityGroupID string) (*ec2.SecurityGroup, error) {
	describeSecurityGroupsRequest := &ec2.DescribeSecurityGroupsInput{
		GroupIds: []*string{&securityGroupID},
	}
	// We don't apply our tag filters because we are retrieving by ID

	groups, err := c.ec2.DescribeSecurityGroups(describeSecurityGroupsRequest)
	if err != nil {
		glog.Warningf("Error retrieving security group: %q", err)
		return nil, err
	}

	if len(groups) == 0 {
		return nil, nil
	}
	if len(groups) != 1 {
		// This should not be possible - ids should be unique
		return nil, fmt.Errorf("multiple security groups found with same id %q", securityGroupID)
	}
	group := groups[0]
	return group, nil
}

func isEqualIntPointer(l, r *int64) bool {
	if l == nil {
		return r == nil
	}
	if r == nil {
		return l == nil
	}
	return *l == *r
}

func isEqualStringPointer(l, r *string) bool {
	if l == nil {
		return r == nil
	}
	if r == nil {
		return l == nil
	}
	return *l == *r
}

func ipPermissionExists(newPermission, existing *ec2.IpPermission, compareGroupUserIDs bool) bool {
	if !isEqualIntPointer(newPermission.FromPort, existing.FromPort) {
		return false
	}
	if !isEqualIntPointer(newPermission.ToPort, existing.ToPort) {
		return false
	}
	if !isEqualStringPointer(newPermission.IpProtocol, existing.IpProtocol) {
		return false
	}
	// Check only if newPermission is a subset of existing. Usually it has zero or one elements.
	// Not doing actual CIDR math yet; not clear it's needed, either.
	glog.V(4).Infof("Comparing %v to %v", newPermission, existing)
	if len(newPermission.IpRanges) > len(existing.IpRanges) {
		return false
	}

	for j := range newPermission.IpRanges {
		found := false
		for k := range existing.IpRanges {
			if isEqualStringPointer(newPermission.IpRanges[j].CidrIp, existing.IpRanges[k].CidrIp) {
				found = true
				break
			}
		}
		if found == false {
			return false
		}
	}
	for _, leftPair := range newPermission.UserIdGroupPairs {
		for _, rightPair := range existing.UserIdGroupPairs {
			if isEqualUserGroupPair(leftPair, rightPair, compareGroupUserIDs) {
				return true
			}
		}
		return false
	}

	return true
}

func isEqualUserGroupPair(l, r *ec2.UserIdGroupPair, compareGroupUserIDs bool) bool {
	glog.V(2).Infof("Comparing %v to %v", *l.GroupId, *r.GroupId)
	if isEqualStringPointer(l.GroupId, r.GroupId) {
		if compareGroupUserIDs {
			if isEqualStringPointer(l.UserId, r.UserId) {
				return true
			}
		} else {
			return true
		}
	}

	return false
}

// Makes sure the security group ingress is exactly the specified permissions
// Returns true if and only if changes were made
// The security group must already exist
func (c *Cloud) setSecurityGroupIngress(securityGroupID string, permissions IPPermissionSet) (bool, error) {
	group, err := c.findSecurityGroup(securityGroupID)
	if err != nil {
		glog.Warning("Error retrieving security group", err)
		return false, err
	}

	if group == nil {
		return false, fmt.Errorf("security group not found: %s", securityGroupID)
	}

	glog.V(2).Infof("Existing security group ingress: %s %v", securityGroupID, group.IpPermissions)

	actual := NewIPPermissionSet(group.IpPermissions...)

	// EC2 groups rules together, for example combining:
	//
	// { Port=80, Range=[A] } and { Port=80, Range=[B] }
	//
	// into { Port=80, Range=[A,B] }
	//
	// We have to ungroup them, because otherwise the logic becomes really
	// complicated, and also because if we have Range=[A,B] and we try to
	// add Range=[A] then EC2 complains about a duplicate rule.
	permissions = permissions.Ungroup()
	actual = actual.Ungroup()

	remove := actual.Difference(permissions)
	add := permissions.Difference(actual)

	if add.Len() == 0 && remove.Len() == 0 {
		return false, nil
	}

	// TODO: There is a limit in VPC of 100 rules per security group, so we
	// probably should try grouping or combining to fit under this limit.
	// But this is only used on the ELB security group currently, so it
	// would require (ports * CIDRS) > 100.  Also, it isn't obvious exactly
	// how removing single permissions from compound rules works, and we
	// don't want to accidentally open more than intended while we're
	// applying changes.
	if add.Len() != 0 {
		glog.V(2).Infof("Adding security group ingress: %s %v", securityGroupID, add.List())

		request := &ec2.AuthorizeSecurityGroupIngressInput{}
		request.GroupId = &securityGroupID
		request.IpPermissions = add.List()
		_, err = c.ec2.AuthorizeSecurityGroupIngress(request)
		if err != nil {
			return false, fmt.Errorf("error authorizing security group ingress: %v", err)
		}
	}
	if remove.Len() != 0 {
		glog.V(2).Infof("Remove security group ingress: %s %v", securityGroupID, remove.List())

		request := &ec2.RevokeSecurityGroupIngressInput{}
		request.GroupId = &securityGroupID
		request.IpPermissions = remove.List()
		_, err = c.ec2.RevokeSecurityGroupIngress(request)
		if err != nil {
			return false, fmt.Errorf("error revoking security group ingress: %v", err)
		}
	}

	return true, nil
}

// Makes sure the security group includes the specified permissions
// Returns true if and only if changes were made
// The security group must already exist
func (c *Cloud) addSecurityGroupIngress(securityGroupID string, addPermissions []*ec2.IpPermission) (bool, error) {
	group, err := c.findSecurityGroup(securityGroupID)
	if err != nil {
		glog.Warningf("Error retrieving security group: %v", err)
		return false, err
	}

	if group == nil {
		return false, fmt.Errorf("security group not found: %s", securityGroupID)
	}

	glog.V(2).Infof("Existing security group ingress: %s %v", securityGroupID, group.IpPermissions)

	changes := []*ec2.IpPermission{}
	for _, addPermission := range addPermissions {
		hasUserID := false
		for i := range addPermission.UserIdGroupPairs {
			if addPermission.UserIdGroupPairs[i].UserId != nil {
				hasUserID = true
			}
		}

		found := false
		for _, groupPermission := range group.IpPermissions {
			if ipPermissionExists(addPermission, groupPermission, hasUserID) {
				found = true
				break
			}
		}

		if !found {
			changes = append(changes, addPermission)
		}
	}

	if len(changes) == 0 {
		return false, nil
	}

	glog.V(2).Infof("Adding security group ingress: %s %v", securityGroupID, changes)

	request := &ec2.AuthorizeSecurityGroupIngressInput{}
	request.GroupId = &securityGroupID
	request.IpPermissions = changes
	_, err = c.ec2.AuthorizeSecurityGroupIngress(request)
	if err != nil {
		glog.Warning("Error authorizing security group ingress", err)
		return false, fmt.Errorf("error authorizing security group ingress: %v", err)
	}

	return true, nil
}

// Makes sure the security group no longer includes the specified permissions
// Returns true if and only if changes were made
// If the security group no longer exists, will return (false, nil)
func (c *Cloud) removeSecurityGroupIngress(securityGroupID string, removePermissions []*ec2.IpPermission) (bool, error) {
	group, err := c.findSecurityGroup(securityGroupID)
	if err != nil {
		glog.Warningf("Error retrieving security group: %v", err)
		return false, err
	}

	if group == nil {
		glog.Warning("Security group not found: ", securityGroupID)
		return false, nil
	}

	changes := []*ec2.IpPermission{}
	for _, removePermission := range removePermissions {
		hasUserID := false
		for i := range removePermission.UserIdGroupPairs {
			if removePermission.UserIdGroupPairs[i].UserId != nil {
				hasUserID = true
			}
		}

		var found *ec2.IpPermission
		for _, groupPermission := range group.IpPermissions {
			if ipPermissionExists(removePermission, groupPermission, hasUserID) {
				found = removePermission
				break
			}
		}

		if found != nil {
			changes = append(changes, found)
		}
	}

	if len(changes) == 0 {
		return false, nil
	}

	glog.V(2).Infof("Removing security group ingress: %s %v", securityGroupID, changes)

	request := &ec2.RevokeSecurityGroupIngressInput{}
	request.GroupId = &securityGroupID
	request.IpPermissions = changes
	_, err = c.ec2.RevokeSecurityGroupIngress(request)
	if err != nil {
		glog.Warningf("Error revoking security group ingress: %v", err)
		return false, err
	}

	return true, nil
}

// Ensure that a resource has the correct tags
// If it has no tags, we assume that this was a problem caused by an error in between creation and tagging,
// and we add the tags.  If it has a different cluster's tags, that is an error.
func (c *Cloud) ensureClusterTags(resourceID string, tags []*ec2.Tag) error {
	actualTags := make(map[string]string)
	for _, tag := range tags {
		actualTags[aws.StringValue(tag.Key)] = aws.StringValue(tag.Value)
	}

	addTags := make(map[string]string)
	for k, expected := range c.filterTags {
		actual := actualTags[k]
		if actual == expected {
			continue
		}
		if actual == "" {
			glog.Warningf("Resource %q was missing expected cluster tag %q.  Will add (with value %q)", resourceID, k, expected)
			addTags[k] = expected
		} else {
			return fmt.Errorf("resource %q has tag belonging to another cluster: %q=%q (expected %q)", resourceID, k, actual, expected)
		}
	}

	if err := c.createTags(resourceID, addTags); err != nil {
		return fmt.Errorf("error adding missing tags to resource %q: %v", resourceID, err)
	}

	return nil
}

// Makes sure the security group exists.
// For multi-cluster isolation, name must be globally unique, for example derived from the service UUID.
// Returns the security group id or error
func (c *Cloud) ensureSecurityGroup(name string, description string) (string, error) {
	groupID := ""
	attempt := 0
	for {
		attempt++

		request := &ec2.DescribeSecurityGroupsInput{}
		filters := []*ec2.Filter{
			newEc2Filter("group-name", name),
			newEc2Filter("vpc-id", c.vpcID),
		}
		// Note that we do _not_ add our tag filters; group-name + vpc-id is the EC2 primary key.
		// However, we do check that it matches our tags.
		// If it doesn't have any tags, we tag it; this is how we recover if we failed to tag before.
		// If it has a different cluster's tags, that is an error.
		// This shouldn't happen because name is expected to be globally unique (UUID derived)
		request.Filters = filters

		securityGroups, err := c.ec2.DescribeSecurityGroups(request)
		if err != nil {
			return "", err
		}

		if len(securityGroups) >= 1 {
			if len(securityGroups) > 1 {
				glog.Warningf("Found multiple security groups with name: %q", name)
			}
			err := c.ensureClusterTags(aws.StringValue(securityGroups[0].GroupId), securityGroups[0].Tags)
			if err != nil {
				return "", err
			}

			return aws.StringValue(securityGroups[0].GroupId), nil
		}

		createRequest := &ec2.CreateSecurityGroupInput{}
		createRequest.VpcId = &c.vpcID
		createRequest.GroupName = &name
		createRequest.Description = &description

		createResponse, err := c.ec2.CreateSecurityGroup(createRequest)
		if err != nil {
			ignore := false
			switch err := err.(type) {
			case awserr.Error:
				if err.Code() == "InvalidGroup.Duplicate" && attempt < MaxReadThenCreateRetries {
					glog.V(2).Infof("Got InvalidGroup.Duplicate while creating security group (race?); will retry")
					ignore = true
				}
			}
			if !ignore {
				glog.Error("Error creating security group: ", err)
				return "", err
			}
			time.Sleep(1 * time.Second)
		} else {
			groupID = orEmpty(createResponse.GroupId)
			break
		}
	}
	if groupID == "" {
		return "", fmt.Errorf("created security group, but id was not returned: %s", name)
	}

	err := c.createTags(groupID, c.filterTags)
	if err != nil {
		// If we retry, ensureClusterTags will recover from this - it
		// will add the missing tags.  We could delete the security
		// group here, but that doesn't feel like the right thing, as
		// the caller is likely to retry the create
		return "", fmt.Errorf("error tagging security group: %v", err)
	}
	return groupID, nil
}

// createTags calls EC2 CreateTags, but adds retry-on-failure logic
// We retry mainly because if we create an object, we cannot tag it until it is "fully created" (eventual consistency)
// The error code varies though (depending on what we are tagging), so we simply retry on all errors
func (c *Cloud) createTags(resourceID string, tags map[string]string) error {
	if tags == nil || len(tags) == 0 {
		return nil
	}

	var awsTags []*ec2.Tag
	for k, v := range tags {
		tag := &ec2.Tag{
			Key:   aws.String(k),
			Value: aws.String(v),
		}
		awsTags = append(awsTags, tag)
	}

	request := &ec2.CreateTagsInput{}
	request.Resources = []*string{&resourceID}
	request.Tags = awsTags

	// TODO: We really should do exponential backoff here
	attempt := 0
	maxAttempts := 60

	for {
		_, err := c.ec2.CreateTags(request)
		if err == nil {
			return nil
		}

		// We could check that the error is retryable, but the error code changes based on what we are tagging
		// SecurityGroup: InvalidGroup.NotFound
		attempt++
		if attempt > maxAttempts {
			glog.Warningf("Failed to create tags (too many attempts): %v", err)
			return err
		}
		glog.V(2).Infof("Failed to create tags; will retry.  Error was %v", err)
		time.Sleep(1 * time.Second)
	}
}

// Finds the value for a given tag.
func findTag(tags []*ec2.Tag, key string) (string, bool) {
	for _, tag := range tags {
		if aws.StringValue(tag.Key) == key {
			return aws.StringValue(tag.Value), true
		}
	}
	return "", false
}

// Finds the subnets associated with the cluster, by matching tags.
// For maximal backwards compatibility, if no subnets are tagged, it will fall-back to the current subnet.
// However, in future this will likely be treated as an error.
func (c *Cloud) findSubnets() ([]*ec2.Subnet, error) {
	request := &ec2.DescribeSubnetsInput{}
	vpcIDFilter := newEc2Filter("vpc-id", c.vpcID)
	filters := []*ec2.Filter{vpcIDFilter}
	filters = c.addFilters(filters)
	request.Filters = filters

	subnets, err := c.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error describing subnets: %v", err)
	}

	if len(subnets) != 0 {
		return subnets, nil
	}

	// Fall back to the current instance subnets, if nothing is tagged
	glog.Warningf("No tagged subnets found; will fall-back to the current subnet only.  This is likely to be an error in a future version of k8s.")

	request = &ec2.DescribeSubnetsInput{}
	filters = []*ec2.Filter{newEc2Filter("subnet-id", c.selfAWSInstance.subnetID)}
	request.Filters = filters

	subnets, err = c.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error describing subnets: %v", err)
	}

	return subnets, nil
}

// Finds the subnets to use for an ELB we are creating.
// Normal (Internet-facing) ELBs must use public subnets, so we skip private subnets.
// Internal ELBs can use public or private subnets, but if we have a private subnet we should prefer that.
func (c *Cloud) findELBSubnets(internalELB bool) ([]string, error) {
	vpcIDFilter := newEc2Filter("vpc-id", c.vpcID)

	subnets, err := c.findSubnets()
	if err != nil {
		return nil, err
	}

	rRequest := &ec2.DescribeRouteTablesInput{}
	rRequest.Filters = []*ec2.Filter{vpcIDFilter}
	rt, err := c.ec2.DescribeRouteTables(rRequest)
	if err != nil {
		return nil, fmt.Errorf("error describe route table: %v", err)
	}

	subnetsByAZ := make(map[string]*ec2.Subnet)
	for _, subnet := range subnets {
		az := aws.StringValue(subnet.AvailabilityZone)
		id := aws.StringValue(subnet.SubnetId)
		if az == "" || id == "" {
			glog.Warningf("Ignoring subnet with empty az/id: %v", subnet)
			continue
		}

		isPublic, err := isSubnetPublic(rt, id)
		if err != nil {
			return nil, err
		}
		if !internalELB && !isPublic {
			glog.V(2).Infof("Ignoring private subnet for public ELB %q", id)
			continue
		}

		existing := subnetsByAZ[az]
		if existing == nil {
			subnetsByAZ[az] = subnet
			continue
		}

		// Try to break the tie using a tag
		var tagName string
		if internalELB {
			tagName = TagNameSubnetInternalELB
		} else {
			tagName = TagNameSubnetPublicELB
		}

		_, existingHasTag := findTag(existing.Tags, tagName)
		_, subnetHasTag := findTag(subnet.Tags, tagName)

		if existingHasTag != subnetHasTag {
			if subnetHasTag {
				subnetsByAZ[az] = subnet
			}
			continue
		}

		// TODO: Should this be an error?
		glog.Warningf("Found multiple subnets in AZ %q; making arbitrary choice between subnets %q and %q", az, *existing.SubnetId, *subnet.SubnetId)
		continue
	}

	var subnetIDs []string
	for _, subnet := range subnetsByAZ {
		subnetIDs = append(subnetIDs, aws.StringValue(subnet.SubnetId))
	}

	return subnetIDs, nil
}

func isSubnetPublic(rt []*ec2.RouteTable, subnetID string) (bool, error) {
	var subnetTable *ec2.RouteTable
	for _, table := range rt {
		for _, assoc := range table.Associations {
			if aws.StringValue(assoc.SubnetId) == subnetID {
				subnetTable = table
				break
			}
		}
	}

	if subnetTable == nil {
		// If there is no explicit association, the subnet will be implicitly
		// associated with the VPC's main routing table.
		for _, table := range rt {
			for _, assoc := range table.Associations {
				if aws.BoolValue(assoc.Main) == true {
					glog.V(4).Infof("Assuming implicit use of main routing table %s for %s",
						aws.StringValue(table.RouteTableId), subnetID)
					subnetTable = table
					break
				}
			}
		}
	}

	if subnetTable == nil {
		return false, fmt.Errorf("Could not locate routing table for subnet %s", subnetID)
	}

	for _, route := range subnetTable.Routes {
		// There is no direct way in the AWS API to determine if a subnet is public or private.
		// A public subnet is one which has an internet gateway route
		// we look for the gatewayId and make sure it has the prefix of igw to differentiate
		// from the default in-subnet route which is called "local"
		// or other virtual gateway (starting with vgv)
		// or vpc peering connections (starting with pcx).
		if strings.HasPrefix(aws.StringValue(route.GatewayId), "igw") {
			return true, nil
		}
	}

	return false, nil
}

type portSets struct {
	names   sets.String
	numbers sets.Int64
}

// getPortSets returns a portSets structure representing port names and numbers
// that the comma-separated string describes. If the input is empty or equal to
// "*", a nil pointer is returned.
func getPortSets(annotation string) (ports *portSets) {
	if annotation != "" && annotation != "*" {
		ports = &portSets{
			sets.NewString(),
			sets.NewInt64(),
		}
		portStringSlice := strings.Split(annotation, ",")
		for _, item := range portStringSlice {
			port, err := strconv.Atoi(item)
			if err != nil {
				ports.names.Insert(item)
			} else {
				ports.numbers.Insert(int64(port))
			}
		}
	}
	return
}

// buildListener creates a new listener from the given port, adding an SSL certificate
// if indicated by the appropriate annotations.
func buildListener(port api.ServicePort, annotations map[string]string, sslPorts *portSets) (*elb.Listener, error) {
	loadBalancerPort := int64(port.Port)
	portName := strings.ToLower(port.Name)
	instancePort := int64(port.NodePort)
	protocol := strings.ToLower(string(port.Protocol))
	instanceProtocol := protocol

	listener := &elb.Listener{}
	listener.InstancePort = &instancePort
	listener.LoadBalancerPort = &loadBalancerPort
	certID := annotations[ServiceAnnotationLoadBalancerCertificate]
	if certID != "" && (sslPorts == nil || sslPorts.numbers.Has(loadBalancerPort) || sslPorts.names.Has(portName)) {
		instanceProtocol = annotations[ServiceAnnotationLoadBalancerBEProtocol]
		if instanceProtocol == "" {
			protocol = "ssl"
			instanceProtocol = "tcp"
		} else {
			protocol = backendProtocolMapping[instanceProtocol]
			if protocol == "" {
				return nil, fmt.Errorf("Invalid backend protocol %s for %s in %s", instanceProtocol, certID, ServiceAnnotationLoadBalancerBEProtocol)
			}
		}
		listener.SSLCertificateId = &certID
	} else if annotationProtocol := annotations[ServiceAnnotationLoadBalancerBEProtocol]; annotationProtocol == "http" {
		instanceProtocol = annotationProtocol
		protocol = "http"
	}

	listener.Protocol = &protocol
	listener.InstanceProtocol = &instanceProtocol

	return listener, nil
}

// EnsureLoadBalancer implements LoadBalancer.EnsureLoadBalancer
func (c *Cloud) EnsureLoadBalancer(clusterName string, apiService *api.Service, hosts []string) (*api.LoadBalancerStatus, error) {
	annotations := apiService.Annotations
	glog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v, %v)",
		clusterName, apiService.Namespace, apiService.Name, c.region, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, hosts, annotations)

	if apiService.Spec.SessionAffinity != api.ServiceAffinityNone {
		// ELB supports sticky sessions, but only when configured for HTTP/HTTPS
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", apiService.Spec.SessionAffinity)
	}

	if len(apiService.Spec.Ports) == 0 {
		return nil, fmt.Errorf("requested load balancer with no ports")
	}

	// Figure out what mappings we want on the load balancer
	listeners := []*elb.Listener{}
	portList := getPortSets(annotations[ServiceAnnotationLoadBalancerSSLPorts])
	for _, port := range apiService.Spec.Ports {
		if port.Protocol != api.ProtocolTCP {
			return nil, fmt.Errorf("Only TCP LoadBalancer is supported for AWS ELB")
		}
		if port.NodePort == 0 {
			glog.Errorf("Ignoring port without NodePort defined: %v", port)
			continue
		}
		listener, err := buildListener(port, annotations, portList)
		if err != nil {
			return nil, err
		}
		listeners = append(listeners, listener)
	}

	if apiService.Spec.LoadBalancerIP != "" {
		return nil, fmt.Errorf("LoadBalancerIP cannot be specified for AWS ELB")
	}

	hostSet := sets.NewString(hosts...)
	instances, err := c.getInstancesByNodeNamesCached(hostSet)
	if err != nil {
		return nil, err
	}

	sourceRanges, err := service.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	// Determine if this is tagged as an Internal ELB
	internalELB := false
	internalAnnotation := apiService.Annotations[ServiceAnnotationLoadBalancerInternal]
	if internalAnnotation != "" {
		if internalAnnotation != "0.0.0.0/0" {
			return nil, fmt.Errorf("annotation %q=%q detected, but the only value supported currently is 0.0.0.0/0", ServiceAnnotationLoadBalancerInternal, internalAnnotation)
		}
		if !service.IsAllowAll(sourceRanges) {
			// TODO: Unify the two annotations
			return nil, fmt.Errorf("source-range annotation cannot be combined with the internal-elb annotation")
		}
		internalELB = true
	}

	// Determine if we need to set the Proxy protocol policy
	proxyProtocol := false
	proxyProtocolAnnotation := apiService.Annotations[ServiceAnnotationLoadBalancerProxyProtocol]
	if proxyProtocolAnnotation != "" {
		if proxyProtocolAnnotation != "*" {
			return nil, fmt.Errorf("annotation %q=%q detected, but the only value supported currently is '*'", ServiceAnnotationLoadBalancerProxyProtocol, proxyProtocolAnnotation)
		}
		proxyProtocol = true
	}

	// Some load balancer attributes are required, so defaults are set. These can be overridden by annotations.
	loadBalancerAttributes := &elb.LoadBalancerAttributes{
		AccessLog:              &elb.AccessLog{Enabled: aws.Bool(false)},
		ConnectionDraining:     &elb.ConnectionDraining{Enabled: aws.Bool(false)},
		ConnectionSettings:     &elb.ConnectionSettings{IdleTimeout: aws.Int64(60)},
		CrossZoneLoadBalancing: &elb.CrossZoneLoadBalancing{Enabled: aws.Bool(false)},
	}

	// Determine if an access log emit interval has been specified
	accessLogEmitIntervalAnnotation := annotations[ServiceAnnotationLoadBalancerAccessLogEmitInterval]
	if accessLogEmitIntervalAnnotation != "" {
		accessLogEmitInterval, err := strconv.ParseInt(accessLogEmitIntervalAnnotation, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerAccessLogEmitInterval,
				accessLogEmitIntervalAnnotation,
			)
		}
		loadBalancerAttributes.AccessLog.EmitInterval = &accessLogEmitInterval
	}

	// Determine if access log enabled/disabled has been specified
	accessLogEnabledAnnotation := annotations[ServiceAnnotationLoadBalancerAccessLogEnabled]
	if accessLogEnabledAnnotation != "" {
		accessLogEnabled, err := strconv.ParseBool(accessLogEnabledAnnotation)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerAccessLogEnabled,
				accessLogEnabledAnnotation,
			)
		}
		loadBalancerAttributes.AccessLog.Enabled = &accessLogEnabled
	}

	// Determine if access log s3 bucket name has been specified
	accessLogS3BucketNameAnnotation := annotations[ServiceAnnotationLoadBalancerAccessLogS3BucketName]
	if accessLogS3BucketNameAnnotation != "" {
		loadBalancerAttributes.AccessLog.S3BucketName = &accessLogS3BucketNameAnnotation
	}

	// Determine if access log s3 bucket prefix has been specified
	accessLogS3BucketPrefixAnnotation := annotations[ServiceAnnotationLoadBalancerAccessLogS3BucketPrefix]
	if accessLogS3BucketPrefixAnnotation != "" {
		loadBalancerAttributes.AccessLog.S3BucketPrefix = &accessLogS3BucketPrefixAnnotation
	}

	// Determine if connection draining enabled/disabled has been specified
	connectionDrainingEnabledAnnotation := annotations[ServiceAnnotationLoadBalancerConnectionDrainingEnabled]
	if connectionDrainingEnabledAnnotation != "" {
		connectionDrainingEnabled, err := strconv.ParseBool(connectionDrainingEnabledAnnotation)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerConnectionDrainingEnabled,
				connectionDrainingEnabledAnnotation,
			)
		}
		loadBalancerAttributes.ConnectionDraining.Enabled = &connectionDrainingEnabled
	}

	// Determine if connection draining timeout has been specified
	connectionDrainingTimeoutAnnotation := annotations[ServiceAnnotationLoadBalancerConnectionDrainingTimeout]
	if connectionDrainingTimeoutAnnotation != "" {
		connectionDrainingTimeout, err := strconv.ParseInt(connectionDrainingTimeoutAnnotation, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerConnectionDrainingTimeout,
				connectionDrainingTimeoutAnnotation,
			)
		}
		loadBalancerAttributes.ConnectionDraining.Timeout = &connectionDrainingTimeout
	}

	// Determine if connection idle timeout has been specified
	connectionIdleTimeoutAnnotation := annotations[ServiceAnnotationLoadBalancerConnectionIdleTimeout]
	if connectionIdleTimeoutAnnotation != "" {
		connectionIdleTimeout, err := strconv.ParseInt(connectionIdleTimeoutAnnotation, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerConnectionIdleTimeout,
				connectionIdleTimeoutAnnotation,
			)
		}
		loadBalancerAttributes.ConnectionSettings.IdleTimeout = &connectionIdleTimeout
	}

	// Determine if cross zone load balancing enabled/disabled has been specified
	crossZoneLoadBalancingEnabledAnnotation := annotations[ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled]
	if crossZoneLoadBalancingEnabledAnnotation != "" {
		crossZoneLoadBalancingEnabled, err := strconv.ParseBool(crossZoneLoadBalancingEnabledAnnotation)
		if err != nil {
			return nil, fmt.Errorf("error parsing service annotation: %s=%s",
				ServiceAnnotationLoadBalancerCrossZoneLoadBalancingEnabled,
				crossZoneLoadBalancingEnabledAnnotation,
			)
		}
		loadBalancerAttributes.CrossZoneLoadBalancing.Enabled = &crossZoneLoadBalancingEnabled
	}

	// Find the subnets that the ELB will live in
	subnetIDs, err := c.findELBSubnets(internalELB)
	if err != nil {
		glog.Error("Error listing subnets in VPC: ", err)
		return nil, err
	}

	// Bail out early if there are no subnets
	if len(subnetIDs) == 0 {
		return nil, fmt.Errorf("could not find any suitable subnets for creating the ELB")
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(apiService)
	serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}

	// Create a security group for the load balancer
	var securityGroupID string
	{
		sgName := "k8s-elb-" + loadBalancerName
		sgDescription := fmt.Sprintf("Security group for Kubernetes ELB %s (%v)", loadBalancerName, serviceName)
		securityGroupID, err = c.ensureSecurityGroup(sgName, sgDescription)
		if err != nil {
			glog.Error("Error creating load balancer security group: ", err)
			return nil, err
		}

		ec2SourceRanges := []*ec2.IpRange{}
		for _, sourceRange := range sourceRanges.StringSlice() {
			ec2SourceRanges = append(ec2SourceRanges, &ec2.IpRange{CidrIp: aws.String(sourceRange)})
		}

		permissions := NewIPPermissionSet()
		for _, port := range apiService.Spec.Ports {
			portInt64 := int64(port.Port)
			protocol := strings.ToLower(string(port.Protocol))

			permission := &ec2.IpPermission{}
			permission.FromPort = &portInt64
			permission.ToPort = &portInt64
			permission.IpRanges = ec2SourceRanges
			permission.IpProtocol = &protocol

			permissions.Insert(permission)
		}

		// Allow ICMP fragmentation packets, important for MTU discovery
		{
			permission := &ec2.IpPermission{
				IpProtocol: aws.String("icmp"),
				FromPort:   aws.Int64(3),
				ToPort:     aws.Int64(4),
				IpRanges:   []*ec2.IpRange{{CidrIp: aws.String("0.0.0.0/0")}},
			}

			permissions.Insert(permission)
		}

		_, err = c.setSecurityGroupIngress(securityGroupID, permissions)
		if err != nil {
			return nil, err
		}
	}
	securityGroupIDs := []string{securityGroupID}

	// Build the load balancer itself
	loadBalancer, err := c.ensureLoadBalancer(
		serviceName,
		loadBalancerName,
		listeners,
		subnetIDs,
		securityGroupIDs,
		internalELB,
		proxyProtocol,
		loadBalancerAttributes,
	)
	if err != nil {
		return nil, err
	}

	err = c.ensureLoadBalancerHealthCheck(loadBalancer, listeners)
	if err != nil {
		return nil, err
	}

	err = c.updateInstanceSecurityGroupsForLoadBalancer(loadBalancer, instances)
	if err != nil {
		glog.Warningf("Error opening ingress rules for the load balancer to the instances: %v", err)
		return nil, err
	}

	err = c.ensureLoadBalancerInstances(orEmpty(loadBalancer.LoadBalancerName), loadBalancer.Instances, instances)
	if err != nil {
		glog.Warningf("Error registering instances with the load balancer: %v", err)
		return nil, err
	}

	glog.V(1).Infof("Loadbalancer %s (%v) has DNS name %s", loadBalancerName, serviceName, orEmpty(loadBalancer.DNSName))

	// TODO: Wait for creation?

	status := toStatus(loadBalancer)
	return status, nil
}

// GetLoadBalancer is an implementation of LoadBalancer.GetLoadBalancer
func (c *Cloud) GetLoadBalancer(clusterName string, service *api.Service) (*api.LoadBalancerStatus, bool, error) {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	lb, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return nil, false, err
	}

	if lb == nil {
		return nil, false, nil
	}

	status := toStatus(lb)
	return status, true, nil
}

func toStatus(lb *elb.LoadBalancerDescription) *api.LoadBalancerStatus {
	status := &api.LoadBalancerStatus{}

	if !isNilOrEmpty(lb.DNSName) {
		var ingress api.LoadBalancerIngress
		ingress.Hostname = orEmpty(lb.DNSName)
		status.Ingress = []api.LoadBalancerIngress{ingress}
	}

	return status
}

// Returns the first security group for an instance, or nil
// We only create instances with one security group, so we don't expect multiple security groups.
// However, if there are multiple security groups, we will choose the one tagged with our cluster filter.
// Otherwise we will return an error.
func findSecurityGroupForInstance(instance *ec2.Instance, taggedSecurityGroups map[string]*ec2.SecurityGroup) (*ec2.GroupIdentifier, error) {
	instanceID := aws.StringValue(instance.InstanceId)

	var tagged []*ec2.GroupIdentifier
	var untagged []*ec2.GroupIdentifier
	for _, group := range instance.SecurityGroups {
		groupID := aws.StringValue(group.GroupId)
		if groupID == "" {
			glog.Warningf("Ignoring security group without id for instance %q: %v", instanceID, group)
			continue
		}
		_, isTagged := taggedSecurityGroups[groupID]
		if isTagged {
			tagged = append(tagged, group)
		} else {
			untagged = append(untagged, group)
		}
	}

	if len(tagged) > 0 {
		// We create instances with one SG
		// If users create multiple SGs, they must tag one of them as being k8s owned
		if len(tagged) != 1 {
			return nil, fmt.Errorf("Multiple tagged security groups found for instance %s; ensure only the k8s security group is tagged", instanceID)
		}
		return tagged[0], nil
	}

	if len(untagged) > 0 {
		// For back-compat, we will allow a single untagged SG
		if len(untagged) != 1 {
			return nil, fmt.Errorf("Multiple untagged security groups found for instance %s; ensure the k8s security group is tagged", instanceID)
		}
		return untagged[0], nil
	}

	glog.Warningf("No security group found for instance %q", instanceID)
	return nil, nil
}

// Return all the security groups that are tagged as being part of our cluster
func (c *Cloud) getTaggedSecurityGroups() (map[string]*ec2.SecurityGroup, error) {
	request := &ec2.DescribeSecurityGroupsInput{}
	request.Filters = c.addFilters(nil)
	groups, err := c.ec2.DescribeSecurityGroups(request)
	if err != nil {
		return nil, fmt.Errorf("error querying security groups: %v", err)
	}

	m := make(map[string]*ec2.SecurityGroup)
	for _, group := range groups {
		id := aws.StringValue(group.GroupId)
		if id == "" {
			glog.Warningf("Ignoring group without id: %v", group)
			continue
		}
		m[id] = group
	}
	return m, nil
}

// Open security group ingress rules on the instances so that the load balancer can talk to them
// Will also remove any security groups ingress rules for the load balancer that are _not_ needed for allInstances
func (c *Cloud) updateInstanceSecurityGroupsForLoadBalancer(lb *elb.LoadBalancerDescription, allInstances []*ec2.Instance) error {
	if c.cfg.Global.DisableSecurityGroupIngress {
		return nil
	}

	// Determine the load balancer security group id
	loadBalancerSecurityGroupID := ""
	for _, securityGroup := range lb.SecurityGroups {
		if isNilOrEmpty(securityGroup) {
			continue
		}
		if loadBalancerSecurityGroupID != "" {
			// We create LBs with one SG
			glog.Warningf("Multiple security groups for load balancer: %q", orEmpty(lb.LoadBalancerName))
		}
		loadBalancerSecurityGroupID = *securityGroup
	}
	if loadBalancerSecurityGroupID == "" {
		return fmt.Errorf("Could not determine security group for load balancer: %s", orEmpty(lb.LoadBalancerName))
	}

	// Get the actual list of groups that allow ingress from the load-balancer
	describeRequest := &ec2.DescribeSecurityGroupsInput{}
	filters := []*ec2.Filter{}
	filters = append(filters, newEc2Filter("ip-permission.group-id", loadBalancerSecurityGroupID))
	describeRequest.Filters = c.addFilters(filters)
	actualGroups, err := c.ec2.DescribeSecurityGroups(describeRequest)
	if err != nil {
		return fmt.Errorf("error querying security groups for ELB: %v", err)
	}

	taggedSecurityGroups, err := c.getTaggedSecurityGroups()
	if err != nil {
		return fmt.Errorf("error querying for tagged security groups: %v", err)
	}

	// Open the firewall from the load balancer to the instance
	// We don't actually have a trivial way to know in advance which security group the instance is in
	// (it is probably the node security group, but we don't easily have that).
	// However, we _do_ have the list of security groups on the instance records.

	// Map containing the changes we want to make; true to add, false to remove
	instanceSecurityGroupIds := map[string]bool{}

	// Scan instances for groups we want open
	for _, instance := range allInstances {
		securityGroup, err := findSecurityGroupForInstance(instance, taggedSecurityGroups)
		if err != nil {
			return err
		}

		if securityGroup == nil {
			glog.Warning("Ignoring instance without security group: ", orEmpty(instance.InstanceId))
			continue
		}
		id := aws.StringValue(securityGroup.GroupId)
		if id == "" {
			glog.Warningf("found security group without id: %v", securityGroup)
			continue
		}

		instanceSecurityGroupIds[id] = true
	}

	// Compare to actual groups
	for _, actualGroup := range actualGroups {
		actualGroupID := aws.StringValue(actualGroup.GroupId)
		if actualGroupID == "" {
			glog.Warning("Ignoring group without ID: ", actualGroup)
			continue
		}

		adding, found := instanceSecurityGroupIds[actualGroupID]
		if found && adding {
			// We don't need to make a change; the permission is already in place
			delete(instanceSecurityGroupIds, actualGroupID)
		} else {
			// This group is not needed by allInstances; delete it
			instanceSecurityGroupIds[actualGroupID] = false
		}
	}

	for instanceSecurityGroupID, add := range instanceSecurityGroupIds {
		if add {
			glog.V(2).Infof("Adding rule for traffic from the load balancer (%s) to instances (%s)", loadBalancerSecurityGroupID, instanceSecurityGroupID)
		} else {
			glog.V(2).Infof("Removing rule for traffic from the load balancer (%s) to instance (%s)", loadBalancerSecurityGroupID, instanceSecurityGroupID)
		}
		sourceGroupID := &ec2.UserIdGroupPair{}
		sourceGroupID.GroupId = &loadBalancerSecurityGroupID

		allProtocols := "-1"

		permission := &ec2.IpPermission{}
		permission.IpProtocol = &allProtocols
		permission.UserIdGroupPairs = []*ec2.UserIdGroupPair{sourceGroupID}

		permissions := []*ec2.IpPermission{permission}

		if add {
			changed, err := c.addSecurityGroupIngress(instanceSecurityGroupID, permissions)
			if err != nil {
				return err
			}
			if !changed {
				glog.Warning("Allowing ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		} else {
			changed, err := c.removeSecurityGroupIngress(instanceSecurityGroupID, permissions)
			if err != nil {
				return err
			}
			if !changed {
				glog.Warning("Revoking ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		}
	}

	return nil
}

// EnsureLoadBalancerDeleted implements LoadBalancer.EnsureLoadBalancerDeleted.
func (c *Cloud) EnsureLoadBalancerDeleted(clusterName string, service *api.Service) error {
	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	lb, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return err
	}

	if lb == nil {
		glog.Info("Load balancer already deleted: ", loadBalancerName)
		return nil
	}

	{
		// De-authorize the load balancer security group from the instances security group
		err = c.updateInstanceSecurityGroupsForLoadBalancer(lb, nil)
		if err != nil {
			glog.Error("Error deregistering load balancer from instance security groups: ", err)
			return err
		}
	}

	{
		// Delete the load balancer itself
		request := &elb.DeleteLoadBalancerInput{}
		request.LoadBalancerName = lb.LoadBalancerName

		_, err = c.elb.DeleteLoadBalancer(request)
		if err != nil {
			// TODO: Check if error was because load balancer was concurrently deleted
			glog.Error("Error deleting load balancer: ", err)
			return err
		}
	}

	{
		// Delete the security group(s) for the load balancer
		// Note that this is annoying: the load balancer disappears from the API immediately, but it is still
		// deleting in the background.  We get a DependencyViolation until the load balancer has deleted itself

		// Collect the security groups to delete
		securityGroupIDs := map[string]struct{}{}
		for _, securityGroupID := range lb.SecurityGroups {
			if isNilOrEmpty(securityGroupID) {
				glog.Warning("Ignoring empty security group in ", service.Name)
				continue
			}
			securityGroupIDs[*securityGroupID] = struct{}{}
		}

		// Loop through and try to delete them
		timeoutAt := time.Now().Add(time.Second * 600)
		for {
			for securityGroupID := range securityGroupIDs {
				request := &ec2.DeleteSecurityGroupInput{}
				request.GroupId = &securityGroupID
				_, err := c.ec2.DeleteSecurityGroup(request)
				if err == nil {
					delete(securityGroupIDs, securityGroupID)
				} else {
					ignore := false
					if awsError, ok := err.(awserr.Error); ok {
						if awsError.Code() == "DependencyViolation" {
							glog.V(2).Infof("Ignoring DependencyViolation while deleting load-balancer security group (%s), assuming because LB is in process of deleting", securityGroupID)
							ignore = true
						}
					}
					if !ignore {
						return fmt.Errorf("error while deleting load balancer security group (%s): %v", securityGroupID, err)
					}
				}
			}

			if len(securityGroupIDs) == 0 {
				glog.V(2).Info("Deleted all security groups for load balancer: ", service.Name)
				break
			}

			if time.Now().After(timeoutAt) {
				ids := []string{}
				for id := range securityGroupIDs {
					ids = append(ids, id)
				}

				return fmt.Errorf("timed out deleting ELB: %s. Could not delete security groups %v", service.Name, strings.Join(ids, ","))
			}

			glog.V(2).Info("Waiting for load-balancer to delete so we can delete security groups: ", service.Name)

			time.Sleep(10 * time.Second)
		}
	}

	return nil
}

// UpdateLoadBalancer implements LoadBalancer.UpdateLoadBalancer
func (c *Cloud) UpdateLoadBalancer(clusterName string, service *api.Service, hosts []string) error {
	hostSet := sets.NewString(hosts...)
	instances, err := c.getInstancesByNodeNamesCached(hostSet)
	if err != nil {
		return err
	}

	loadBalancerName := cloudprovider.GetLoadBalancerName(service)
	lb, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return err
	}

	if lb == nil {
		return fmt.Errorf("Load balancer not found")
	}

	err = c.ensureLoadBalancerInstances(orEmpty(lb.LoadBalancerName), lb.Instances, instances)
	if err != nil {
		return nil
	}

	err = c.updateInstanceSecurityGroupsForLoadBalancer(lb, instances)
	if err != nil {
		return err
	}

	return nil
}

// Returns the instance with the specified ID
func (c *Cloud) getInstanceByID(instanceID string) (*ec2.Instance, error) {
	instances, err := c.getInstancesByIDs([]*string{&instanceID})
	if err != nil {
		return nil, err
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances found for instance: %s", instanceID)
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}

	return instances[instanceID], nil
}

func (c *Cloud) getInstancesByIDs(instanceIDs []*string) (map[string]*ec2.Instance, error) {
	instancesByID := make(map[string]*ec2.Instance)
	if len(instanceIDs) == 0 {
		return instancesByID, nil
	}

	request := &ec2.DescribeInstancesInput{
		InstanceIds: instanceIDs,
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}

	for _, instance := range instances {
		instanceID := orEmpty(instance.InstanceId)
		if instanceID == "" {
			continue
		}

		instancesByID[instanceID] = instance
	}

	return instancesByID, nil
}

// Fetches and caches instances by node names; returns an error if any cannot be found.
// This is implemented with a multi value filter on the node names, fetching the desired instances with a single query.
// TODO(therc): make all the caching more rational during the 1.4 timeframe
func (c *Cloud) getInstancesByNodeNamesCached(nodeNames sets.String) ([]*ec2.Instance, error) {
	c.mutex.Lock()
	defer c.mutex.Unlock()
	if nodeNames.Equal(c.lastNodeNames) {
		if len(c.lastInstancesByNodeNames) > 0 {
			// We assume that if the list of nodes is the same, the underlying
			// instances have not changed. Later we might guard this with TTLs.
			glog.V(2).Infof("Returning cached instances for %v", nodeNames)
			return c.lastInstancesByNodeNames, nil
		}
	}
	names := aws.StringSlice(nodeNames.List())

	nodeNameFilter := &ec2.Filter{
		Name:   aws.String("private-dns-name"),
		Values: names,
	}

	filters := []*ec2.Filter{
		nodeNameFilter,
		newEc2Filter("instance-state-name", "running"),
	}

	filters = c.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		glog.V(2).Infof("Failed to describe instances %v", nodeNames)
		return nil, err
	}

	if len(instances) == 0 {
		glog.V(3).Infof("Failed to find any instances %v", nodeNames)
		return nil, nil
	}

	glog.V(2).Infof("Caching instances for %v", nodeNames)
	c.lastNodeNames = nodeNames
	c.lastInstancesByNodeNames = instances
	return instances, nil
}

// mapNodeNameToPrivateDNSName maps a k8s NodeName to an AWS Instance PrivateDNSName
// This is a simple string cast
func mapNodeNameToPrivateDNSName(nodeName types.NodeName) string {
	return string(nodeName)
}

// mapInstanceToNodeName maps a EC2 instance to a k8s NodeName, by extracting the PrivateDNSName
func mapInstanceToNodeName(i *ec2.Instance) types.NodeName {
	return types.NodeName(aws.StringValue(i.PrivateDnsName))
}

// Returns the instance with the specified node name
// Returns nil if it does not exist
func (c *Cloud) findInstanceByNodeName(nodeName types.NodeName) (*ec2.Instance, error) {
	privateDNSName := mapNodeNameToPrivateDNSName(nodeName)
	filters := []*ec2.Filter{
		newEc2Filter("private-dns-name", privateDNSName),
		newEc2Filter("instance-state-name", "running"),
	}
	filters = c.addFilters(filters)
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}
	if len(instances) == 0 {
		return nil, nil
	}
	if len(instances) > 1 {
		return nil, fmt.Errorf("multiple instances found for name: %s", nodeName)
	}
	return instances[0], nil
}

// Returns the instance with the specified node name
// Like findInstanceByNodeName, but returns error if node not found
func (c *Cloud) getInstanceByNodeName(nodeName types.NodeName) (*ec2.Instance, error) {
	instance, err := c.findInstanceByNodeName(nodeName)
	if err == nil && instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}
	return instance, err
}

// Add additional filters, to match on our tags
// This lets us run multiple k8s clusters in a single EC2 AZ
func (c *Cloud) addFilters(filters []*ec2.Filter) []*ec2.Filter {
	for k, v := range c.filterTags {
		filters = append(filters, newEc2Filter("tag:"+k, v))
	}
	if len(filters) == 0 {
		// We can't pass a zero-length Filters to AWS (it's an error)
		// So if we end up with no filters; just return nil
		return nil
	}

	return filters
}

// Returns the cluster name or an empty string
func (c *Cloud) getClusterName() string {
	return c.filterTags[TagNameKubernetesCluster]
}
