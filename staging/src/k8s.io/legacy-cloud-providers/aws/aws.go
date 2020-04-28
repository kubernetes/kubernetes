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
	"context"
	"errors"
	"fmt"
	"io"
	"net"
	"path"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/aws/aws-sdk-go/aws"
	"github.com/aws/aws-sdk-go/aws/awserr"
	"github.com/aws/aws-sdk-go/aws/credentials"
	"github.com/aws/aws-sdk-go/aws/credentials/ec2rolecreds"
	"github.com/aws/aws-sdk-go/aws/credentials/stscreds"
	"github.com/aws/aws-sdk-go/aws/ec2metadata"
	"github.com/aws/aws-sdk-go/aws/endpoints"
	"github.com/aws/aws-sdk-go/aws/request"
	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/autoscaling"
	"github.com/aws/aws-sdk-go/service/ec2"
	"github.com/aws/aws-sdk-go/service/elb"
	"github.com/aws/aws-sdk-go/service/elbv2"
	"github.com/aws/aws-sdk-go/service/kms"
	"github.com/aws/aws-sdk-go/service/sts"
	"gopkg.in/gcfg.v1"
	"k8s.io/api/core/v1"
	"k8s.io/klog"

	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/client-go/informers"
	informercorev1 "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	v1core "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/pkg/version"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/record"
	"k8s.io/cloud-provider"
	nodehelpers "k8s.io/cloud-provider/node/helpers"
	servicehelpers "k8s.io/cloud-provider/service/helpers"
	cloudvolume "k8s.io/cloud-provider/volume"
	volerr "k8s.io/cloud-provider/volume/errors"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
)

// NLBHealthCheckRuleDescription is the comment used on a security group rule to
// indicate that it is used for health checks
const NLBHealthCheckRuleDescription = "kubernetes.io/rule/nlb/health"

// NLBClientRuleDescription is the comment used on a security group rule to
// indicate that it is used for client traffic
const NLBClientRuleDescription = "kubernetes.io/rule/nlb/client"

// NLBMtuDiscoveryRuleDescription is the comment used on a security group rule
// to indicate that it is used for mtu discovery
const NLBMtuDiscoveryRuleDescription = "kubernetes.io/rule/nlb/mtu"

// ProviderName is the name of this cloud provider.
const ProviderName = "aws"

// TagNameKubernetesService is the tag name we use to differentiate multiple
// services. Used currently for ELBs only.
const TagNameKubernetesService = "kubernetes.io/service-name"

// TagNameSubnetInternalELB is the tag name used on a subnet to designate that
// it should be used for internal ELBs
const TagNameSubnetInternalELB = "kubernetes.io/role/internal-elb"

// TagNameSubnetPublicELB is the tag name used on a subnet to designate that
// it should be used for internet ELBs
const TagNameSubnetPublicELB = "kubernetes.io/role/elb"

// ServiceAnnotationLoadBalancerType is the annotation used on the service
// to indicate what type of Load Balancer we want. Right now, the only accepted
// value is "nlb"
const ServiceAnnotationLoadBalancerType = "service.beta.kubernetes.io/aws-load-balancer-type"

// ServiceAnnotationLoadBalancerInternal is the annotation used on the service
// to indicate that we want an internal ELB.
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

// ServiceAnnotationLoadBalancerExtraSecurityGroups is the annotation used
// on the service to specify additional security groups to be added to ELB created
const ServiceAnnotationLoadBalancerExtraSecurityGroups = "service.beta.kubernetes.io/aws-load-balancer-extra-security-groups"

// ServiceAnnotationLoadBalancerSecurityGroups is the annotation used
// on the service to specify the security groups to be added to ELB created. Differently from the annotation
// "service.beta.kubernetes.io/aws-load-balancer-extra-security-groups", this replaces all other security groups previously assigned to the ELB.
const ServiceAnnotationLoadBalancerSecurityGroups = "service.beta.kubernetes.io/aws-load-balancer-security-groups"

// ServiceAnnotationLoadBalancerCertificate is the annotation used on the
// service to request a secure listener. Value is a valid certificate ARN.
// For more, see http://docs.aws.amazon.com/ElasticLoadBalancing/latest/DeveloperGuide/elb-listener-config.html
// CertARN is an IAM or CM certificate ARN, e.g. arn:aws:acm:us-east-1:123456789012:certificate/12345678-1234-1234-1234-123456789012
const ServiceAnnotationLoadBalancerCertificate = "service.beta.kubernetes.io/aws-load-balancer-ssl-cert"

// ServiceAnnotationLoadBalancerSSLPorts is the annotation used on the service
// to specify a comma-separated list of ports that will use SSL/HTTPS
// listeners. Defaults to '*' (all).
const ServiceAnnotationLoadBalancerSSLPorts = "service.beta.kubernetes.io/aws-load-balancer-ssl-ports"

// ServiceAnnotationLoadBalancerSSLNegotiationPolicy is the annotation used on
// the service to specify a SSL negotiation settings for the HTTPS/SSL listeners
// of your load balancer. Defaults to AWS's default
const ServiceAnnotationLoadBalancerSSLNegotiationPolicy = "service.beta.kubernetes.io/aws-load-balancer-ssl-negotiation-policy"

// ServiceAnnotationLoadBalancerBEProtocol is the annotation used on the service
// to specify the protocol spoken by the backend (pod) behind a listener.
// If `http` (default) or `https`, an HTTPS listener that terminates the
//  connection and parses headers is created.
// If set to `ssl` or `tcp`, a "raw" SSL listener is used.
// If set to `http` and `aws-load-balancer-ssl-cert` is not used then
// a HTTP listener is used.
const ServiceAnnotationLoadBalancerBEProtocol = "service.beta.kubernetes.io/aws-load-balancer-backend-protocol"

// ServiceAnnotationLoadBalancerAdditionalTags is the annotation used on the service
// to specify a comma-separated list of key-value pairs which will be recorded as
// additional tags in the ELB.
// For example: "Key1=Val1,Key2=Val2,KeyNoVal1=,KeyNoVal2"
const ServiceAnnotationLoadBalancerAdditionalTags = "service.beta.kubernetes.io/aws-load-balancer-additional-resource-tags"

// ServiceAnnotationLoadBalancerHCHealthyThreshold is the annotation used on
// the service to specify the number of successive successful health checks
// required for a backend to be considered healthy for traffic.
const ServiceAnnotationLoadBalancerHCHealthyThreshold = "service.beta.kubernetes.io/aws-load-balancer-healthcheck-healthy-threshold"

// ServiceAnnotationLoadBalancerHCUnhealthyThreshold is the annotation used
// on the service to specify the number of unsuccessful health checks
// required for a backend to be considered unhealthy for traffic
const ServiceAnnotationLoadBalancerHCUnhealthyThreshold = "service.beta.kubernetes.io/aws-load-balancer-healthcheck-unhealthy-threshold"

// ServiceAnnotationLoadBalancerHCTimeout is the annotation used on the
// service to specify, in seconds, how long to wait before marking a health
// check as failed.
const ServiceAnnotationLoadBalancerHCTimeout = "service.beta.kubernetes.io/aws-load-balancer-healthcheck-timeout"

// ServiceAnnotationLoadBalancerHCInterval is the annotation used on the
// service to specify, in seconds, the interval between health checks.
const ServiceAnnotationLoadBalancerHCInterval = "service.beta.kubernetes.io/aws-load-balancer-healthcheck-interval"

// ServiceAnnotationLoadBalancerEIPAllocations is the annotation used on the
// service to specify a comma separated list of EIP allocations to use as
// static IP addresses for the NLB. Only supported on elbv2 (NLB)
const ServiceAnnotationLoadBalancerEIPAllocations = "service.beta.kubernetes.io/aws-load-balancer-eip-allocations"

// Event key when a volume is stuck on attaching state when being attached to a volume
const volumeAttachmentStuck = "VolumeAttachmentStuck"

// Indicates that a node has volumes stuck in attaching state and hence it is not fit for scheduling more pods
const nodeWithImpairedVolumes = "NodeWithImpairedVolumes"

const (
	// volumeAttachmentConsecutiveErrorLimit is the number of consecutive errors we will ignore when waiting for a volume to attach/detach
	volumeAttachmentStatusConsecutiveErrorLimit = 10

	// Attach typically takes 2-5 seconds (average is 2). Asking before 2 seconds is just waste of API quota.
	volumeAttachmentStatusInitialDelay = 2 * time.Second
	// Detach typically takes 5-10 seconds (average is 6). Asking before 5 seconds is just waste of API quota.
	volumeDetachmentStatusInitialDelay = 5 * time.Second
	// After the initial delay, poll attach/detach with exponential backoff (2046 seconds total)
	volumeAttachmentStatusPollDelay = 2 * time.Second
	volumeAttachmentStatusFactor    = 2
	volumeAttachmentStatusSteps     = 11

	// createTag* is configuration of exponential backoff for CreateTag call. We
	// retry mainly because if we create an object, we cannot tag it until it is
	// "fully created" (eventual consistency). Starting with 1 second, doubling
	// it every step and taking 9 steps results in 255 second total waiting
	// time.
	createTagInitialDelay = 1 * time.Second
	createTagFactor       = 2.0
	createTagSteps        = 9

	// volumeCreate* is configuration of exponential backoff for created volume.
	// On a random AWS account (shared among several developers) it took 4s on
	// average, 8s max.
	volumeCreateInitialDelay  = 5 * time.Second
	volumeCreateBackoffFactor = 1.2
	volumeCreateBackoffSteps  = 10

	// Number of node names that can be added to a filter. The AWS limit is 200
	// but we are using a lower limit on purpose
	filterNodeLimit = 150
)

// awsTagNameMasterRoles is a set of well-known AWS tag names that indicate the instance is a master
// The major consequence is that it is then not considered for AWS zone discovery for dynamic volume creation.
var awsTagNameMasterRoles = sets.NewString("kubernetes.io/role/master", "k8s.io/role/master")

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

// Services is an abstraction over AWS, to allow mocking/other implementations
type Services interface {
	Compute(region string) (EC2, error)
	LoadBalancing(region string) (ELB, error)
	LoadBalancingV2(region string) (ELBV2, error)
	Autoscaling(region string) (ASG, error)
	Metadata() (EC2Metadata, error)
	KeyManagement(region string) (KMS, error)
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

	ModifyVolume(*ec2.ModifyVolumeInput) (*ec2.ModifyVolumeOutput, error)

	DescribeVolumeModifications(*ec2.DescribeVolumesModificationsInput) ([]*ec2.VolumeModification, error)

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

	DescribeVpcs(input *ec2.DescribeVpcsInput) (*ec2.DescribeVpcsOutput, error)
}

// ELB is a simple pass-through of AWS' ELB client interface, which allows for testing
type ELB interface {
	CreateLoadBalancer(*elb.CreateLoadBalancerInput) (*elb.CreateLoadBalancerOutput, error)
	DeleteLoadBalancer(*elb.DeleteLoadBalancerInput) (*elb.DeleteLoadBalancerOutput, error)
	DescribeLoadBalancers(*elb.DescribeLoadBalancersInput) (*elb.DescribeLoadBalancersOutput, error)
	AddTags(*elb.AddTagsInput) (*elb.AddTagsOutput, error)
	RegisterInstancesWithLoadBalancer(*elb.RegisterInstancesWithLoadBalancerInput) (*elb.RegisterInstancesWithLoadBalancerOutput, error)
	DeregisterInstancesFromLoadBalancer(*elb.DeregisterInstancesFromLoadBalancerInput) (*elb.DeregisterInstancesFromLoadBalancerOutput, error)
	CreateLoadBalancerPolicy(*elb.CreateLoadBalancerPolicyInput) (*elb.CreateLoadBalancerPolicyOutput, error)
	SetLoadBalancerPoliciesForBackendServer(*elb.SetLoadBalancerPoliciesForBackendServerInput) (*elb.SetLoadBalancerPoliciesForBackendServerOutput, error)
	SetLoadBalancerPoliciesOfListener(input *elb.SetLoadBalancerPoliciesOfListenerInput) (*elb.SetLoadBalancerPoliciesOfListenerOutput, error)
	DescribeLoadBalancerPolicies(input *elb.DescribeLoadBalancerPoliciesInput) (*elb.DescribeLoadBalancerPoliciesOutput, error)

	DetachLoadBalancerFromSubnets(*elb.DetachLoadBalancerFromSubnetsInput) (*elb.DetachLoadBalancerFromSubnetsOutput, error)
	AttachLoadBalancerToSubnets(*elb.AttachLoadBalancerToSubnetsInput) (*elb.AttachLoadBalancerToSubnetsOutput, error)

	CreateLoadBalancerListeners(*elb.CreateLoadBalancerListenersInput) (*elb.CreateLoadBalancerListenersOutput, error)
	DeleteLoadBalancerListeners(*elb.DeleteLoadBalancerListenersInput) (*elb.DeleteLoadBalancerListenersOutput, error)

	ApplySecurityGroupsToLoadBalancer(*elb.ApplySecurityGroupsToLoadBalancerInput) (*elb.ApplySecurityGroupsToLoadBalancerOutput, error)

	ConfigureHealthCheck(*elb.ConfigureHealthCheckInput) (*elb.ConfigureHealthCheckOutput, error)

	DescribeLoadBalancerAttributes(*elb.DescribeLoadBalancerAttributesInput) (*elb.DescribeLoadBalancerAttributesOutput, error)
	ModifyLoadBalancerAttributes(*elb.ModifyLoadBalancerAttributesInput) (*elb.ModifyLoadBalancerAttributesOutput, error)
}

// ELBV2 is a simple pass-through of AWS' ELBV2 client interface, which allows for testing
type ELBV2 interface {
	AddTags(input *elbv2.AddTagsInput) (*elbv2.AddTagsOutput, error)

	CreateLoadBalancer(*elbv2.CreateLoadBalancerInput) (*elbv2.CreateLoadBalancerOutput, error)
	DescribeLoadBalancers(*elbv2.DescribeLoadBalancersInput) (*elbv2.DescribeLoadBalancersOutput, error)
	DeleteLoadBalancer(*elbv2.DeleteLoadBalancerInput) (*elbv2.DeleteLoadBalancerOutput, error)

	ModifyLoadBalancerAttributes(*elbv2.ModifyLoadBalancerAttributesInput) (*elbv2.ModifyLoadBalancerAttributesOutput, error)
	DescribeLoadBalancerAttributes(*elbv2.DescribeLoadBalancerAttributesInput) (*elbv2.DescribeLoadBalancerAttributesOutput, error)

	CreateTargetGroup(*elbv2.CreateTargetGroupInput) (*elbv2.CreateTargetGroupOutput, error)
	DescribeTargetGroups(*elbv2.DescribeTargetGroupsInput) (*elbv2.DescribeTargetGroupsOutput, error)
	ModifyTargetGroup(*elbv2.ModifyTargetGroupInput) (*elbv2.ModifyTargetGroupOutput, error)
	DeleteTargetGroup(*elbv2.DeleteTargetGroupInput) (*elbv2.DeleteTargetGroupOutput, error)

	DescribeTargetHealth(input *elbv2.DescribeTargetHealthInput) (*elbv2.DescribeTargetHealthOutput, error)

	DescribeTargetGroupAttributes(*elbv2.DescribeTargetGroupAttributesInput) (*elbv2.DescribeTargetGroupAttributesOutput, error)
	ModifyTargetGroupAttributes(*elbv2.ModifyTargetGroupAttributesInput) (*elbv2.ModifyTargetGroupAttributesOutput, error)

	RegisterTargets(*elbv2.RegisterTargetsInput) (*elbv2.RegisterTargetsOutput, error)
	DeregisterTargets(*elbv2.DeregisterTargetsInput) (*elbv2.DeregisterTargetsOutput, error)

	CreateListener(*elbv2.CreateListenerInput) (*elbv2.CreateListenerOutput, error)
	DescribeListeners(*elbv2.DescribeListenersInput) (*elbv2.DescribeListenersOutput, error)
	DeleteListener(*elbv2.DeleteListenerInput) (*elbv2.DeleteListenerOutput, error)
	ModifyListener(*elbv2.ModifyListenerInput) (*elbv2.ModifyListenerOutput, error)

	WaitUntilLoadBalancersDeleted(*elbv2.DescribeLoadBalancersInput) error
}

// ASG is a simple pass-through of the Autoscaling client interface, which
// allows for testing.
type ASG interface {
	UpdateAutoScalingGroup(*autoscaling.UpdateAutoScalingGroupInput) (*autoscaling.UpdateAutoScalingGroupOutput, error)
	DescribeAutoScalingGroups(*autoscaling.DescribeAutoScalingGroupsInput) (*autoscaling.DescribeAutoScalingGroupsOutput, error)
}

// KMS is a simple pass-through of the Key Management Service client interface,
// which allows for testing.
type KMS interface {
	DescribeKey(*kms.DescribeKeyInput) (*kms.DescribeKeyOutput, error)
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
	VolumeType       string
	AvailabilityZone string
	// IOPSPerGB x CapacityGB will give total IOPS of the volume to create.
	// Calculated total IOPS will be capped at MaxTotalIOPS.
	IOPSPerGB int
	Encrypted bool
	// fully qualified resource name to the key to use for encryption.
	// example: arn:aws:kms:us-east-1:012345678910:key/abcd1234-a123-456a-a12b-a123b4cd56ef
	KmsKeyID string
}

// Volumes is an interface for managing cloud-provisioned volumes
// TODO: Allow other clouds to implement this
type Volumes interface {
	// Attach the disk to the node with the specified NodeName
	// nodeName can be empty to mean "the instance on which we are running"
	// Returns the device (e.g. /dev/xvdf) where we attached the volume
	AttachDisk(diskName KubernetesVolumeID, nodeName types.NodeName) (string, error)
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

	// Check if disks specified in argument map are still attached to their respective nodes.
	DisksAreAttached(map[types.NodeName][]KubernetesVolumeID) (map[types.NodeName]map[KubernetesVolumeID]bool, error)

	// Expand the disk to new size
	ResizeDisk(diskName KubernetesVolumeID, oldSize resource.Quantity, newSize resource.Quantity) (resource.Quantity, error)
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

var _ cloudprovider.Interface = (*Cloud)(nil)
var _ cloudprovider.Instances = (*Cloud)(nil)
var _ cloudprovider.LoadBalancer = (*Cloud)(nil)
var _ cloudprovider.Routes = (*Cloud)(nil)
var _ cloudprovider.Zones = (*Cloud)(nil)
var _ cloudprovider.PVLabeler = (*Cloud)(nil)

// Cloud is an implementation of Interface, LoadBalancer and Instances for Amazon Web Services.
type Cloud struct {
	ec2      EC2
	elb      ELB
	elbv2    ELBV2
	asg      ASG
	kms      KMS
	metadata EC2Metadata
	cfg      *CloudConfig
	region   string
	vpcID    string

	tagging awsTagging

	// The AWS instance that we are running on
	// Note that we cache some state in awsInstance (mountpoints), so we must preserve the instance
	selfAWSInstance *awsInstance

	instanceCache instanceCache

	clientBuilder cloudprovider.ControllerClientBuilder
	kubeClient    clientset.Interface

	nodeInformer informercorev1.NodeInformer
	// Extract the function out to make it easier to test
	nodeInformerHasSynced cache.InformerSynced

	eventBroadcaster record.EventBroadcaster
	eventRecorder    record.EventRecorder

	// We keep an active list of devices we have assigned but not yet
	// attached, to avoid a race condition where we assign a device mapping
	// and then get a second request before we attach the volume
	attachingMutex sync.Mutex
	attaching      map[types.NodeName]map[mountDevice]EBSVolumeID

	// state of our device allocator for each node
	deviceAllocators map[types.NodeName]DeviceAllocator
}

var _ Volumes = &Cloud{}

// CloudConfig wraps the settings for the AWS cloud provider.
// NOTE: Cloud config files should follow the same Kubernetes deprecation policy as
// flags or CLIs. Config fields should not change behavior in incompatible ways and
// should be deprecated for at least 2 release prior to removing.
// See https://kubernetes.io/docs/reference/using-api/deprecation-policy/#deprecating-a-flag-or-cli
// for more details.
type CloudConfig struct {
	Global struct {
		// TODO: Is there any use for this?  We can get it from the instance metadata service
		// Maybe if we're not running on AWS, e.g. bootstrap; for now it is not very useful
		Zone string

		// The AWS VPC flag enables the possibility to run the master components
		// on a different aws account, on a different cloud provider or on-premises.
		// If the flag is set also the KubernetesClusterTag must be provided
		VPC string
		// SubnetID enables using a specific subnet to use for ELB's
		SubnetID string
		// RouteTableID enables using a specific RouteTable
		RouteTableID string

		// RoleARN is the IAM role to assume when interaction with AWS APIs.
		RoleARN string

		// KubernetesClusterTag is the legacy cluster id we'll use to identify our cluster resources
		KubernetesClusterTag string
		// KubernetesClusterID is the cluster id we'll use to identify our cluster resources
		KubernetesClusterID string

		//The aws provider creates an inbound rule per load balancer on the node security
		//group. However, this can run into the AWS security group rule limit of 50 if
		//many LoadBalancers are created.
		//
		//This flag disables the automatic ingress creation. It requires that the user
		//has setup a rule that allows inbound traffic on kubelet ports from the
		//local VPC subnet (so load balancers can access it). E.g. 10.82.0.0/16 30000-32000.
		DisableSecurityGroupIngress bool

		//AWS has a hard limit of 500 security groups. For large clusters creating a security group for each ELB
		//can cause the max number of security groups to be reached. If this is set instead of creating a new
		//Security group for each ELB this security group will be used instead.
		ElbSecurityGroup string

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
	// [ServiceOverride "1"]
	//  Service = s3
	//  Region = region1
	//  URL = https://s3.foo.bar
	//  SigningRegion = signing_region
	//  SigningMethod = signing_method
	//
	//  [ServiceOverride "2"]
	//     Service = ec2
	//     Region = region2
	//     URL = https://ec2.foo.bar
	//     SigningRegion = signing_region
	//     SigningMethod = signing_method
	ServiceOverride map[string]*struct {
		Service       string
		Region        string
		URL           string
		SigningRegion string
		SigningMethod string
		SigningName   string
	}
}

func (cfg *CloudConfig) validateOverrides() error {
	if len(cfg.ServiceOverride) == 0 {
		return nil
	}
	set := make(map[string]bool)
	for onum, ovrd := range cfg.ServiceOverride {
		// Note: gcfg does not space trim, so we have to when comparing to empty string ""
		name := strings.TrimSpace(ovrd.Service)
		if name == "" {
			return fmt.Errorf("service name is missing [Service is \"\"] in override %s", onum)
		}
		// insure the map service name is space trimmed
		ovrd.Service = name

		region := strings.TrimSpace(ovrd.Region)
		if region == "" {
			return fmt.Errorf("service region is missing [Region is \"\"] in override %s", onum)
		}
		// insure the map region is space trimmed
		ovrd.Region = region

		url := strings.TrimSpace(ovrd.URL)
		if url == "" {
			return fmt.Errorf("url is missing [URL is \"\"] in override %s", onum)
		}
		signingRegion := strings.TrimSpace(ovrd.SigningRegion)
		if signingRegion == "" {
			return fmt.Errorf("signingRegion is missing [SigningRegion is \"\"] in override %s", onum)
		}
		signature := name + "_" + region
		if set[signature] {
			return fmt.Errorf("duplicate entry found for service override [%s] (%s in %s)", onum, name, region)
		}
		set[signature] = true
	}
	return nil
}

func (cfg *CloudConfig) getResolver() endpoints.ResolverFunc {
	defaultResolver := endpoints.DefaultResolver()
	defaultResolverFn := func(service, region string,
		optFns ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		return defaultResolver.EndpointFor(service, region, optFns...)
	}
	if len(cfg.ServiceOverride) == 0 {
		return defaultResolverFn
	}

	return func(service, region string,
		optFns ...func(*endpoints.Options)) (endpoints.ResolvedEndpoint, error) {
		for _, override := range cfg.ServiceOverride {
			if override.Service == service && override.Region == region {
				return endpoints.ResolvedEndpoint{
					URL:           override.URL,
					SigningRegion: override.SigningRegion,
					SigningMethod: override.SigningMethod,
					SigningName:   override.SigningName,
				}, nil
			}
		}
		return defaultResolver.EndpointFor(service, region, optFns...)
	}
}

// awsSdkEC2 is an implementation of the EC2 interface, backed by aws-sdk-go
type awsSdkEC2 struct {
	ec2 *ec2.EC2
}

// Interface to make the CloudConfig immutable for awsSDKProvider
type awsCloudConfigProvider interface {
	getResolver() endpoints.ResolverFunc
}

type awsSDKProvider struct {
	creds *credentials.Credentials
	cfg   awsCloudConfigProvider

	mutex          sync.Mutex
	regionDelayers map[string]*CrossRequestRetryDelay
}

func newAWSSDKProvider(creds *credentials.Credentials, cfg *CloudConfig) *awsSDKProvider {
	return &awsSDKProvider{
		creds:          creds,
		cfg:            cfg,
		regionDelayers: make(map[string]*CrossRequestRetryDelay),
	}
}

func (p *awsSDKProvider) addHandlers(regionName string, h *request.Handlers) {
	h.Build.PushFrontNamed(request.NamedHandler{
		Name: "k8s/user-agent",
		Fn:   request.MakeAddToUserAgentHandler("kubernetes", version.Get().String()),
	})

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

	p.addAPILoggingHandlers(h)
}

func (p *awsSDKProvider) addAPILoggingHandlers(h *request.Handlers) {
	h.Send.PushBackNamed(request.NamedHandler{
		Name: "k8s/api-request",
		Fn:   awsSendHandlerLogger,
	})

	h.ValidateResponse.PushFrontNamed(request.NamedHandler{
		Name: "k8s/api-validate-response",
		Fn:   awsValidateResponseHandlerLogger,
	})
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

// SetInformers implements InformerUser interface by setting up informer-fed caches for aws lib to
// leverage Kubernetes API for caching
func (c *Cloud) SetInformers(informerFactory informers.SharedInformerFactory) {
	klog.Infof("Setting up informers for Cloud")
	c.nodeInformer = informerFactory.Core().V1().Nodes()
	c.nodeInformerHasSynced = c.nodeInformer.Informer().HasSynced
}

func (p *awsSDKProvider) Compute(regionName string) (EC2, error) {
	awsConfig := &aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}
	awsConfig = awsConfig.WithCredentialsChainVerboseErrors(true).
		WithEndpointResolver(p.cfg.getResolver())

	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	service := ec2.New(sess)

	p.addHandlers(regionName, &service.Handlers)

	ec2 := &awsSdkEC2{
		ec2: service,
	}
	return ec2, nil
}

func (p *awsSDKProvider) LoadBalancing(regionName string) (ELB, error) {
	awsConfig := &aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}
	awsConfig = awsConfig.WithCredentialsChainVerboseErrors(true).
		WithEndpointResolver(p.cfg.getResolver())

	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	elbClient := elb.New(sess)
	p.addHandlers(regionName, &elbClient.Handlers)

	return elbClient, nil
}

func (p *awsSDKProvider) LoadBalancingV2(regionName string) (ELBV2, error) {
	awsConfig := &aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}
	awsConfig = awsConfig.WithCredentialsChainVerboseErrors(true).
		WithEndpointResolver(p.cfg.getResolver())

	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	elbClient := elbv2.New(sess)

	p.addHandlers(regionName, &elbClient.Handlers)

	return elbClient, nil
}

func (p *awsSDKProvider) Autoscaling(regionName string) (ASG, error) {
	awsConfig := &aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}
	awsConfig = awsConfig.WithCredentialsChainVerboseErrors(true).
		WithEndpointResolver(p.cfg.getResolver())

	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	client := autoscaling.New(sess)

	p.addHandlers(regionName, &client.Handlers)

	return client, nil
}

func (p *awsSDKProvider) Metadata() (EC2Metadata, error) {
	sess, err := session.NewSession(&aws.Config{
		EndpointResolver: p.cfg.getResolver(),
	})
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	client := ec2metadata.New(sess)
	p.addAPILoggingHandlers(&client.Handlers)
	return client, nil
}

func (p *awsSDKProvider) KeyManagement(regionName string) (KMS, error) {
	awsConfig := &aws.Config{
		Region:      &regionName,
		Credentials: p.creds,
	}
	awsConfig = awsConfig.WithCredentialsChainVerboseErrors(true).
		WithEndpointResolver(p.cfg.getResolver())

	sess, err := session.NewSession(awsConfig)
	if err != nil {
		return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
	}
	kmsClient := kms.New(sess)

	p.addHandlers(regionName, &kmsClient.Handlers)

	return kmsClient, nil
}

func newEc2Filter(name string, values ...string) *ec2.Filter {
	filter := &ec2.Filter{
		Name: aws.String(name),
	}
	for _, value := range values {
		filter.Values = append(filter.Values, aws.String(value))
	}
	return filter
}

// AddSSHKeyToAllInstances is currently not implemented.
func (c *Cloud) AddSSHKeyToAllInstances(ctx context.Context, user string, keyData []byte) error {
	return cloudprovider.NotImplemented
}

// CurrentNodeName returns the name of the current node
func (c *Cloud) CurrentNodeName(ctx context.Context, hostname string) (types.NodeName, error) {
	return c.selfAWSInstance.nodeName, nil
}

// Implementation of EC2.Instances
func (s *awsSdkEC2) DescribeInstances(request *ec2.DescribeInstancesInput) ([]*ec2.Instance, error) {
	// Instances are paged
	results := []*ec2.Instance{}
	var nextToken *string
	requestTime := time.Now()
	for {
		response, err := s.ec2.DescribeInstances(request)
		if err != nil {
			recordAWSMetric("describe_instance", 0, err)
			return nil, fmt.Errorf("error listing AWS instances: %q", err)
		}

		for _, reservation := range response.Reservations {
			results = append(results, reservation.Instances...)
		}

		nextToken = response.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("describe_instance", timeTaken, nil)
	return results, nil
}

// Implements EC2.DescribeSecurityGroups
func (s *awsSdkEC2) DescribeSecurityGroups(request *ec2.DescribeSecurityGroupsInput) ([]*ec2.SecurityGroup, error) {
	// Security groups are paged
	results := []*ec2.SecurityGroup{}
	var nextToken *string
	requestTime := time.Now()
	for {
		response, err := s.ec2.DescribeSecurityGroups(request)
		if err != nil {
			recordAWSMetric("describe_security_groups", 0, err)
			return nil, fmt.Errorf("error listing AWS security groups: %q", err)
		}

		results = append(results, response.SecurityGroups...)

		nextToken = response.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("describe_security_groups", timeTaken, nil)
	return results, nil
}

func (s *awsSdkEC2) AttachVolume(request *ec2.AttachVolumeInput) (*ec2.VolumeAttachment, error) {
	requestTime := time.Now()
	resp, err := s.ec2.AttachVolume(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("attach_volume", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) DetachVolume(request *ec2.DetachVolumeInput) (*ec2.VolumeAttachment, error) {
	requestTime := time.Now()
	resp, err := s.ec2.DetachVolume(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("detach_volume", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) DescribeVolumes(request *ec2.DescribeVolumesInput) ([]*ec2.Volume, error) {
	// Volumes are paged
	results := []*ec2.Volume{}
	var nextToken *string
	requestTime := time.Now()
	for {
		response, err := s.ec2.DescribeVolumes(request)

		if err != nil {
			recordAWSMetric("describe_volume", 0, err)
			return nil, err
		}

		results = append(results, response.Volumes...)

		nextToken = response.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("describe_volume", timeTaken, nil)
	return results, nil
}

func (s *awsSdkEC2) CreateVolume(request *ec2.CreateVolumeInput) (*ec2.Volume, error) {
	requestTime := time.Now()
	resp, err := s.ec2.CreateVolume(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("create_volume", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) DeleteVolume(request *ec2.DeleteVolumeInput) (*ec2.DeleteVolumeOutput, error) {
	requestTime := time.Now()
	resp, err := s.ec2.DeleteVolume(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("delete_volume", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) ModifyVolume(request *ec2.ModifyVolumeInput) (*ec2.ModifyVolumeOutput, error) {
	requestTime := time.Now()
	resp, err := s.ec2.ModifyVolume(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("modify_volume", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) DescribeVolumeModifications(request *ec2.DescribeVolumesModificationsInput) ([]*ec2.VolumeModification, error) {
	requestTime := time.Now()
	results := []*ec2.VolumeModification{}
	var nextToken *string
	for {
		resp, err := s.ec2.DescribeVolumesModifications(request)
		if err != nil {
			recordAWSMetric("describe_volume_modification", 0, err)
			return nil, fmt.Errorf("error listing volume modifictions : %v", err)
		}
		results = append(results, resp.VolumesModifications...)
		nextToken = resp.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("describe_volume_modification", timeTaken, nil)
	return results, nil
}

func (s *awsSdkEC2) DescribeSubnets(request *ec2.DescribeSubnetsInput) ([]*ec2.Subnet, error) {
	// Subnets are not paged
	response, err := s.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error listing AWS subnets: %q", err)
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
	requestTime := time.Now()
	resp, err := s.ec2.CreateTags(request)
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("create_tags", timeTaken, err)
	return resp, err
}

func (s *awsSdkEC2) DescribeRouteTables(request *ec2.DescribeRouteTablesInput) ([]*ec2.RouteTable, error) {
	results := []*ec2.RouteTable{}
	var nextToken *string
	requestTime := time.Now()
	for {
		response, err := s.ec2.DescribeRouteTables(request)
		if err != nil {
			recordAWSMetric("describe_route_tables", 0, err)
			return nil, fmt.Errorf("error listing AWS route tables: %q", err)
		}

		results = append(results, response.RouteTables...)

		nextToken = response.NextToken
		if aws.StringValue(nextToken) == "" {
			break
		}
		request.NextToken = nextToken
	}
	timeTaken := time.Since(requestTime).Seconds()
	recordAWSMetric("describe_route_tables", timeTaken, nil)
	return results, nil
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

func (s *awsSdkEC2) DescribeVpcs(request *ec2.DescribeVpcsInput) (*ec2.DescribeVpcsOutput, error) {
	return s.ec2.DescribeVpcs(request)
}

func init() {
	registerMetrics()
	cloudprovider.RegisterCloudProvider(ProviderName, func(config io.Reader) (cloudprovider.Interface, error) {
		cfg, err := readAWSCloudConfig(config)
		if err != nil {
			return nil, fmt.Errorf("unable to read AWS cloud provider config file: %v", err)
		}

		if err = cfg.validateOverrides(); err != nil {
			return nil, fmt.Errorf("unable to validate custom endpoint overrides: %v", err)
		}

		sess, err := session.NewSession(&aws.Config{})
		if err != nil {
			return nil, fmt.Errorf("unable to initialize AWS session: %v", err)
		}

		var provider credentials.Provider
		if cfg.Global.RoleARN == "" {
			provider = &ec2rolecreds.EC2RoleProvider{
				Client: ec2metadata.New(sess),
			}
		} else {
			klog.Infof("Using AWS assumed role %v", cfg.Global.RoleARN)
			provider = &stscreds.AssumeRoleProvider{
				Client:  sts.New(sess),
				RoleARN: cfg.Global.RoleARN,
			}
		}

		creds := credentials.NewChainCredentials(
			[]credentials.Provider{
				&credentials.EnvProvider{},
				provider,
				&credentials.SharedCredentialsProvider{},
			})

		aws := newAWSSDKProvider(creds, cfg)
		return newAWSCloud(*cfg, aws)
	})
}

// readAWSCloudConfig reads an instance of AWSCloudConfig from config reader.
func readAWSCloudConfig(config io.Reader) (*CloudConfig, error) {
	var cfg CloudConfig
	var err error

	if config != nil {
		err = gcfg.ReadInto(&cfg, config)
		if err != nil {
			return nil, err
		}
	}

	return &cfg, nil
}

func updateConfigZone(cfg *CloudConfig, metadata EC2Metadata) error {
	if cfg.Global.Zone == "" {
		if metadata != nil {
			klog.Info("Zone not specified in configuration file; querying AWS metadata service")
			var err error
			cfg.Global.Zone, err = getAvailabilityZone(metadata)
			if err != nil {
				return err
			}
		}
		if cfg.Global.Zone == "" {
			return fmt.Errorf("no zone specified in configuration file")
		}
	}

	return nil
}

func getAvailabilityZone(metadata EC2Metadata) (string, error) {
	return metadata.GetMetadata("placement/availability-zone")
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
func newAWSCloud(cfg CloudConfig, awsServices Services) (*Cloud, error) {
	// We have some state in the Cloud object - in particular the attaching map
	// Log so that if we are building multiple Cloud objects, it is obvious!
	klog.Infof("Building AWS cloudprovider")

	metadata, err := awsServices.Metadata()
	if err != nil {
		return nil, fmt.Errorf("error creating AWS metadata client: %q", err)
	}

	err = updateConfigZone(&cfg, metadata)
	if err != nil {
		return nil, fmt.Errorf("unable to determine AWS zone from cloud provider config or EC2 instance metadata: %v", err)
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
		if !isRegionValid(regionName, metadata) {
			return nil, fmt.Errorf("not a valid AWS zone (unknown region): %s", zone)
		}
	} else {
		klog.Warningf("Strict AWS zone checking is disabled.  Proceeding with zone: %s", zone)
	}

	ec2, err := awsServices.Compute(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS EC2 client: %v", err)
	}

	elb, err := awsServices.LoadBalancing(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS ELB client: %v", err)
	}

	elbv2, err := awsServices.LoadBalancingV2(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS ELBV2 client: %v", err)
	}

	asg, err := awsServices.Autoscaling(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS autoscaling client: %v", err)
	}

	kms, err := awsServices.KeyManagement(regionName)
	if err != nil {
		return nil, fmt.Errorf("error creating AWS key management client: %v", err)
	}

	awsCloud := &Cloud{
		ec2:      ec2,
		elb:      elb,
		elbv2:    elbv2,
		asg:      asg,
		metadata: metadata,
		kms:      kms,
		cfg:      &cfg,
		region:   regionName,

		attaching:        make(map[types.NodeName]map[mountDevice]EBSVolumeID),
		deviceAllocators: make(map[types.NodeName]DeviceAllocator),
	}
	awsCloud.instanceCache.cloud = awsCloud

	tagged := cfg.Global.KubernetesClusterTag != "" || cfg.Global.KubernetesClusterID != ""
	if cfg.Global.VPC != "" && (cfg.Global.SubnetID != "" || cfg.Global.RoleARN != "") && tagged {
		// When the master is running on a different AWS account, cloud provider or on-premise
		// build up a dummy instance and use the VPC from the nodes account
		klog.Info("Master is configured to run on a different AWS account, different cloud provider or on-premises")
		awsCloud.selfAWSInstance = &awsInstance{
			nodeName: "master-dummy",
			vpcID:    cfg.Global.VPC,
			subnetID: cfg.Global.SubnetID,
		}
		awsCloud.vpcID = cfg.Global.VPC
	} else {
		selfAWSInstance, err := awsCloud.buildSelfAWSInstance()
		if err != nil {
			return nil, err
		}
		awsCloud.selfAWSInstance = selfAWSInstance
		awsCloud.vpcID = selfAWSInstance.vpcID
	}

	if cfg.Global.KubernetesClusterTag != "" || cfg.Global.KubernetesClusterID != "" {
		if err := awsCloud.tagging.init(cfg.Global.KubernetesClusterTag, cfg.Global.KubernetesClusterID); err != nil {
			return nil, err
		}
	} else {
		// TODO: Clean up double-API query
		info, err := awsCloud.selfAWSInstance.describeInstance()
		if err != nil {
			return nil, err
		}
		if err := awsCloud.tagging.initFromTags(info.Tags); err != nil {
			return nil, err
		}
	}
	return awsCloud, nil
}

// isRegionValid accepts an AWS region name and returns if the region is a
// valid region known to the AWS SDK. Considers the region returned from the
// EC2 metadata service to be a valid region as it's only available on a host
// running in a valid AWS region.
func isRegionValid(region string, metadata EC2Metadata) bool {
	// Does the AWS SDK know about the region?
	for _, p := range endpoints.DefaultPartitions() {
		for r := range p.Regions() {
			if r == region {
				return true
			}
		}
	}

	// ap-northeast-3 is purposely excluded from the SDK because it
	// requires an access request (for more details see):
	// https://github.com/aws/aws-sdk-go/issues/1863
	if region == "ap-northeast-3" {
		return true
	}

	// Fallback to checking if the region matches the instance metadata region
	// (ignoring any user overrides). This just accounts for running an old
	// build of Kubernetes in a new region that wasn't compiled into the SDK
	// when Kubernetes was built.
	if az, err := getAvailabilityZone(metadata); err == nil {
		if r, err := azToRegion(az); err == nil && region == r {
			return true
		}
	}

	return false
}

// Initialize passes a Kubernetes clientBuilder interface to the cloud provider
func (c *Cloud) Initialize(clientBuilder cloudprovider.ControllerClientBuilder, stop <-chan struct{}) {
	c.clientBuilder = clientBuilder
	c.kubeClient = clientBuilder.ClientOrDie("aws-cloud-provider")
	c.eventBroadcaster = record.NewBroadcaster()
	c.eventBroadcaster.StartLogging(klog.Infof)
	c.eventBroadcaster.StartRecordingToSink(&v1core.EventSinkImpl{Interface: c.kubeClient.CoreV1().Events("")})
	c.eventRecorder = c.eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "aws-cloud-provider"})
}

// Clusters returns the list of clusters.
func (c *Cloud) Clusters() (cloudprovider.Clusters, bool) {
	return nil, false
}

// ProviderName returns the cloud provider ID.
func (c *Cloud) ProviderName() string {
	return ProviderName
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

// HasClusterID returns true if the cluster has a clusterID
func (c *Cloud) HasClusterID() bool {
	return len(c.tagging.clusterID()) > 0
}

// NodeAddresses is an implementation of Instances.NodeAddresses.
func (c *Cloud) NodeAddresses(ctx context.Context, name types.NodeName) ([]v1.NodeAddress, error) {
	if c.selfAWSInstance.nodeName == name || len(name) == 0 {
		addresses := []v1.NodeAddress{}

		macs, err := c.metadata.GetMetadata("network/interfaces/macs/")
		if err != nil {
			return nil, fmt.Errorf("error querying AWS metadata for %q: %q", "network/interfaces/macs", err)
		}

		for _, macID := range strings.Split(macs, "\n") {
			if macID == "" {
				continue
			}
			macPath := path.Join("network/interfaces/macs/", macID, "local-ipv4s")
			internalIPs, err := c.metadata.GetMetadata(macPath)
			if err != nil {
				return nil, fmt.Errorf("error querying AWS metadata for %q: %q", macPath, err)
			}
			for _, internalIP := range strings.Split(internalIPs, "\n") {
				if internalIP == "" {
					continue
				}
				addresses = append(addresses, v1.NodeAddress{Type: v1.NodeInternalIP, Address: internalIP})
			}
		}

		externalIP, err := c.metadata.GetMetadata("public-ipv4")
		if err != nil {
			//TODO: It would be nice to be able to determine the reason for the failure,
			// but the AWS client masks all failures with the same error description.
			klog.V(4).Info("Could not determine public IP from AWS metadata.")
		} else {
			addresses = append(addresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: externalIP})
		}

		localHostname, err := c.metadata.GetMetadata("local-hostname")
		if err != nil || len(localHostname) == 0 {
			//TODO: It would be nice to be able to determine the reason for the failure,
			// but the AWS client masks all failures with the same error description.
			klog.V(4).Info("Could not determine private DNS from AWS metadata.")
		} else {
			hostname, internalDNS := parseMetadataLocalHostname(localHostname)
			addresses = append(addresses, v1.NodeAddress{Type: v1.NodeHostName, Address: hostname})
			for _, d := range internalDNS {
				addresses = append(addresses, v1.NodeAddress{Type: v1.NodeInternalDNS, Address: d})
			}
		}

		externalDNS, err := c.metadata.GetMetadata("public-hostname")
		if err != nil || len(externalDNS) == 0 {
			//TODO: It would be nice to be able to determine the reason for the failure,
			// but the AWS client masks all failures with the same error description.
			klog.V(4).Info("Could not determine public DNS from AWS metadata.")
		} else {
			addresses = append(addresses, v1.NodeAddress{Type: v1.NodeExternalDNS, Address: externalDNS})
		}

		return addresses, nil
	}

	instance, err := c.getInstanceByNodeName(name)
	if err != nil {
		return nil, fmt.Errorf("getInstanceByNodeName failed for %q with %q", name, err)
	}
	return extractNodeAddresses(instance)
}

// parseMetadataLocalHostname parses the output of "local-hostname" metadata.
// If a DHCP option set is configured for a VPC and it has multiple domain names, GetMetadata
// returns a string containing first the hostname followed by additional domain names,
// space-separated. For example, if the DHCP option set has:
// domain-name = us-west-2.compute.internal a.a b.b c.c d.d;
// $ curl http://169.254.169.254/latest/meta-data/local-hostname
// ip-192-168-111-51.us-west-2.compute.internal a.a b.b c.c d.d
func parseMetadataLocalHostname(metadata string) (string, []string) {
	localHostnames := strings.Fields(metadata)
	hostname := localHostnames[0]
	internalDNS := []string{hostname}

	privateAddress := strings.Split(hostname, ".")[0]
	for _, h := range localHostnames[1:] {
		internalDNSAddress := privateAddress + "." + h
		internalDNS = append(internalDNS, internalDNSAddress)
	}
	return hostname, internalDNS
}

// extractNodeAddresses maps the instance information from EC2 to an array of NodeAddresses
func extractNodeAddresses(instance *ec2.Instance) ([]v1.NodeAddress, error) {
	// Not clear if the order matters here, but we might as well indicate a sensible preference order

	if instance == nil {
		return nil, fmt.Errorf("nil instance passed to extractNodeAddresses")
	}

	addresses := []v1.NodeAddress{}

	// handle internal network interfaces
	for _, networkInterface := range instance.NetworkInterfaces {
		// skip network interfaces that are not currently in use
		if aws.StringValue(networkInterface.Status) != ec2.NetworkInterfaceStatusInUse {
			continue
		}

		for _, internalIP := range networkInterface.PrivateIpAddresses {
			if ipAddress := aws.StringValue(internalIP.PrivateIpAddress); ipAddress != "" {
				ip := net.ParseIP(ipAddress)
				if ip == nil {
					return nil, fmt.Errorf("EC2 instance had invalid private address: %s (%q)", aws.StringValue(instance.InstanceId), ipAddress)
				}
				addresses = append(addresses, v1.NodeAddress{Type: v1.NodeInternalIP, Address: ip.String()})
			}
		}
	}

	// TODO: Other IP addresses (multiple ips)?
	publicIPAddress := aws.StringValue(instance.PublicIpAddress)
	if publicIPAddress != "" {
		ip := net.ParseIP(publicIPAddress)
		if ip == nil {
			return nil, fmt.Errorf("EC2 instance had invalid public address: %s (%s)", aws.StringValue(instance.InstanceId), publicIPAddress)
		}
		addresses = append(addresses, v1.NodeAddress{Type: v1.NodeExternalIP, Address: ip.String()})
	}

	privateDNSName := aws.StringValue(instance.PrivateDnsName)
	if privateDNSName != "" {
		addresses = append(addresses, v1.NodeAddress{Type: v1.NodeInternalDNS, Address: privateDNSName})
		addresses = append(addresses, v1.NodeAddress{Type: v1.NodeHostName, Address: privateDNSName})
	}

	publicDNSName := aws.StringValue(instance.PublicDnsName)
	if publicDNSName != "" {
		addresses = append(addresses, v1.NodeAddress{Type: v1.NodeExternalDNS, Address: publicDNSName})
	}

	return addresses, nil
}

// NodeAddressesByProviderID returns the node addresses of an instances with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (c *Cloud) NodeAddressesByProviderID(ctx context.Context, providerID string) ([]v1.NodeAddress, error) {
	instanceID, err := KubernetesInstanceID(providerID).MapToAWSInstanceID()
	if err != nil {
		return nil, err
	}

	instance, err := describeInstance(c.ec2, instanceID)
	if err != nil {
		return nil, err
	}

	return extractNodeAddresses(instance)
}

// InstanceExistsByProviderID returns true if the instance with the given provider id still exists.
// If false is returned with no error, the instance will be immediately deleted by the cloud controller manager.
func (c *Cloud) InstanceExistsByProviderID(ctx context.Context, providerID string) (bool, error) {
	instanceID, err := KubernetesInstanceID(providerID).MapToAWSInstanceID()
	if err != nil {
		return false, err
	}

	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return false, err
	}
	if len(instances) == 0 {
		return false, nil
	}
	if len(instances) > 1 {
		return false, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}

	state := instances[0].State.Name
	if *state == ec2.InstanceStateNameTerminated {
		klog.Warningf("the instance %s is terminated", instanceID)
		return false, nil
	}

	return true, nil
}

// InstanceShutdownByProviderID returns true if the instance is in safe state to detach volumes
func (c *Cloud) InstanceShutdownByProviderID(ctx context.Context, providerID string) (bool, error) {
	instanceID, err := KubernetesInstanceID(providerID).MapToAWSInstanceID()
	if err != nil {
		return false, err
	}

	request := &ec2.DescribeInstancesInput{
		InstanceIds: []*string{instanceID.awsString()},
	}

	instances, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return false, err
	}
	if len(instances) == 0 {
		klog.Warningf("the instance %s does not exist anymore", providerID)
		// returns false, because otherwise node is not deleted from cluster
		// false means that it will continue to check InstanceExistsByProviderID
		return false, nil
	}
	if len(instances) > 1 {
		return false, fmt.Errorf("multiple instances found for instance: %s", instanceID)
	}

	instance := instances[0]
	if instance.State != nil {
		state := aws.StringValue(instance.State.Name)
		// valid state for detaching volumes
		if state == ec2.InstanceStateNameStopped {
			return true, nil
		}
	}
	return false, nil
}

// InstanceID returns the cloud provider ID of the node with the specified nodeName.
func (c *Cloud) InstanceID(ctx context.Context, nodeName types.NodeName) (string, error) {
	// In the future it is possible to also return an endpoint as:
	// <endpoint>/<zone>/<instanceid>
	if c.selfAWSInstance.nodeName == nodeName {
		return "/" + c.selfAWSInstance.availabilityZone + "/" + c.selfAWSInstance.awsID, nil
	}
	inst, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		if err == cloudprovider.InstanceNotFound {
			// The Instances interface requires that we return InstanceNotFound (without wrapping)
			return "", err
		}
		return "", fmt.Errorf("getInstanceByNodeName failed for %q with %q", nodeName, err)
	}
	return "/" + aws.StringValue(inst.Placement.AvailabilityZone) + "/" + aws.StringValue(inst.InstanceId), nil
}

// InstanceTypeByProviderID returns the cloudprovider instance type of the node with the specified unique providerID
// This method will not be called from the node that is requesting this ID. i.e. metadata service
// and other local methods cannot be used here
func (c *Cloud) InstanceTypeByProviderID(ctx context.Context, providerID string) (string, error) {
	instanceID, err := KubernetesInstanceID(providerID).MapToAWSInstanceID()
	if err != nil {
		return "", err
	}

	instance, err := describeInstance(c.ec2, instanceID)
	if err != nil {
		return "", err
	}

	return aws.StringValue(instance.InstanceType), nil
}

// InstanceType returns the type of the node with the specified nodeName.
func (c *Cloud) InstanceType(ctx context.Context, nodeName types.NodeName) (string, error) {
	if c.selfAWSInstance.nodeName == nodeName {
		return c.selfAWSInstance.instanceType, nil
	}
	inst, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		return "", fmt.Errorf("getInstanceByNodeName failed for %q with %q", nodeName, err)
	}
	return aws.StringValue(inst.InstanceType), nil
}

// GetCandidateZonesForDynamicVolume retrieves  a list of all the zones in which nodes are running
// It currently involves querying all instances
func (c *Cloud) GetCandidateZonesForDynamicVolume() (sets.String, error) {
	// We don't currently cache this; it is currently used only in volume
	// creation which is expected to be a comparatively rare occurrence.

	// TODO: Caching / expose v1.Nodes to the cloud provider?
	// TODO: We could also query for subnets, I think

	// Note: It is more efficient to call the EC2 API twice with different tag
	// filters than to call it once with a tag filter that results in a logical
	// OR. For really large clusters the logical OR will result in EC2 API rate
	// limiting.
	instances := []*ec2.Instance{}

	baseFilters := []*ec2.Filter{newEc2Filter("instance-state-name", "running")}

	filters := c.tagging.addFilters(baseFilters)
	di, err := c.describeInstances(filters)
	if err != nil {
		return nil, err
	}

	instances = append(instances, di...)

	if c.tagging.usesLegacyTags {
		filters = c.tagging.addLegacyFilters(baseFilters)
		di, err = c.describeInstances(filters)
		if err != nil {
			return nil, err
		}

		instances = append(instances, di...)
	}

	if len(instances) == 0 {
		return nil, fmt.Errorf("no instances returned")
	}

	zones := sets.NewString()

	for _, instance := range instances {
		// We skip over master nodes, if the installation tool labels them with one of the well-known master labels
		// This avoids creating a volume in a zone where only the master is running - e.g. #34583
		// This is a short-term workaround until the scheduler takes care of zone selection
		master := false
		for _, tag := range instance.Tags {
			tagKey := aws.StringValue(tag.Key)
			if awsTagNameMasterRoles.Has(tagKey) {
				master = true
			}
		}

		if master {
			klog.V(4).Infof("Ignoring master instance %q in zone discovery", aws.StringValue(instance.InstanceId))
			continue
		}

		if instance.Placement != nil {
			zone := aws.StringValue(instance.Placement.AvailabilityZone)
			zones.Insert(zone)
		}
	}

	klog.V(2).Infof("Found instances in zones %s", zones)
	return zones, nil
}

// GetZone implements Zones.GetZone
func (c *Cloud) GetZone(ctx context.Context) (cloudprovider.Zone, error) {
	return cloudprovider.Zone{
		FailureDomain: c.selfAWSInstance.availabilityZone,
		Region:        c.region,
	}, nil
}

// GetZoneByProviderID implements Zones.GetZoneByProviderID
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (c *Cloud) GetZoneByProviderID(ctx context.Context, providerID string) (cloudprovider.Zone, error) {
	instanceID, err := KubernetesInstanceID(providerID).MapToAWSInstanceID()
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	instance, err := c.getInstanceByID(string(instanceID))
	if err != nil {
		return cloudprovider.Zone{}, err
	}

	zone := cloudprovider.Zone{
		FailureDomain: *(instance.Placement.AvailabilityZone),
		Region:        c.region,
	}

	return zone, nil
}

// GetZoneByNodeName implements Zones.GetZoneByNodeName
// This is particularly useful in external cloud providers where the kubelet
// does not initialize node data.
func (c *Cloud) GetZoneByNodeName(ctx context.Context, nodeName types.NodeName) (cloudprovider.Zone, error) {
	instance, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		return cloudprovider.Zone{}, err
	}
	zone := cloudprovider.Zone{
		FailureDomain: *(instance.Placement.AvailabilityZone),
		Region:        c.region,
	}

	return zone, nil

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

// Gets the full information about this instance from the EC2 API
func (i *awsInstance) describeInstance() (*ec2.Instance, error) {
	return describeInstance(i.ec2, InstanceID(i.awsID))
}

// Gets the mountDevice already assigned to the volume, or assigns an unused mountDevice.
// If the volume is already assigned, this will return the existing mountDevice with alreadyAttached=true.
// Otherwise the mountDevice is assigned by finding the first available mountDevice, and it is returned with alreadyAttached=false.
func (c *Cloud) getMountDevice(
	i *awsInstance,
	info *ec2.Instance,
	volumeID EBSVolumeID,
	assign bool) (assigned mountDevice, alreadyAttached bool, err error) {

	deviceMappings := map[mountDevice]EBSVolumeID{}
	for _, blockDevice := range info.BlockDeviceMappings {
		name := aws.StringValue(blockDevice.DeviceName)
		if strings.HasPrefix(name, "/dev/sd") {
			name = name[7:]
		}
		if strings.HasPrefix(name, "/dev/xvd") {
			name = name[8:]
		}
		if len(name) < 1 || len(name) > 2 {
			klog.Warningf("Unexpected EBS DeviceName: %q", aws.StringValue(blockDevice.DeviceName))
		}
		deviceMappings[mountDevice(name)] = EBSVolumeID(aws.StringValue(blockDevice.Ebs.VolumeId))
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
				klog.Warningf("Got assignment call for already-assigned volume: %s@%s", mountDevice, mappingVolumeID)
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
		// we want device names with two significant characters, starting with /dev/xvdbb
		// the allowed range is /dev/xvd[b-c][a-z]
		// http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/device_naming.html
		deviceAllocator = NewDeviceAllocator()
		c.deviceAllocators[i.nodeName] = deviceAllocator
	}
	// We need to lock deviceAllocator to prevent possible race with Deprioritize function
	deviceAllocator.Lock()
	defer deviceAllocator.Unlock()

	chosen, err := deviceAllocator.GetNext(deviceMappings)
	if err != nil {
		klog.Warningf("Could not assign a mount device.  mappings=%v, error: %v", deviceMappings, err)
		return "", false, fmt.Errorf("too many EBS volumes attached to node %s", i.nodeName)
	}

	attaching := c.attaching[i.nodeName]
	if attaching == nil {
		attaching = make(map[mountDevice]EBSVolumeID)
		c.attaching[i.nodeName] = attaching
	}
	attaching[chosen] = volumeID
	klog.V(2).Infof("Assigned mount device %s -> volume %s", chosen, volumeID)

	return chosen, false, nil
}

// endAttaching removes the entry from the "attachments in progress" map
// It returns true if it was found (and removed), false otherwise
func (c *Cloud) endAttaching(i *awsInstance, volumeID EBSVolumeID, mountDevice mountDevice) bool {
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
		klog.Infof("endAttaching on device %q assigned to different volume: %q vs %q", mountDevice, volumeID, existingVolumeID)
		return false
	}
	klog.V(2).Infof("Releasing in-process attachment entry: %s -> volume %s", mountDevice, volumeID)
	delete(c.attaching[i.nodeName], mountDevice)
	return true
}

type awsDisk struct {
	ec2 EC2

	// Name in k8s
	name KubernetesVolumeID
	// id in AWS
	awsID EBSVolumeID
}

func newAWSDisk(aws *Cloud, name KubernetesVolumeID) (*awsDisk, error) {
	awsID, err := name.MapToAWSVolumeID()
	if err != nil {
		return nil, err
	}
	disk := &awsDisk{ec2: aws.ec2, name: name, awsID: awsID}
	return disk, nil
}

// Helper function for describeVolume callers. Tries to retype given error to AWS error
// and returns true in case the AWS error is "InvalidVolume.NotFound", false otherwise
func isAWSErrorVolumeNotFound(err error) bool {
	if err != nil {
		if awsError, ok := err.(awserr.Error); ok {
			// https://docs.aws.amazon.com/AWSEC2/latest/APIReference/errors-overview.html
			if awsError.Code() == "InvalidVolume.NotFound" {
				return true
			}
		}
	}
	return false
}

// Gets the full information about this volume from the EC2 API
func (d *awsDisk) describeVolume() (*ec2.Volume, error) {
	volumeID := d.awsID

	request := &ec2.DescribeVolumesInput{
		VolumeIds: []*string{volumeID.awsString()},
	}

	volumes, err := d.ec2.DescribeVolumes(request)
	if err != nil {
		return nil, err
	}
	if len(volumes) == 0 {
		return nil, fmt.Errorf("no volumes found")
	}
	if len(volumes) > 1 {
		return nil, fmt.Errorf("multiple volumes found")
	}
	return volumes[0], nil
}

func (d *awsDisk) describeVolumeModification() (*ec2.VolumeModification, error) {
	volumeID := d.awsID
	request := &ec2.DescribeVolumesModificationsInput{
		VolumeIds: []*string{volumeID.awsString()},
	}
	volumeMods, err := d.ec2.DescribeVolumeModifications(request)

	if err != nil {
		return nil, fmt.Errorf("error describing volume modification %s with %v", volumeID, err)
	}

	if len(volumeMods) == 0 {
		return nil, fmt.Errorf("no volume modifications found for %s", volumeID)
	}
	lastIndex := len(volumeMods) - 1
	return volumeMods[lastIndex], nil
}

func (d *awsDisk) modifyVolume(requestGiB int64) (int64, error) {
	volumeID := d.awsID

	request := &ec2.ModifyVolumeInput{
		VolumeId: volumeID.awsString(),
		Size:     aws.Int64(requestGiB),
	}
	output, err := d.ec2.ModifyVolume(request)
	if err != nil {
		modifyError := fmt.Errorf("AWS modifyVolume failed for %s with %v", volumeID, err)
		return requestGiB, modifyError
	}

	volumeModification := output.VolumeModification

	if aws.StringValue(volumeModification.ModificationState) == ec2.VolumeModificationStateCompleted {
		return aws.Int64Value(volumeModification.TargetSize), nil
	}

	backoff := wait.Backoff{
		Duration: 1 * time.Second,
		Factor:   2,
		Steps:    10,
	}

	checkForResize := func() (bool, error) {
		volumeModification, err := d.describeVolumeModification()

		if err != nil {
			return false, err
		}

		// According to https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/monitoring_mods.html
		// Size changes usually take a few seconds to complete and take effect after a volume is in the Optimizing state.
		if aws.StringValue(volumeModification.ModificationState) == ec2.VolumeModificationStateOptimizing {
			return true, nil
		}
		return false, nil
	}
	waitWithErr := wait.ExponentialBackoff(backoff, checkForResize)
	return requestGiB, waitWithErr
}

// applyUnSchedulableTaint applies a unschedulable taint to a node after verifying
// if node has become unusable because of volumes getting stuck in attaching state.
func (c *Cloud) applyUnSchedulableTaint(nodeName types.NodeName, reason string) {
	node, fetchErr := c.kubeClient.CoreV1().Nodes().Get(string(nodeName), metav1.GetOptions{})
	if fetchErr != nil {
		klog.Errorf("Error fetching node %s with %v", nodeName, fetchErr)
		return
	}

	taint := &v1.Taint{
		Key:    nodeWithImpairedVolumes,
		Value:  "true",
		Effect: v1.TaintEffectNoSchedule,
	}
	err := nodehelpers.AddOrUpdateTaintOnNode(c.kubeClient, string(nodeName), taint)
	if err != nil {
		klog.Errorf("Error applying taint to node %s with error %v", nodeName, err)
		return
	}
	c.eventRecorder.Eventf(node, v1.EventTypeWarning, volumeAttachmentStuck, reason)
}

// waitForAttachmentStatus polls until the attachment status is the expected value
// On success, it returns the last attachment state.
func (d *awsDisk) waitForAttachmentStatus(status string, expectedInstance, expectedDevice string) (*ec2.VolumeAttachment, error) {
	backoff := wait.Backoff{
		Duration: volumeAttachmentStatusPollDelay,
		Factor:   volumeAttachmentStatusFactor,
		Steps:    volumeAttachmentStatusSteps,
	}

	// Because of rate limiting, we often see errors from describeVolume.
	// Or AWS eventual consistency returns unexpected data.
	// So we tolerate a limited number of failures.
	// But once we see more than 10 errors in a row, we return the error.
	errorCount := 0

	// Attach/detach usually takes time. It does not make sense to start
	// polling DescribeVolumes before some initial delay to let AWS
	// process the request.
	time.Sleep(getInitialAttachDetachDelay(status))

	var attachment *ec2.VolumeAttachment

	err := wait.ExponentialBackoff(backoff, func() (bool, error) {
		info, err := d.describeVolume()
		if err != nil {
			// The VolumeNotFound error is special -- we don't need to wait for it to repeat
			if isAWSErrorVolumeNotFound(err) {
				if status == "detached" {
					// The disk doesn't exist, assume it's detached, log warning and stop waiting
					klog.Warningf("Waiting for volume %q to be detached but the volume does not exist", d.awsID)
					stateStr := "detached"
					attachment = &ec2.VolumeAttachment{
						State: &stateStr,
					}
					return true, nil
				}
				if status == "attached" {
					// The disk doesn't exist, complain, give up waiting and report error
					klog.Warningf("Waiting for volume %q to be attached but the volume does not exist", d.awsID)
					return false, err
				}
			}
			errorCount++
			if errorCount > volumeAttachmentStatusConsecutiveErrorLimit {
				// report the error
				return false, err
			}

			klog.Warningf("Ignoring error from describe volume for volume %q; will retry: %q", d.awsID, err)
			return false, nil
		}

		if len(info.Attachments) > 1 {
			// Shouldn't happen; log so we know if it is
			klog.Warningf("Found multiple attachments for volume %q: %v", d.awsID, info)
		}
		attachmentStatus := ""
		for _, a := range info.Attachments {
			if attachmentStatus != "" {
				// Shouldn't happen; log so we know if it is
				klog.Warningf("Found multiple attachments for volume %q: %v", d.awsID, info)
			}
			if a.State != nil {
				attachment = a
				attachmentStatus = *a.State
			} else {
				// Shouldn't happen; log so we know if it is
				klog.Warningf("Ignoring nil attachment state for volume %q: %v", d.awsID, a)
			}
		}
		if attachmentStatus == "" {
			attachmentStatus = "detached"
		}
		if attachment != nil {
			// AWS eventual consistency can go back in time.
			// For example, we're waiting for a volume to be attached as /dev/xvdba, but AWS can tell us it's
			// attached as /dev/xvdbb, where it was attached before and it was already detached.
			// Retry couple of times, hoping AWS starts reporting the right status.
			device := aws.StringValue(attachment.Device)
			if expectedDevice != "" && device != "" && device != expectedDevice {
				klog.Warningf("Expected device %s %s for volume %s, but found device %s %s", expectedDevice, status, d.name, device, attachmentStatus)
				errorCount++
				if errorCount > volumeAttachmentStatusConsecutiveErrorLimit {
					// report the error
					return false, fmt.Errorf("attachment of disk %q failed: requested device %q but found %q", d.name, expectedDevice, device)
				}
				return false, nil
			}
			instanceID := aws.StringValue(attachment.InstanceId)
			if expectedInstance != "" && instanceID != "" && instanceID != expectedInstance {
				klog.Warningf("Expected instance %s/%s for volume %s, but found instance %s/%s", expectedInstance, status, d.name, instanceID, attachmentStatus)
				errorCount++
				if errorCount > volumeAttachmentStatusConsecutiveErrorLimit {
					// report the error
					return false, fmt.Errorf("attachment of disk %q failed: requested device %q but found %q", d.name, expectedDevice, device)
				}
				return false, nil
			}
		}

		if attachmentStatus == status {
			// Attachment is in requested state, finish waiting
			return true, nil
		}
		// continue waiting
		errorCount = 0
		klog.V(2).Infof("Waiting for volume %q state: actual=%s, desired=%s", d.awsID, attachmentStatus, status)
		return false, nil
	})
	return attachment, err
}

// Deletes the EBS disk
func (d *awsDisk) deleteVolume() (bool, error) {
	request := &ec2.DeleteVolumeInput{VolumeId: d.awsID.awsString()}
	_, err := d.ec2.DeleteVolume(request)
	if err != nil {
		if isAWSErrorVolumeNotFound(err) {
			return false, nil
		}
		if awsError, ok := err.(awserr.Error); ok {
			if awsError.Code() == "VolumeInUse" {
				return false, volerr.NewDeletedVolumeInUseError(err.Error())
			}
		}
		return false, fmt.Errorf("error deleting EBS volume %q: %q", d.awsID, err)
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
		return nil, fmt.Errorf("error fetching instance-id from ec2 metadata service: %q", err)
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
		return nil, fmt.Errorf("error finding instance %s: %q", instanceID, err)
	}
	return newAWSInstance(c.ec2, instance), nil
}

// wrapAttachError wraps the error returned by an AttachVolume request with
// additional information, if needed and possible.
func wrapAttachError(err error, disk *awsDisk, instance string) error {
	if awsError, ok := err.(awserr.Error); ok {
		if awsError.Code() == "VolumeInUse" {
			info, err := disk.describeVolume()
			if err != nil {
				klog.Errorf("Error describing volume %q: %q", disk.awsID, err)
			} else {
				for _, a := range info.Attachments {
					if disk.awsID != EBSVolumeID(aws.StringValue(a.VolumeId)) {
						klog.Warningf("Expected to get attachment info of volume %q but instead got info of %q", disk.awsID, aws.StringValue(a.VolumeId))
					} else if aws.StringValue(a.State) == "attached" {
						return fmt.Errorf("error attaching EBS volume %q to instance %q: %q. The volume is currently attached to instance %q", disk.awsID, instance, awsError, aws.StringValue(a.InstanceId))
					}
				}
			}
		}
	}
	return fmt.Errorf("error attaching EBS volume %q to instance %q: %q", disk.awsID, instance, err)
}

// AttachDisk implements Volumes.AttachDisk
func (c *Cloud) AttachDisk(diskName KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	disk, err := newAWSDisk(c, diskName)
	if err != nil {
		return "", err
	}

	awsInstance, info, err := c.getFullInstance(nodeName)
	if err != nil {
		return "", fmt.Errorf("error finding instance %s: %q", nodeName, err)
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
				klog.Errorf("endAttaching called for disk %q when attach not in progress", disk.awsID)
			}
		}
	}()

	mountDevice, alreadyAttached, err = c.getMountDevice(awsInstance, info, disk.awsID, true)
	if err != nil {
		return "", err
	}

	// Inside the instance, the mountpoint always looks like /dev/xvdX (?)
	hostDevice := "/dev/xvd" + string(mountDevice)
	// We are using xvd names (so we are HVM only)
	// See http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/device_naming.html
	ec2Device := "/dev/xvd" + string(mountDevice)

	if !alreadyAttached {
		available, err := c.checkIfAvailable(disk, "attaching", awsInstance.awsID)
		if err != nil {
			klog.Error(err)
		}

		if !available {
			attachEnded = true
			return "", err
		}
		request := &ec2.AttachVolumeInput{
			Device:     aws.String(ec2Device),
			InstanceId: aws.String(awsInstance.awsID),
			VolumeId:   disk.awsID.awsString(),
		}

		attachResponse, err := c.ec2.AttachVolume(request)
		if err != nil {
			attachEnded = true
			// TODO: Check if the volume was concurrently attached?
			return "", wrapAttachError(err, disk, awsInstance.awsID)
		}
		if da, ok := c.deviceAllocators[awsInstance.nodeName]; ok {
			da.Deprioritize(mountDevice)
		}
		klog.V(2).Infof("AttachVolume volume=%q instance=%q request returned %v", disk.awsID, awsInstance.awsID, attachResponse)
	}

	attachment, err := disk.waitForAttachmentStatus("attached", awsInstance.awsID, ec2Device)

	if err != nil {
		if err == wait.ErrWaitTimeout {
			c.applyUnSchedulableTaint(nodeName, "Volume stuck in attaching state - node needs reboot to fix impaired state.")
		}
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
		// Already checked in waitForAttachmentStatus(), but just to be sure...
		return "", fmt.Errorf("disk attachment of %q to %q failed: requested device %q but found %q", diskName, nodeName, ec2Device, aws.StringValue(attachment.Device))
	}
	if awsInstance.awsID != aws.StringValue(attachment.InstanceId) {
		return "", fmt.Errorf("disk attachment of %q to %q failed: requested instance %q but found %q", diskName, nodeName, awsInstance.awsID, aws.StringValue(attachment.InstanceId))
	}

	return hostDevice, nil
}

// DetachDisk implements Volumes.DetachDisk
func (c *Cloud) DetachDisk(diskName KubernetesVolumeID, nodeName types.NodeName) (string, error) {
	diskInfo, attached, err := c.checkIfAttachedToNode(diskName, nodeName)
	if err != nil {
		if isAWSErrorVolumeNotFound(err) {
			// Someone deleted the volume being detached; complain, but do nothing else and return success
			klog.Warningf("DetachDisk %s called for node %s but volume does not exist; assuming the volume is detached", diskName, nodeName)
			return "", nil
		}

		return "", err
	}

	if !attached && diskInfo.ec2Instance != nil {
		klog.Warningf("DetachDisk %s called for node %s but volume is attached to node %s", diskName, nodeName, diskInfo.nodeName)
		return "", nil
	}

	if !attached {
		return "", nil
	}

	awsInstance := newAWSInstance(c.ec2, diskInfo.ec2Instance)

	mountDevice, alreadyAttached, err := c.getMountDevice(awsInstance, diskInfo.ec2Instance, diskInfo.disk.awsID, false)
	if err != nil {
		return "", err
	}

	if !alreadyAttached {
		klog.Warningf("DetachDisk called on non-attached disk: %s", diskName)
		// TODO: Continue?  Tolerate non-attached error from the AWS DetachVolume call?
	}

	request := ec2.DetachVolumeInput{
		InstanceId: &awsInstance.awsID,
		VolumeId:   diskInfo.disk.awsID.awsString(),
	}

	response, err := c.ec2.DetachVolume(&request)
	if err != nil {
		return "", fmt.Errorf("error detaching EBS volume %q from %q: %q", diskInfo.disk.awsID, awsInstance.awsID, err)
	}

	if response == nil {
		return "", errors.New("no response from DetachVolume")
	}

	attachment, err := diskInfo.disk.waitForAttachmentStatus("detached", awsInstance.awsID, "")
	if err != nil {
		return "", err
	}
	if da, ok := c.deviceAllocators[awsInstance.nodeName]; ok {
		da.Deprioritize(mountDevice)
	}
	if attachment != nil {
		// We expect it to be nil, it is (maybe) interesting if it is not
		klog.V(2).Infof("waitForAttachmentStatus returned non-nil attachment with state=detached: %v", attachment)
	}

	if mountDevice != "" {
		c.endAttaching(awsInstance, diskInfo.disk.awsID, mountDevice)
		// We don't check the return value - we don't really expect the attachment to have been
		// in progress, though it might have been
	}

	hostDevicePath := "/dev/xvd" + string(mountDevice)
	return hostDevicePath, err
}

// CreateDisk implements Volumes.CreateDisk
func (c *Cloud) CreateDisk(volumeOptions *VolumeOptions) (KubernetesVolumeID, error) {
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

	request := &ec2.CreateVolumeInput{}
	request.AvailabilityZone = aws.String(volumeOptions.AvailabilityZone)
	request.Size = aws.Int64(int64(volumeOptions.CapacityGB))
	request.VolumeType = aws.String(createType)
	request.Encrypted = aws.Bool(volumeOptions.Encrypted)
	if len(volumeOptions.KmsKeyID) > 0 {
		request.KmsKeyId = aws.String(volumeOptions.KmsKeyID)
		request.Encrypted = aws.Bool(true)
	}
	if iops > 0 {
		request.Iops = aws.Int64(iops)
	}

	tags := volumeOptions.Tags
	tags = c.tagging.buildTags(ResourceLifecycleOwned, tags)

	var tagList []*ec2.Tag
	for k, v := range tags {
		tagList = append(tagList, &ec2.Tag{
			Key: aws.String(k), Value: aws.String(v),
		})
	}
	request.TagSpecifications = append(request.TagSpecifications, &ec2.TagSpecification{
		Tags:         tagList,
		ResourceType: aws.String(ec2.ResourceTypeVolume),
	})

	response, err := c.ec2.CreateVolume(request)
	if err != nil {
		return "", err
	}

	awsID := EBSVolumeID(aws.StringValue(response.VolumeId))
	if awsID == "" {
		return "", fmt.Errorf("VolumeID was not returned by CreateVolume")
	}
	volumeName := KubernetesVolumeID("aws://" + aws.StringValue(response.AvailabilityZone) + "/" + string(awsID))

	err = c.waitUntilVolumeAvailable(volumeName)
	if err != nil {
		// AWS has a bad habbit of reporting success when creating a volume with
		// encryption keys that either don't exists or have wrong permissions.
		// Such volume lives for couple of seconds and then it's silently deleted
		// by AWS. There is no other check to ensure that given KMS key is correct,
		// because Kubernetes may have limited permissions to the key.
		if isAWSErrorVolumeNotFound(err) {
			err = fmt.Errorf("failed to create encrypted volume: the volume disappeared after creation, most likely due to inaccessible KMS encryption key")
		}
		return "", err
	}

	return volumeName, nil
}

func (c *Cloud) waitUntilVolumeAvailable(volumeName KubernetesVolumeID) error {
	disk, err := newAWSDisk(c, volumeName)
	if err != nil {
		// Unreachable code
		return err
	}
	time.Sleep(5 * time.Second)
	backoff := wait.Backoff{
		Duration: volumeCreateInitialDelay,
		Factor:   volumeCreateBackoffFactor,
		Steps:    volumeCreateBackoffSteps,
	}
	err = wait.ExponentialBackoff(backoff, func() (done bool, err error) {
		vol, err := disk.describeVolume()
		if err != nil {
			return true, err
		}
		if vol.State != nil {
			switch *vol.State {
			case "available":
				// The volume is Available, it won't be deleted now.
				return true, nil
			case "creating":
				return false, nil
			default:
				return true, fmt.Errorf("unexpected State of newly created AWS EBS volume %s: %q", volumeName, *vol.State)
			}
		}
		return false, nil
	})
	return err
}

// DeleteDisk implements Volumes.DeleteDisk
func (c *Cloud) DeleteDisk(volumeName KubernetesVolumeID) (bool, error) {
	awsDisk, err := newAWSDisk(c, volumeName)
	if err != nil {
		return false, err
	}
	available, err := c.checkIfAvailable(awsDisk, "deleting", "")
	if err != nil {
		if isAWSErrorVolumeNotFound(err) {
			klog.V(2).Infof("Volume %s not found when deleting it, assuming it's deleted", awsDisk.awsID)
			return false, nil
		}
		klog.Error(err)
	}

	if !available {
		return false, err
	}

	return awsDisk.deleteVolume()
}

func (c *Cloud) checkIfAvailable(disk *awsDisk, opName string, instance string) (bool, error) {
	info, err := disk.describeVolume()

	if err != nil {
		klog.Errorf("Error describing volume %q: %q", disk.awsID, err)
		// if for some reason we can not describe volume we will return error
		return false, err
	}

	volumeState := aws.StringValue(info.State)
	opError := fmt.Sprintf("Error %s EBS volume %q", opName, disk.awsID)
	if len(instance) != 0 {
		opError = fmt.Sprintf("%q to instance %q", opError, instance)
	}

	// Only available volumes can be attached or deleted
	if volumeState != "available" {
		// Volume is attached somewhere else and we can not attach it here
		if len(info.Attachments) > 0 {
			attachment := info.Attachments[0]
			instanceID := aws.StringValue(attachment.InstanceId)
			attachedInstance, ierr := c.getInstanceByID(instanceID)
			attachErr := fmt.Sprintf("%s since volume is currently attached to %q", opError, instanceID)
			if ierr != nil {
				klog.Error(attachErr)
				return false, errors.New(attachErr)
			}
			devicePath := aws.StringValue(attachment.Device)
			nodeName := mapInstanceToNodeName(attachedInstance)

			danglingErr := volerr.NewDanglingError(attachErr, nodeName, devicePath)
			return false, danglingErr
		}

		attachErr := fmt.Errorf("%s since volume is in %q state", opError, volumeState)
		return false, attachErr
	}

	return true, nil
}

// GetLabelsForVolume gets the volume labels for a volume
func (c *Cloud) GetLabelsForVolume(ctx context.Context, pv *v1.PersistentVolume) (map[string]string, error) {
	// Ignore if not AWSElasticBlockStore.
	if pv.Spec.AWSElasticBlockStore == nil {
		return nil, nil
	}

	// Ignore any volumes that are being provisioned
	if pv.Spec.AWSElasticBlockStore.VolumeID == cloudvolume.ProvisionedVolumeName {
		return nil, nil
	}

	spec := KubernetesVolumeID(pv.Spec.AWSElasticBlockStore.VolumeID)
	labels, err := c.GetVolumeLabels(spec)
	if err != nil {
		return nil, err
	}

	return labels, nil
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
		return nil, fmt.Errorf("volume did not have AZ information: %q", aws.StringValue(info.VolumeId))
	}

	labels[v1.LabelZoneFailureDomain] = az
	region, err := azToRegion(az)
	if err != nil {
		return nil, err
	}
	labels[v1.LabelZoneRegion] = region

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
	_, attached, err := c.checkIfAttachedToNode(diskName, nodeName)
	if err != nil {
		if isAWSErrorVolumeNotFound(err) {
			// The disk doesn't exist, can't be attached
			klog.Warningf("DiskIsAttached called for volume %s on node %s but the volume does not exist", diskName, nodeName)
			return false, nil
		}

		return true, err
	}

	return attached, nil
}

// DisksAreAttached returns a map of nodes and Kubernetes volume IDs indicating
// if the volumes are attached to the node
func (c *Cloud) DisksAreAttached(nodeDisks map[types.NodeName][]KubernetesVolumeID) (map[types.NodeName]map[KubernetesVolumeID]bool, error) {
	attached := make(map[types.NodeName]map[KubernetesVolumeID]bool)

	if len(nodeDisks) == 0 {
		return attached, nil
	}

	nodeNames := []string{}
	for nodeName, diskNames := range nodeDisks {
		for _, diskName := range diskNames {
			setNodeDisk(attached, diskName, nodeName, false)
		}
		nodeNames = append(nodeNames, mapNodeNameToPrivateDNSName(nodeName))
	}

	// Note that we get instances regardless of state.
	// This means there might be multiple nodes with the same node names.
	awsInstances, err := c.getInstancesByNodeNames(nodeNames)
	if err != nil {
		// When there is an error fetching instance information
		// it is safer to return nil and let volume information not be touched.
		return nil, err
	}

	if len(awsInstances) == 0 {
		klog.V(2).Infof("DisksAreAttached found no instances matching node names; will assume disks not attached")
		return attached, nil
	}

	// Note that we check that the volume is attached to the correct node, not that it is attached to _a_ node
	for _, awsInstance := range awsInstances {
		nodeName := mapInstanceToNodeName(awsInstance)

		diskNames := nodeDisks[nodeName]
		if len(diskNames) == 0 {
			continue
		}

		awsInstanceState := "<nil>"
		if awsInstance != nil && awsInstance.State != nil {
			awsInstanceState = aws.StringValue(awsInstance.State.Name)
		}
		if awsInstanceState == "terminated" {
			// Instance is terminated, safe to assume volumes not attached
			// Note that we keep volumes attached to instances in other states (most notably, stopped)
			continue
		}

		idToDiskName := make(map[EBSVolumeID]KubernetesVolumeID)
		for _, diskName := range diskNames {
			volumeID, err := diskName.MapToAWSVolumeID()
			if err != nil {
				return nil, fmt.Errorf("error mapping volume spec %q to aws id: %v", diskName, err)
			}
			idToDiskName[volumeID] = diskName
		}

		for _, blockDevice := range awsInstance.BlockDeviceMappings {
			volumeID := EBSVolumeID(aws.StringValue(blockDevice.Ebs.VolumeId))
			diskName, found := idToDiskName[volumeID]
			if found {
				// Disk is still attached to node
				setNodeDisk(attached, diskName, nodeName, true)
			}
		}
	}

	return attached, nil
}

// ResizeDisk resizes an EBS volume in GiB increments, it will round up to the
// next GiB if arguments are not provided in even GiB increments
func (c *Cloud) ResizeDisk(
	diskName KubernetesVolumeID,
	oldSize resource.Quantity,
	newSize resource.Quantity) (resource.Quantity, error) {
	awsDisk, err := newAWSDisk(c, diskName)
	if err != nil {
		return oldSize, err
	}

	volumeInfo, err := awsDisk.describeVolume()
	if err != nil {
		descErr := fmt.Errorf("AWS.ResizeDisk Error describing volume %s with %v", diskName, err)
		return oldSize, descErr
	}
	// AWS resizes in chunks of GiB (not GB)
	requestGiB := volumehelpers.RoundUpToGiB(newSize)
	newSizeQuant := resource.MustParse(fmt.Sprintf("%dGi", requestGiB))

	// If disk already if of greater or equal size than requested we return
	if aws.Int64Value(volumeInfo.Size) >= requestGiB {
		return newSizeQuant, nil
	}
	_, err = awsDisk.modifyVolume(requestGiB)

	if err != nil {
		return oldSize, err
	}
	return newSizeQuant, nil
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
			klog.Errorf("Found multiple load balancers with name: %s", name)
		}
		ret = loadBalancer
	}
	return ret, nil
}

func (c *Cloud) addLoadBalancerTags(loadBalancerName string, requested map[string]string) error {
	var tags []*elb.Tag
	for k, v := range requested {
		tag := &elb.Tag{
			Key:   aws.String(k),
			Value: aws.String(v),
		}
		tags = append(tags, tag)
	}

	request := &elb.AddTagsInput{}
	request.LoadBalancerNames = []*string{&loadBalancerName}
	request.Tags = tags

	_, err := c.elb.AddTags(request)
	if err != nil {
		return fmt.Errorf("error adding tags to load balancer: %v", err)
	}
	return nil
}

// Gets the current load balancer state
func (c *Cloud) describeLoadBalancerv2(name string) (*elbv2.LoadBalancer, error) {
	request := &elbv2.DescribeLoadBalancersInput{
		Names: []*string{aws.String(name)},
	}

	response, err := c.elbv2.DescribeLoadBalancers(request)
	if err != nil {
		if awsError, ok := err.(awserr.Error); ok {
			if awsError.Code() == elbv2.ErrCodeLoadBalancerNotFoundException {
				return nil, nil
			}
		}
		return nil, fmt.Errorf("error describing load balancer: %q", err)
	}

	// AWS will not return 2 load balancers with the same name _and_ type.
	for i := range response.LoadBalancers {
		if aws.StringValue(response.LoadBalancers[i].Type) == elbv2.LoadBalancerTypeEnumNetwork {
			return response.LoadBalancers[i], nil
		}
	}

	return nil, fmt.Errorf("NLB '%s' could not be found", name)
}

// Retrieves instance's vpc id from metadata
func (c *Cloud) findVPCID() (string, error) {
	macs, err := c.metadata.GetMetadata("network/interfaces/macs/")
	if err != nil {
		return "", fmt.Errorf("could not list interfaces of the instance: %q", err)
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
	return "", fmt.Errorf("could not find VPC ID in instance metadata")
}

// Retrieves the specified security group from the AWS API, or returns nil if not found
func (c *Cloud) findSecurityGroup(securityGroupID string) (*ec2.SecurityGroup, error) {
	describeSecurityGroupsRequest := &ec2.DescribeSecurityGroupsInput{
		GroupIds: []*string{&securityGroupID},
	}
	// We don't apply our tag filters because we are retrieving by ID

	groups, err := c.ec2.DescribeSecurityGroups(describeSecurityGroupsRequest)
	if err != nil {
		klog.Warningf("Error retrieving security group: %q", err)
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
	klog.V(4).Infof("Comparing %v to %v", newPermission, existing)
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
		if !found {
			return false
		}
	}

	for _, leftPair := range newPermission.UserIdGroupPairs {
		found := false
		for _, rightPair := range existing.UserIdGroupPairs {
			if isEqualUserGroupPair(leftPair, rightPair, compareGroupUserIDs) {
				found = true
				break
			}
		}
		if !found {
			return false
		}
	}

	return true
}

func isEqualUserGroupPair(l, r *ec2.UserIdGroupPair, compareGroupUserIDs bool) bool {
	klog.V(2).Infof("Comparing %v to %v", *l.GroupId, *r.GroupId)
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
		klog.Warningf("Error retrieving security group %q", err)
		return false, err
	}

	if group == nil {
		return false, fmt.Errorf("security group not found: %s", securityGroupID)
	}

	klog.V(2).Infof("Existing security group ingress: %s %v", securityGroupID, group.IpPermissions)

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
		klog.V(2).Infof("Adding security group ingress: %s %v", securityGroupID, add.List())

		request := &ec2.AuthorizeSecurityGroupIngressInput{}
		request.GroupId = &securityGroupID
		request.IpPermissions = add.List()
		_, err = c.ec2.AuthorizeSecurityGroupIngress(request)
		if err != nil {
			return false, fmt.Errorf("error authorizing security group ingress: %q", err)
		}
	}
	if remove.Len() != 0 {
		klog.V(2).Infof("Remove security group ingress: %s %v", securityGroupID, remove.List())

		request := &ec2.RevokeSecurityGroupIngressInput{}
		request.GroupId = &securityGroupID
		request.IpPermissions = remove.List()
		_, err = c.ec2.RevokeSecurityGroupIngress(request)
		if err != nil {
			return false, fmt.Errorf("error revoking security group ingress: %q", err)
		}
	}

	return true, nil
}

// Makes sure the security group includes the specified permissions
// Returns true if and only if changes were made
// The security group must already exist
func (c *Cloud) addSecurityGroupIngress(securityGroupID string, addPermissions []*ec2.IpPermission) (bool, error) {
	// We do not want to make changes to the Global defined SG
	if securityGroupID == c.cfg.Global.ElbSecurityGroup {
		return false, nil
	}

	group, err := c.findSecurityGroup(securityGroupID)
	if err != nil {
		klog.Warningf("Error retrieving security group: %q", err)
		return false, err
	}

	if group == nil {
		return false, fmt.Errorf("security group not found: %s", securityGroupID)
	}

	klog.V(2).Infof("Existing security group ingress: %s %v", securityGroupID, group.IpPermissions)

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

	klog.V(2).Infof("Adding security group ingress: %s %v", securityGroupID, changes)

	request := &ec2.AuthorizeSecurityGroupIngressInput{}
	request.GroupId = &securityGroupID
	request.IpPermissions = changes
	_, err = c.ec2.AuthorizeSecurityGroupIngress(request)
	if err != nil {
		klog.Warningf("Error authorizing security group ingress %q", err)
		return false, fmt.Errorf("error authorizing security group ingress: %q", err)
	}

	return true, nil
}

// Makes sure the security group no longer includes the specified permissions
// Returns true if and only if changes were made
// If the security group no longer exists, will return (false, nil)
func (c *Cloud) removeSecurityGroupIngress(securityGroupID string, removePermissions []*ec2.IpPermission) (bool, error) {
	// We do not want to make changes to the Global defined SG
	if securityGroupID == c.cfg.Global.ElbSecurityGroup {
		return false, nil
	}

	group, err := c.findSecurityGroup(securityGroupID)
	if err != nil {
		klog.Warningf("Error retrieving security group: %q", err)
		return false, err
	}

	if group == nil {
		klog.Warning("Security group not found: ", securityGroupID)
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

	klog.V(2).Infof("Removing security group ingress: %s %v", securityGroupID, changes)

	request := &ec2.RevokeSecurityGroupIngressInput{}
	request.GroupId = &securityGroupID
	request.IpPermissions = changes
	_, err = c.ec2.RevokeSecurityGroupIngress(request)
	if err != nil {
		klog.Warningf("Error revoking security group ingress: %q", err)
		return false, err
	}

	return true, nil
}

// Makes sure the security group exists.
// For multi-cluster isolation, name must be globally unique, for example derived from the service UUID.
// Additional tags can be specified
// Returns the security group id or error
func (c *Cloud) ensureSecurityGroup(name string, description string, additionalTags map[string]string) (string, error) {
	groupID := ""
	attempt := 0
	for {
		attempt++

		// Note that we do _not_ add our tag filters; group-name + vpc-id is the EC2 primary key.
		// However, we do check that it matches our tags.
		// If it doesn't have any tags, we tag it; this is how we recover if we failed to tag before.
		// If it has a different cluster's tags, that is an error.
		// This shouldn't happen because name is expected to be globally unique (UUID derived)
		request := &ec2.DescribeSecurityGroupsInput{}
		request.Filters = []*ec2.Filter{
			newEc2Filter("group-name", name),
			newEc2Filter("vpc-id", c.vpcID),
		}

		securityGroups, err := c.ec2.DescribeSecurityGroups(request)
		if err != nil {
			return "", err
		}

		if len(securityGroups) >= 1 {
			if len(securityGroups) > 1 {
				klog.Warningf("Found multiple security groups with name: %q", name)
			}
			err := c.tagging.readRepairClusterTags(
				c.ec2, aws.StringValue(securityGroups[0].GroupId),
				ResourceLifecycleOwned, nil, securityGroups[0].Tags)
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
					klog.V(2).Infof("Got InvalidGroup.Duplicate while creating security group (race?); will retry")
					ignore = true
				}
			}
			if !ignore {
				klog.Errorf("Error creating security group: %q", err)
				return "", err
			}
			time.Sleep(1 * time.Second)
		} else {
			groupID = aws.StringValue(createResponse.GroupId)
			break
		}
	}
	if groupID == "" {
		return "", fmt.Errorf("created security group, but id was not returned: %s", name)
	}

	err := c.tagging.createTags(c.ec2, groupID, ResourceLifecycleOwned, additionalTags)
	if err != nil {
		// If we retry, ensureClusterTags will recover from this - it
		// will add the missing tags.  We could delete the security
		// group here, but that doesn't feel like the right thing, as
		// the caller is likely to retry the create
		return "", fmt.Errorf("error tagging security group: %q", err)
	}
	return groupID, nil
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
	request.Filters = []*ec2.Filter{newEc2Filter("vpc-id", c.vpcID)}

	subnets, err := c.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error describing subnets: %q", err)
	}

	var matches []*ec2.Subnet
	for _, subnet := range subnets {
		if c.tagging.hasClusterTag(subnet.Tags) {
			matches = append(matches, subnet)
		}
	}

	if len(matches) != 0 {
		return matches, nil
	}

	// Fall back to the current instance subnets, if nothing is tagged
	klog.Warningf("No tagged subnets found; will fall-back to the current subnet only.  This is likely to be an error in a future version of k8s.")

	request = &ec2.DescribeSubnetsInput{}
	request.Filters = []*ec2.Filter{newEc2Filter("subnet-id", c.selfAWSInstance.subnetID)}

	subnets, err = c.ec2.DescribeSubnets(request)
	if err != nil {
		return nil, fmt.Errorf("error describing subnets: %q", err)
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
		return nil, fmt.Errorf("error describe route table: %q", err)
	}

	subnetsByAZ := make(map[string]*ec2.Subnet)
	for _, subnet := range subnets {
		az := aws.StringValue(subnet.AvailabilityZone)
		id := aws.StringValue(subnet.SubnetId)
		if az == "" || id == "" {
			klog.Warningf("Ignoring subnet with empty az/id: %v", subnet)
			continue
		}

		isPublic, err := isSubnetPublic(rt, id)
		if err != nil {
			return nil, err
		}
		if !internalELB && !isPublic {
			klog.V(2).Infof("Ignoring private subnet for public ELB %q", id)
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

		// If we have two subnets for the same AZ we arbitrarily choose the one that is first lexicographically.
		// TODO: Should this be an error.
		if strings.Compare(*existing.SubnetId, *subnet.SubnetId) > 0 {
			klog.Warningf("Found multiple subnets in AZ %q; choosing %q between subnets %q and %q", az, *subnet.SubnetId, *existing.SubnetId, *subnet.SubnetId)
			subnetsByAZ[az] = subnet
			continue
		}

		klog.Warningf("Found multiple subnets in AZ %q; choosing %q between subnets %q and %q", az, *existing.SubnetId, *existing.SubnetId, *subnet.SubnetId)
		continue
	}

	var azNames []string
	for key := range subnetsByAZ {
		azNames = append(azNames, key)
	}

	sort.Strings(azNames)

	var subnetIDs []string
	for _, key := range azNames {
		subnetIDs = append(subnetIDs, aws.StringValue(subnetsByAZ[key].SubnetId))
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
					klog.V(4).Infof("Assuming implicit use of main routing table %s for %s",
						aws.StringValue(table.RouteTableId), subnetID)
					subnetTable = table
					break
				}
			}
		}
	}

	if subnetTable == nil {
		return false, fmt.Errorf("could not locate routing table for subnet %s", subnetID)
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

// This function is useful in extracting the security group list from annotation
func getSGListFromAnnotation(annotatedSG string) []string {
	sgList := []string{}
	for _, extraSG := range strings.Split(annotatedSG, ",") {
		extraSG = strings.TrimSpace(extraSG)
		if len(extraSG) > 0 {
			sgList = append(sgList, extraSG)
		}
	}
	return sgList
}

// buildELBSecurityGroupList returns list of SecurityGroups which should be
// attached to ELB created by a service. List always consist of at least
// 1 member which is an SG created for this service or a SG from the Global config.
// Extra groups can be specified via annotation, as can extra tags for any
// new groups. The annotation "ServiceAnnotationLoadBalancerSecurityGroups" allows for
// setting the security groups specified.
func (c *Cloud) buildELBSecurityGroupList(serviceName types.NamespacedName, loadBalancerName string, annotations map[string]string) ([]string, bool, error) {
	var err error
	var securityGroupID string
	// We do not want to make changes to a Global defined SG
	var setupSg = false

	sgList := getSGListFromAnnotation(annotations[ServiceAnnotationLoadBalancerSecurityGroups])

	// If no Security Groups have been specified with the ServiceAnnotationLoadBalancerSecurityGroups annotation, we add the default one.
	if len(sgList) == 0 {
		if c.cfg.Global.ElbSecurityGroup != "" {
			sgList = append(sgList, c.cfg.Global.ElbSecurityGroup)
		} else {
			// Create a security group for the load balancer
			sgName := "k8s-elb-" + loadBalancerName
			sgDescription := fmt.Sprintf("Security group for Kubernetes ELB %s (%v)", loadBalancerName, serviceName)
			securityGroupID, err = c.ensureSecurityGroup(sgName, sgDescription, getLoadBalancerAdditionalTags(annotations))
			if err != nil {
				klog.Errorf("Error creating load balancer security group: %q", err)
				return nil, setupSg, err
			}
			sgList = append(sgList, securityGroupID)
			setupSg = true
		}
	}

	extraSGList := getSGListFromAnnotation(annotations[ServiceAnnotationLoadBalancerExtraSecurityGroups])
	sgList = append(sgList, extraSGList...)

	return sgList, setupSg, nil
}

// buildListener creates a new listener from the given port, adding an SSL certificate
// if indicated by the appropriate annotations.
func buildListener(port v1.ServicePort, annotations map[string]string, sslPorts *portSets) (*elb.Listener, error) {
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
func (c *Cloud) EnsureLoadBalancer(ctx context.Context, clusterName string, apiService *v1.Service, nodes []*v1.Node) (*v1.LoadBalancerStatus, error) {
	annotations := apiService.Annotations
	klog.V(2).Infof("EnsureLoadBalancer(%v, %v, %v, %v, %v, %v, %v)",
		clusterName, apiService.Namespace, apiService.Name, c.region, apiService.Spec.LoadBalancerIP, apiService.Spec.Ports, annotations)

	if apiService.Spec.SessionAffinity != v1.ServiceAffinityNone {
		// ELB supports sticky sessions, but only when configured for HTTP/HTTPS
		return nil, fmt.Errorf("unsupported load balancer affinity: %v", apiService.Spec.SessionAffinity)
	}

	if len(apiService.Spec.Ports) == 0 {
		return nil, fmt.Errorf("requested load balancer with no ports")
	}

	// Figure out what mappings we want on the load balancer
	listeners := []*elb.Listener{}
	v2Mappings := []nlbPortMapping{}

	sslPorts := getPortSets(annotations[ServiceAnnotationLoadBalancerSSLPorts])
	for _, port := range apiService.Spec.Ports {
		if port.Protocol != v1.ProtocolTCP {
			return nil, fmt.Errorf("Only TCP LoadBalancer is supported for AWS ELB")
		}
		if port.NodePort == 0 {
			klog.Errorf("Ignoring port without NodePort defined: %v", port)
			continue
		}

		if isNLB(annotations) {
			portMapping := nlbPortMapping{
				FrontendPort:     int64(port.Port),
				FrontendProtocol: string(port.Protocol),
				TrafficPort:      int64(port.NodePort),
				TrafficProtocol:  string(port.Protocol),

				// if externalTrafficPolicy == "Local", we'll override the
				// health check later
				HealthCheckPort:     int64(port.NodePort),
				HealthCheckProtocol: elbv2.ProtocolEnumTcp,
			}

			certificateARN := annotations[ServiceAnnotationLoadBalancerCertificate]
			if certificateARN != "" && (sslPorts == nil || sslPorts.numbers.Has(int64(port.Port)) || sslPorts.names.Has(port.Name)) {
				portMapping.FrontendProtocol = elbv2.ProtocolEnumTls
				portMapping.SSLCertificateARN = certificateARN
				portMapping.SSLPolicy = annotations[ServiceAnnotationLoadBalancerSSLNegotiationPolicy]

				if backendProtocol := annotations[ServiceAnnotationLoadBalancerBEProtocol]; backendProtocol == "ssl" {
					portMapping.TrafficProtocol = elbv2.ProtocolEnumTls
				}
			}

			v2Mappings = append(v2Mappings, portMapping)
		}
		listener, err := buildListener(port, annotations, sslPorts)
		if err != nil {
			return nil, err
		}
		listeners = append(listeners, listener)
	}

	if apiService.Spec.LoadBalancerIP != "" {
		return nil, fmt.Errorf("LoadBalancerIP cannot be specified for AWS ELB")
	}

	instances, err := c.findInstancesForELB(nodes)
	if err != nil {
		return nil, err
	}

	sourceRanges, err := servicehelpers.GetLoadBalancerSourceRanges(apiService)
	if err != nil {
		return nil, err
	}

	// Determine if this is tagged as an Internal ELB
	internalELB := false
	internalAnnotation := apiService.Annotations[ServiceAnnotationLoadBalancerInternal]
	if internalAnnotation == "false" {
		internalELB = false
	} else if internalAnnotation != "" {
		internalELB = true
	}

	if isNLB(annotations) {

		if path, healthCheckNodePort := servicehelpers.GetServiceHealthCheckPathPort(apiService); path != "" {
			for i := range v2Mappings {
				v2Mappings[i].HealthCheckPort = int64(healthCheckNodePort)
				v2Mappings[i].HealthCheckPath = path
				v2Mappings[i].HealthCheckProtocol = elbv2.ProtocolEnumHttp
			}
		}

		// Find the subnets that the ELB will live in
		subnetIDs, err := c.findELBSubnets(internalELB)
		if err != nil {
			klog.Errorf("Error listing subnets in VPC: %q", err)
			return nil, err
		}
		// Bail out early if there are no subnets
		if len(subnetIDs) == 0 {
			return nil, fmt.Errorf("could not find any suitable subnets for creating the ELB")
		}

		loadBalancerName := c.GetLoadBalancerName(ctx, clusterName, apiService)
		serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}

		instanceIDs := []string{}
		for id := range instances {
			instanceIDs = append(instanceIDs, string(id))
		}

		v2LoadBalancer, err := c.ensureLoadBalancerv2(
			serviceName,
			loadBalancerName,
			v2Mappings,
			instanceIDs,
			subnetIDs,
			internalELB,
			annotations,
		)
		if err != nil {
			return nil, err
		}

		sourceRangeCidrs := []string{}
		for cidr := range sourceRanges {
			sourceRangeCidrs = append(sourceRangeCidrs, cidr)
		}
		if len(sourceRangeCidrs) == 0 {
			sourceRangeCidrs = append(sourceRangeCidrs, "0.0.0.0/0")
		}

		err = c.updateInstanceSecurityGroupsForNLB(loadBalancerName, instances, sourceRangeCidrs, v2Mappings)
		if err != nil {
			klog.Warningf("Error opening ingress rules for the load balancer to the instances: %q", err)
			return nil, err
		}

		// We don't have an `ensureLoadBalancerInstances()` function for elbv2
		// because `ensureLoadBalancerv2()` requires instance Ids

		// TODO: Wait for creation?
		return v2toStatus(v2LoadBalancer), nil
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
		klog.Errorf("Error listing subnets in VPC: %q", err)
		return nil, err
	}

	// Bail out early if there are no subnets
	if len(subnetIDs) == 0 {
		return nil, fmt.Errorf("could not find any suitable subnets for creating the ELB")
	}

	loadBalancerName := c.GetLoadBalancerName(ctx, clusterName, apiService)
	serviceName := types.NamespacedName{Namespace: apiService.Namespace, Name: apiService.Name}
	securityGroupIDs, setupSg, err := c.buildELBSecurityGroupList(serviceName, loadBalancerName, annotations)
	if err != nil {
		return nil, err
	}
	if len(securityGroupIDs) == 0 {
		return nil, fmt.Errorf("[BUG] ELB can't have empty list of Security Groups to be assigned, this is a Kubernetes bug, please report")
	}

	if setupSg {
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
				IpRanges:   ec2SourceRanges,
			}

			permissions.Insert(permission)
		}
		_, err = c.setSecurityGroupIngress(securityGroupIDs[0], permissions)
		if err != nil {
			return nil, err
		}
	}

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
		annotations,
	)
	if err != nil {
		return nil, err
	}

	if sslPolicyName, ok := annotations[ServiceAnnotationLoadBalancerSSLNegotiationPolicy]; ok {
		err := c.ensureSSLNegotiationPolicy(loadBalancer, sslPolicyName)
		if err != nil {
			return nil, err
		}

		for _, port := range c.getLoadBalancerTLSPorts(loadBalancer) {
			err := c.setSSLNegotiationPolicy(loadBalancerName, sslPolicyName, port)
			if err != nil {
				return nil, err
			}
		}
	}

	if path, healthCheckNodePort := servicehelpers.GetServiceHealthCheckPathPort(apiService); path != "" {
		klog.V(4).Infof("service %v (%v) needs health checks on :%d%s)", apiService.Name, loadBalancerName, healthCheckNodePort, path)
		err = c.ensureLoadBalancerHealthCheck(loadBalancer, "HTTP", healthCheckNodePort, path, annotations)
		if err != nil {
			return nil, fmt.Errorf("Failed to ensure health check for localized service %v on node port %v: %q", loadBalancerName, healthCheckNodePort, err)
		}
	} else {
		klog.V(4).Infof("service %v does not need custom health checks", apiService.Name)
		// We only configure a TCP health-check on the first port
		var tcpHealthCheckPort int32
		for _, listener := range listeners {
			if listener.InstancePort == nil {
				continue
			}
			tcpHealthCheckPort = int32(*listener.InstancePort)
			break
		}
		annotationProtocol := strings.ToLower(annotations[ServiceAnnotationLoadBalancerBEProtocol])
		var hcProtocol string
		if annotationProtocol == "https" || annotationProtocol == "ssl" {
			hcProtocol = "SSL"
		} else {
			hcProtocol = "TCP"
		}
		// there must be no path on TCP health check
		err = c.ensureLoadBalancerHealthCheck(loadBalancer, hcProtocol, tcpHealthCheckPort, "", annotations)
		if err != nil {
			return nil, err
		}
	}

	err = c.updateInstanceSecurityGroupsForLoadBalancer(loadBalancer, instances)
	if err != nil {
		klog.Warningf("Error opening ingress rules for the load balancer to the instances: %q", err)
		return nil, err
	}

	err = c.ensureLoadBalancerInstances(aws.StringValue(loadBalancer.LoadBalancerName), loadBalancer.Instances, instances)
	if err != nil {
		klog.Warningf("Error registering instances with the load balancer: %q", err)
		return nil, err
	}

	klog.V(1).Infof("Loadbalancer %s (%v) has DNS name %s", loadBalancerName, serviceName, aws.StringValue(loadBalancer.DNSName))

	// TODO: Wait for creation?

	status := toStatus(loadBalancer)
	return status, nil
}

// GetLoadBalancer is an implementation of LoadBalancer.GetLoadBalancer
func (c *Cloud) GetLoadBalancer(ctx context.Context, clusterName string, service *v1.Service) (*v1.LoadBalancerStatus, bool, error) {
	loadBalancerName := c.GetLoadBalancerName(ctx, clusterName, service)

	if isNLB(service.Annotations) {
		lb, err := c.describeLoadBalancerv2(loadBalancerName)
		if err != nil {
			return nil, false, err
		}
		if lb == nil {
			return nil, false, nil
		}
		return v2toStatus(lb), true, nil
	}

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

// GetLoadBalancerName is an implementation of LoadBalancer.GetLoadBalancerName
func (c *Cloud) GetLoadBalancerName(ctx context.Context, clusterName string, service *v1.Service) string {
	// TODO: replace DefaultLoadBalancerName to generate more meaningful loadbalancer names.
	return cloudprovider.DefaultLoadBalancerName(service)
}

func toStatus(lb *elb.LoadBalancerDescription) *v1.LoadBalancerStatus {
	status := &v1.LoadBalancerStatus{}

	if aws.StringValue(lb.DNSName) != "" {
		var ingress v1.LoadBalancerIngress
		ingress.Hostname = aws.StringValue(lb.DNSName)
		status.Ingress = []v1.LoadBalancerIngress{ingress}
	}

	return status
}

func v2toStatus(lb *elbv2.LoadBalancer) *v1.LoadBalancerStatus {
	status := &v1.LoadBalancerStatus{}
	if lb == nil {
		klog.Error("[BUG] v2toStatus got nil input, this is a Kubernetes bug, please report")
		return status
	}

	// We check for Active or Provisioning, the only successful statuses
	if aws.StringValue(lb.DNSName) != "" && (aws.StringValue(lb.State.Code) == elbv2.LoadBalancerStateEnumActive ||
		aws.StringValue(lb.State.Code) == elbv2.LoadBalancerStateEnumProvisioning) {
		var ingress v1.LoadBalancerIngress
		ingress.Hostname = aws.StringValue(lb.DNSName)
		status.Ingress = []v1.LoadBalancerIngress{ingress}
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
			klog.Warningf("Ignoring security group without id for instance %q: %v", instanceID, group)
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
			taggedGroups := ""
			for _, v := range tagged {
				taggedGroups += fmt.Sprintf("%s(%s) ", *v.GroupId, *v.GroupName)
			}
			return nil, fmt.Errorf("Multiple tagged security groups found for instance %s; ensure only the k8s security group is tagged; the tagged groups were %v", instanceID, taggedGroups)
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

	klog.Warningf("No security group found for instance %q", instanceID)
	return nil, nil
}

// Return all the security groups that are tagged as being part of our cluster
func (c *Cloud) getTaggedSecurityGroups() (map[string]*ec2.SecurityGroup, error) {
	request := &ec2.DescribeSecurityGroupsInput{}
	groups, err := c.ec2.DescribeSecurityGroups(request)
	if err != nil {
		return nil, fmt.Errorf("error querying security groups: %q", err)
	}

	m := make(map[string]*ec2.SecurityGroup)
	for _, group := range groups {
		if !c.tagging.hasClusterTag(group.Tags) {
			continue
		}

		id := aws.StringValue(group.GroupId)
		if id == "" {
			klog.Warningf("Ignoring group without id: %v", group)
			continue
		}
		m[id] = group
	}
	return m, nil
}

// Open security group ingress rules on the instances so that the load balancer can talk to them
// Will also remove any security groups ingress rules for the load balancer that are _not_ needed for allInstances
func (c *Cloud) updateInstanceSecurityGroupsForLoadBalancer(lb *elb.LoadBalancerDescription, instances map[InstanceID]*ec2.Instance) error {
	if c.cfg.Global.DisableSecurityGroupIngress {
		return nil
	}

	// Determine the load balancer security group id
	loadBalancerSecurityGroupID := ""
	for _, securityGroup := range lb.SecurityGroups {
		if aws.StringValue(securityGroup) == "" {
			continue
		}
		if loadBalancerSecurityGroupID != "" {
			// We create LBs with one SG
			klog.Warningf("Multiple security groups for load balancer: %q", aws.StringValue(lb.LoadBalancerName))
		}
		loadBalancerSecurityGroupID = *securityGroup
	}
	if loadBalancerSecurityGroupID == "" {
		return fmt.Errorf("could not determine security group for load balancer: %s", aws.StringValue(lb.LoadBalancerName))
	}

	// Get the actual list of groups that allow ingress from the load-balancer
	var actualGroups []*ec2.SecurityGroup
	{
		describeRequest := &ec2.DescribeSecurityGroupsInput{}
		describeRequest.Filters = []*ec2.Filter{
			newEc2Filter("ip-permission.group-id", loadBalancerSecurityGroupID),
		}
		response, err := c.ec2.DescribeSecurityGroups(describeRequest)
		if err != nil {
			return fmt.Errorf("error querying security groups for ELB: %q", err)
		}
		for _, sg := range response {
			if !c.tagging.hasClusterTag(sg.Tags) {
				continue
			}
			actualGroups = append(actualGroups, sg)
		}
	}

	taggedSecurityGroups, err := c.getTaggedSecurityGroups()
	if err != nil {
		return fmt.Errorf("error querying for tagged security groups: %q", err)
	}

	// Open the firewall from the load balancer to the instance
	// We don't actually have a trivial way to know in advance which security group the instance is in
	// (it is probably the node security group, but we don't easily have that).
	// However, we _do_ have the list of security groups on the instance records.

	// Map containing the changes we want to make; true to add, false to remove
	instanceSecurityGroupIds := map[string]bool{}

	// Scan instances for groups we want open
	for _, instance := range instances {
		securityGroup, err := findSecurityGroupForInstance(instance, taggedSecurityGroups)
		if err != nil {
			return err
		}

		if securityGroup == nil {
			klog.Warning("Ignoring instance without security group: ", aws.StringValue(instance.InstanceId))
			continue
		}
		id := aws.StringValue(securityGroup.GroupId)
		if id == "" {
			klog.Warningf("found security group without id: %v", securityGroup)
			continue
		}

		instanceSecurityGroupIds[id] = true
	}

	// Compare to actual groups
	for _, actualGroup := range actualGroups {
		actualGroupID := aws.StringValue(actualGroup.GroupId)
		if actualGroupID == "" {
			klog.Warning("Ignoring group without ID: ", actualGroup)
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
			klog.V(2).Infof("Adding rule for traffic from the load balancer (%s) to instances (%s)", loadBalancerSecurityGroupID, instanceSecurityGroupID)
		} else {
			klog.V(2).Infof("Removing rule for traffic from the load balancer (%s) to instance (%s)", loadBalancerSecurityGroupID, instanceSecurityGroupID)
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
				klog.Warning("Allowing ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		} else {
			changed, err := c.removeSecurityGroupIngress(instanceSecurityGroupID, permissions)
			if err != nil {
				return err
			}
			if !changed {
				klog.Warning("Revoking ingress was not needed; concurrent change? groupId=", instanceSecurityGroupID)
			}
		}
	}

	return nil
}

// EnsureLoadBalancerDeleted implements LoadBalancer.EnsureLoadBalancerDeleted.
func (c *Cloud) EnsureLoadBalancerDeleted(ctx context.Context, clusterName string, service *v1.Service) error {
	loadBalancerName := c.GetLoadBalancerName(ctx, clusterName, service)

	if isNLB(service.Annotations) {
		lb, err := c.describeLoadBalancerv2(loadBalancerName)
		if err != nil {
			return err
		}
		if lb == nil {
			klog.Info("Load balancer already deleted: ", loadBalancerName)
			return nil
		}

		// Delete the LoadBalancer and target groups
		//
		// Deleting a target group while associated with a load balancer will
		// fail. We delete the loadbalancer first. This does leave the
		// possibility of zombie target groups if DeleteLoadBalancer() fails
		//
		// * Get target groups for NLB
		// * Delete Load Balancer
		// * Delete target groups
		// * Clean up SecurityGroupRules
		{

			targetGroups, err := c.elbv2.DescribeTargetGroups(
				&elbv2.DescribeTargetGroupsInput{LoadBalancerArn: lb.LoadBalancerArn},
			)
			if err != nil {
				return fmt.Errorf("error listing target groups before deleting load balancer: %q", err)
			}

			_, err = c.elbv2.DeleteLoadBalancer(
				&elbv2.DeleteLoadBalancerInput{LoadBalancerArn: lb.LoadBalancerArn},
			)
			if err != nil {
				return fmt.Errorf("error deleting load balancer %q: %v", loadBalancerName, err)
			}

			for _, group := range targetGroups.TargetGroups {
				_, err := c.elbv2.DeleteTargetGroup(
					&elbv2.DeleteTargetGroupInput{TargetGroupArn: group.TargetGroupArn},
				)
				if err != nil {
					return fmt.Errorf("error deleting target groups after deleting load balancer: %q", err)
				}
			}
		}

		return c.updateInstanceSecurityGroupsForNLB(loadBalancerName, nil, nil, nil)
	}

	lb, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return err
	}

	if lb == nil {
		klog.Info("Load balancer already deleted: ", loadBalancerName)
		return nil
	}

	{
		// De-authorize the load balancer security group from the instances security group
		err = c.updateInstanceSecurityGroupsForLoadBalancer(lb, nil)
		if err != nil {
			klog.Errorf("Error deregistering load balancer from instance security groups: %q", err)
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
			klog.Errorf("Error deleting load balancer: %q", err)
			return err
		}
	}

	{
		// Delete the security group(s) for the load balancer
		// Note that this is annoying: the load balancer disappears from the API immediately, but it is still
		// deleting in the background.  We get a DependencyViolation until the load balancer has deleted itself

		var loadBalancerSGs = aws.StringValueSlice(lb.SecurityGroups)

		describeRequest := &ec2.DescribeSecurityGroupsInput{}
		describeRequest.Filters = []*ec2.Filter{
			newEc2Filter("group-id", loadBalancerSGs...),
		}
		response, err := c.ec2.DescribeSecurityGroups(describeRequest)
		if err != nil {
			return fmt.Errorf("error querying security groups for ELB: %q", err)
		}

		// Collect the security groups to delete
		securityGroupIDs := map[string]struct{}{}
		annotatedSgSet := map[string]bool{}
		annotatedSgsList := getSGListFromAnnotation(service.Annotations[ServiceAnnotationLoadBalancerSecurityGroups])
		annotatedExtraSgsList := getSGListFromAnnotation(service.Annotations[ServiceAnnotationLoadBalancerExtraSecurityGroups])
		annotatedSgsList = append(annotatedSgsList, annotatedExtraSgsList...)

		for _, sg := range annotatedSgsList {
			annotatedSgSet[sg] = true
		}

		for _, sg := range response {
			sgID := aws.StringValue(sg.GroupId)

			if sgID == c.cfg.Global.ElbSecurityGroup {
				//We don't want to delete a security group that was defined in the Cloud Configuration.
				continue
			}
			if sgID == "" {
				klog.Warningf("Ignoring empty security group in %s", service.Name)
				continue
			}

			if !c.tagging.hasClusterTag(sg.Tags) {
				klog.Warningf("Ignoring security group with no cluster tag in %s", service.Name)
				continue
			}

			// This is an extra protection of deletion of non provisioned Security Group which is annotated with `service.beta.kubernetes.io/aws-load-balancer-security-groups`.
			if _, ok := annotatedSgSet[sgID]; ok {
				klog.Warningf("Ignoring security group with annotation `service.beta.kubernetes.io/aws-load-balancer-security-groups` or service.beta.kubernetes.io/aws-load-balancer-extra-security-groups in %s", service.Name)
				continue
			}

			securityGroupIDs[sgID] = struct{}{}
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
							klog.V(2).Infof("Ignoring DependencyViolation while deleting load-balancer security group (%s), assuming because LB is in process of deleting", securityGroupID)
							ignore = true
						}
					}
					if !ignore {
						return fmt.Errorf("error while deleting load balancer security group (%s): %q", securityGroupID, err)
					}
				}
			}

			if len(securityGroupIDs) == 0 {
				klog.V(2).Info("Deleted all security groups for load balancer: ", service.Name)
				break
			}

			if time.Now().After(timeoutAt) {
				ids := []string{}
				for id := range securityGroupIDs {
					ids = append(ids, id)
				}

				return fmt.Errorf("timed out deleting ELB: %s. Could not delete security groups %v", service.Name, strings.Join(ids, ","))
			}

			klog.V(2).Info("Waiting for load-balancer to delete so we can delete security groups: ", service.Name)

			time.Sleep(10 * time.Second)
		}
	}

	return nil
}

// UpdateLoadBalancer implements LoadBalancer.UpdateLoadBalancer
func (c *Cloud) UpdateLoadBalancer(ctx context.Context, clusterName string, service *v1.Service, nodes []*v1.Node) error {
	instances, err := c.findInstancesForELB(nodes)
	if err != nil {
		return err
	}

	loadBalancerName := c.GetLoadBalancerName(ctx, clusterName, service)
	if isNLB(service.Annotations) {
		lb, err := c.describeLoadBalancerv2(loadBalancerName)
		if err != nil {
			return err
		}
		if lb == nil {
			return fmt.Errorf("Load balancer not found")
		}
		_, err = c.EnsureLoadBalancer(ctx, clusterName, service, nodes)
		return err
	}
	lb, err := c.describeLoadBalancer(loadBalancerName)
	if err != nil {
		return err
	}

	if lb == nil {
		return fmt.Errorf("Load balancer not found")
	}

	if sslPolicyName, ok := service.Annotations[ServiceAnnotationLoadBalancerSSLNegotiationPolicy]; ok {
		err := c.ensureSSLNegotiationPolicy(lb, sslPolicyName)
		if err != nil {
			return err
		}
		for _, port := range c.getLoadBalancerTLSPorts(lb) {
			err := c.setSSLNegotiationPolicy(loadBalancerName, sslPolicyName, port)
			if err != nil {
				return err
			}
		}
	}

	err = c.ensureLoadBalancerInstances(aws.StringValue(lb.LoadBalancerName), lb.Instances, instances)
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
		return nil, cloudprovider.InstanceNotFound
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
		instanceID := aws.StringValue(instance.InstanceId)
		if instanceID == "" {
			continue
		}

		instancesByID[instanceID] = instance
	}

	return instancesByID, nil
}

func (c *Cloud) getInstancesByNodeNames(nodeNames []string, states ...string) ([]*ec2.Instance, error) {
	names := aws.StringSlice(nodeNames)
	ec2Instances := []*ec2.Instance{}

	for i := 0; i < len(names); i += filterNodeLimit {
		end := i + filterNodeLimit
		if end > len(names) {
			end = len(names)
		}

		nameSlice := names[i:end]

		nodeNameFilter := &ec2.Filter{
			Name:   aws.String("private-dns-name"),
			Values: nameSlice,
		}

		filters := []*ec2.Filter{nodeNameFilter}
		if len(states) > 0 {
			filters = append(filters, newEc2Filter("instance-state-name", states...))
		}

		instances, err := c.describeInstances(filters)
		if err != nil {
			klog.V(2).Infof("Failed to describe instances %v", nodeNames)
			return nil, err
		}
		ec2Instances = append(ec2Instances, instances...)
	}

	if len(ec2Instances) == 0 {
		klog.V(3).Infof("Failed to find any instances %v", nodeNames)
		return nil, nil
	}
	return ec2Instances, nil
}

// TODO: Move to instanceCache
func (c *Cloud) describeInstances(filters []*ec2.Filter) ([]*ec2.Instance, error) {
	request := &ec2.DescribeInstancesInput{
		Filters: filters,
	}

	response, err := c.ec2.DescribeInstances(request)
	if err != nil {
		return nil, err
	}

	var matches []*ec2.Instance
	for _, instance := range response {
		if c.tagging.hasClusterTag(instance.Tags) {
			matches = append(matches, instance)
		}
	}
	return matches, nil
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

var aliveFilter = []string{
	ec2.InstanceStateNamePending,
	ec2.InstanceStateNameRunning,
	ec2.InstanceStateNameShuttingDown,
	ec2.InstanceStateNameStopping,
	ec2.InstanceStateNameStopped,
}

// Returns the instance with the specified node name
// Returns nil if it does not exist
func (c *Cloud) findInstanceByNodeName(nodeName types.NodeName) (*ec2.Instance, error) {
	privateDNSName := mapNodeNameToPrivateDNSName(nodeName)
	filters := []*ec2.Filter{
		newEc2Filter("private-dns-name", privateDNSName),
		// exclude instances in "terminated" state
		newEc2Filter("instance-state-name", aliveFilter...),
	}

	instances, err := c.describeInstances(filters)
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
	var instance *ec2.Instance

	// we leverage node cache to try to retrieve node's provider id first, as
	// get instance by provider id is way more efficient than by filters in
	// aws context
	awsID, err := c.nodeNameToProviderID(nodeName)
	if err != nil {
		klog.V(3).Infof("Unable to convert node name %q to aws instanceID, fall back to findInstanceByNodeName: %v", nodeName, err)
		instance, err = c.findInstanceByNodeName(nodeName)
	} else {
		instance, err = c.getInstanceByID(string(awsID))
	}
	if err == nil && instance == nil {
		return nil, cloudprovider.InstanceNotFound
	}
	return instance, err
}

func (c *Cloud) getFullInstance(nodeName types.NodeName) (*awsInstance, *ec2.Instance, error) {
	if nodeName == "" {
		instance, err := c.getInstanceByID(c.selfAWSInstance.awsID)
		return c.selfAWSInstance, instance, err
	}
	instance, err := c.getInstanceByNodeName(nodeName)
	if err != nil {
		return nil, nil, err
	}
	awsInstance := newAWSInstance(c.ec2, instance)
	return awsInstance, instance, err
}

func (c *Cloud) nodeNameToProviderID(nodeName types.NodeName) (InstanceID, error) {
	if len(nodeName) == 0 {
		return "", fmt.Errorf("no nodeName provided")
	}

	if c.nodeInformerHasSynced == nil || !c.nodeInformerHasSynced() {
		return "", fmt.Errorf("node informer has not synced yet")
	}

	node, err := c.nodeInformer.Lister().Get(string(nodeName))
	if err != nil {
		return "", err
	}
	if len(node.Spec.ProviderID) == 0 {
		return "", fmt.Errorf("node has no providerID")
	}

	return KubernetesInstanceID(node.Spec.ProviderID).MapToAWSInstanceID()
}

func setNodeDisk(
	nodeDiskMap map[types.NodeName]map[KubernetesVolumeID]bool,
	volumeID KubernetesVolumeID,
	nodeName types.NodeName,
	check bool) {

	volumeMap := nodeDiskMap[nodeName]

	if volumeMap == nil {
		volumeMap = make(map[KubernetesVolumeID]bool)
		nodeDiskMap[nodeName] = volumeMap
	}
	volumeMap[volumeID] = check
}

func getInitialAttachDetachDelay(status string) time.Duration {
	if status == "detached" {
		return volumeDetachmentStatusInitialDelay
	}
	return volumeAttachmentStatusInitialDelay
}
