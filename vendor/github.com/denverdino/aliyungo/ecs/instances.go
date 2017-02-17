package ecs

import (
	"encoding/base64"
	"encoding/json"
	"strconv"
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

// InstanceStatus represents instance status
type InstanceStatus string

// Constants of InstanceStatus
const (
	Creating = InstanceStatus("Creating")
	Running  = InstanceStatus("Running")
	Starting = InstanceStatus("Starting")

	Stopped  = InstanceStatus("Stopped")
	Stopping = InstanceStatus("Stopping")
)

type LockReason string

const (
	LockReasonFinancial = LockReason("financial")
	LockReasonSecurity  = LockReason("security")
)

type LockReasonType struct {
	LockReason LockReason
}

type DescribeInstanceStatusArgs struct {
	RegionId common.Region
	ZoneId   string
	common.Pagination
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&instancestatusitemtype
type InstanceStatusItemType struct {
	InstanceId string
	Status     InstanceStatus
}

type DescribeInstanceStatusResponse struct {
	common.Response
	common.PaginationResult
	InstanceStatuses struct {
		InstanceStatus []InstanceStatusItemType
	}
}

// DescribeInstanceStatus describes instance status
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&describeinstancestatus
func (client *Client) DescribeInstanceStatus(args *DescribeInstanceStatusArgs) (instanceStatuses []InstanceStatusItemType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeInstanceStatusResponse{}

	err = client.Invoke("DescribeInstanceStatus", args, &response)

	if err == nil {
		return response.InstanceStatuses.InstanceStatus, &response.PaginationResult, nil
	}

	return nil, nil, err
}

type StopInstanceArgs struct {
	InstanceId string
	ForceStop  bool
}

type StopInstanceResponse struct {
	common.Response
}

// StopInstance stops instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&stopinstance
func (client *Client) StopInstance(instanceId string, forceStop bool) error {
	args := StopInstanceArgs{
		InstanceId: instanceId,
		ForceStop:  forceStop,
	}
	response := StopInstanceResponse{}
	err := client.Invoke("StopInstance", &args, &response)
	return err
}

type StartInstanceArgs struct {
	InstanceId string
}

type StartInstanceResponse struct {
	common.Response
}

// StartInstance starts instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&startinstance
func (client *Client) StartInstance(instanceId string) error {
	args := StartInstanceArgs{InstanceId: instanceId}
	response := StartInstanceResponse{}
	err := client.Invoke("StartInstance", &args, &response)
	return err
}

type RebootInstanceArgs struct {
	InstanceId string
	ForceStop  bool
}

type RebootInstanceResponse struct {
	common.Response
}

// RebootInstance reboot instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&rebootinstance
func (client *Client) RebootInstance(instanceId string, forceStop bool) error {
	request := RebootInstanceArgs{
		InstanceId: instanceId,
		ForceStop:  forceStop,
	}
	response := RebootInstanceResponse{}
	err := client.Invoke("RebootInstance", &request, &response)
	return err
}

type DescribeInstanceAttributeArgs struct {
	InstanceId string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&operationlockstype
type OperationLocksType struct {
	LockReason []LockReasonType //enum for financial, security
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&securitygroupidsettype
type SecurityGroupIdSetType struct {
	SecurityGroupId string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&ipaddresssettype
type IpAddressSetType struct {
	IpAddress []string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&vpcattributestype
type VpcAttributesType struct {
	VpcId            string
	VSwitchId        string
	PrivateIpAddress IpAddressSetType
	NatIpAddress     string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&eipaddressassociatetype
type EipAddressAssociateType struct {
	AllocationId       string
	IpAddress          string
	Bandwidth          int
	InternetChargeType common.InternetChargeType
}

// Experimental feature
type SpotStrategyType string

// Constants of SpotStrategyType
const (
	NoSpot             = SpotStrategyType("NoSpot")
	SpotWithPriceLimit = SpotStrategyType("SpotWithPriceLimit")
)

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&instanceattributestype
type InstanceAttributesType struct {
	InstanceId         string
	InstanceName       string
	Description        string
	ImageId            string
	RegionId           common.Region
	ZoneId             string
	CPU                int
	Memory             int
	ClusterId          string
	InstanceType       string
	InstanceTypeFamily string
	HostName           string
	SerialNumber       string
	Status             InstanceStatus
	OperationLocks     OperationLocksType
	SecurityGroupIds   struct {
		SecurityGroupId []string
	}
	PublicIpAddress         IpAddressSetType
	InnerIpAddress          IpAddressSetType
	InstanceNetworkType     string //enum Classic | Vpc
	InternetMaxBandwidthIn  int
	InternetMaxBandwidthOut int
	InternetChargeType      common.InternetChargeType
	CreationTime            util.ISO6801Time //time.Time
	VpcAttributes           VpcAttributesType
	EipAddress              EipAddressAssociateType
	IoOptimized             StringOrBool
	InstanceChargeType      common.InstanceChargeType
	ExpiredTime             util.ISO6801Time
	Tags                    struct {
		Tag []TagItemType
	}
	SpotStrategy SpotStrategyType
}

type DescribeInstanceAttributeResponse struct {
	common.Response
	InstanceAttributesType
}

// DescribeInstanceAttribute describes instance attribute
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&describeinstanceattribute
func (client *Client) DescribeInstanceAttribute(instanceId string) (instance *InstanceAttributesType, err error) {
	args := DescribeInstanceAttributeArgs{InstanceId: instanceId}

	response := DescribeInstanceAttributeResponse{}
	err = client.Invoke("DescribeInstanceAttribute", &args, &response)
	if err != nil {
		return nil, err
	}
	return &response.InstanceAttributesType, err
}

type ModifyInstanceAttributeArgs struct {
	InstanceId   string
	InstanceName string
	Description  string
	Password     string
	HostName     string
}

type ModifyInstanceAttributeResponse struct {
	common.Response
}

//ModifyInstanceAttribute  modify instance attrbute
//
// You can read doc at https://help.aliyun.com/document_detail/ecs/open-api/instance/modifyinstanceattribute.html
func (client *Client) ModifyInstanceAttribute(args *ModifyInstanceAttributeArgs) error {
	response := ModifyInstanceAttributeResponse{}
	err := client.Invoke("ModifyInstanceAttribute", args, &response)
	return err
}

// Default timeout value for WaitForInstance method
const InstanceDefaultTimeout = 120

// WaitForInstance waits for instance to given status
func (client *Client) WaitForInstance(instanceId string, status InstanceStatus, timeout int) error {
	if timeout <= 0 {
		timeout = InstanceDefaultTimeout
	}
	for {
		instance, err := client.DescribeInstanceAttribute(instanceId)
		if err != nil {
			return err
		}
		if instance.Status == status {
			//TODO
			//Sleep one more time for timing issues
			time.Sleep(DefaultWaitForInterval * time.Second)
			break
		}
		timeout = timeout - DefaultWaitForInterval
		if timeout <= 0 {
			return common.GetClientErrorFromString("Timeout")
		}
		time.Sleep(DefaultWaitForInterval * time.Second)

	}
	return nil
}

type DescribeInstanceVncUrlArgs struct {
	RegionId   common.Region
	InstanceId string
}

type DescribeInstanceVncUrlResponse struct {
	common.Response
	VncUrl string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&describeinstancevncurl
func (client *Client) DescribeInstanceVncUrl(args *DescribeInstanceVncUrlArgs) (string, error) {
	response := DescribeInstanceVncUrlResponse{}

	err := client.Invoke("DescribeInstanceVncUrl", args, &response)

	if err == nil {
		return response.VncUrl, nil
	}

	return "", err
}

type DescribeInstancesArgs struct {
	RegionId            common.Region
	VpcId               string
	VSwitchId           string
	ZoneId              string
	InstanceIds         string
	InstanceNetworkType string
	InstanceName        string
	Status              InstanceStatus
	PrivateIpAddresses  string
	InnerIpAddresses    string
	PublicIpAddresses   string
	SecurityGroupId     string
	Tag                 map[string]string
	InstanceType        string
	SpotStrategy        SpotStrategyType
	common.Pagination
}

type DescribeInstancesResponse struct {
	common.Response
	common.PaginationResult
	Instances struct {
		Instance []InstanceAttributesType
	}
}

// DescribeInstances describes instances
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&describeinstances
func (client *Client) DescribeInstances(args *DescribeInstancesArgs) (instances []InstanceAttributesType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeInstancesResponse{}

	err = client.Invoke("DescribeInstances", args, &response)

	if err == nil {
		return response.Instances.Instance, &response.PaginationResult, nil
	}

	return nil, nil, err
}

type DeleteInstanceArgs struct {
	InstanceId string
}

type DeleteInstanceResponse struct {
	common.Response
}

// DeleteInstance deletes instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&deleteinstance
func (client *Client) DeleteInstance(instanceId string) error {
	args := DeleteInstanceArgs{InstanceId: instanceId}
	response := DeleteInstanceResponse{}
	err := client.Invoke("DeleteInstance", &args, &response)
	return err
}

type DataDiskType struct {
	Size               int
	Category           DiskCategory //Enum cloud, ephemeral, ephemeral_ssd
	SnapshotId         string
	DiskName           string
	Description        string
	Device             string
	DeleteWithInstance bool
}

type SystemDiskType struct {
	Size        int
	Category    DiskCategory //Enum cloud, ephemeral, ephemeral_ssd
	DiskName    string
	Description string
}

type IoOptimized string

type StringOrBool struct {
	Value bool
}

// UnmarshalJSON implements the json.Unmarshaller interface.
func (io *StringOrBool) UnmarshalJSON(value []byte) error {
	if value[0] == '"' {
		var str string
		err := json.Unmarshal(value, &str)
		if err == nil {
			io.Value = (str == "true" || str == "optimized")
		}
		return err
	}
	var boolVal bool
	err := json.Unmarshal(value, &boolVal)
	if err == nil {
		io.Value = boolVal
	}
	return err
}

func (io StringOrBool) Bool() bool {
	return io.Value
}

func (io StringOrBool) String() string {
	return strconv.FormatBool(io.Value)
}

var (
	IoOptimizedNone      = IoOptimized("none")
	IoOptimizedOptimized = IoOptimized("optimized")
)

type CreateInstanceArgs struct {
	RegionId                common.Region
	ZoneId                  string
	ImageId                 string
	InstanceType            string
	SecurityGroupId         string
	InstanceName            string
	Description             string
	InternetChargeType      common.InternetChargeType
	InternetMaxBandwidthIn  int
	InternetMaxBandwidthOut int
	HostName                string
	Password                string
	IoOptimized             IoOptimized
	SystemDisk              SystemDiskType
	DataDisk                []DataDiskType
	VSwitchId               string
	PrivateIpAddress        string
	ClientToken             string
	InstanceChargeType      common.InstanceChargeType
	Period                  int
	UserData                string
	AutoRenew               bool
	AutoRenewPeriod         int
	SpotStrategy            SpotStrategyType
}

type CreateInstanceResponse struct {
	common.Response
	InstanceId string
}

// CreateInstance creates instance
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/instance&createinstance
func (client *Client) CreateInstance(args *CreateInstanceArgs) (instanceId string, err error) {
	if args.UserData != "" {
		// Encode to base64 string
		args.UserData = base64.StdEncoding.EncodeToString([]byte(args.UserData))
	}
	response := CreateInstanceResponse{}
	err = client.Invoke("CreateInstance", args, &response)
	if err != nil {
		return "", err
	}
	return response.InstanceId, err
}

type SecurityGroupArgs struct {
	InstanceId      string
	SecurityGroupId string
}

type SecurityGroupResponse struct {
	common.Response
}

//JoinSecurityGroup
//
//You can read doc at https://help.aliyun.com/document_detail/ecs/open-api/instance/joinsecuritygroup.html
func (client *Client) JoinSecurityGroup(instanceId string, securityGroupId string) error {
	args := SecurityGroupArgs{InstanceId: instanceId, SecurityGroupId: securityGroupId}
	response := SecurityGroupResponse{}
	err := client.Invoke("JoinSecurityGroup", &args, &response)
	return err
}

//LeaveSecurityGroup
//
//You can read doc at https://help.aliyun.com/document_detail/ecs/open-api/instance/leavesecuritygroup.html
func (client *Client) LeaveSecurityGroup(instanceId string, securityGroupId string) error {
	args := SecurityGroupArgs{InstanceId: instanceId, SecurityGroupId: securityGroupId}
	response := SecurityGroupResponse{}
	err := client.Invoke("LeaveSecurityGroup", &args, &response)
	return err
}
