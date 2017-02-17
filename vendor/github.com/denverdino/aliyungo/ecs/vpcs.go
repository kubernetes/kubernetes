package ecs

import (
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type CreateVpcArgs struct {
	RegionId    common.Region
	CidrBlock   string //192.168.0.0/16 or 172.16.0.0/16 (default)
	VpcName     string
	Description string
	ClientToken string
}

type CreateVpcResponse struct {
	common.Response
	VpcId        string
	VRouterId    string
	RouteTableId string
}

// CreateVpc creates Virtual Private Cloud
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vpc&createvpc
func (client *Client) CreateVpc(args *CreateVpcArgs) (resp *CreateVpcResponse, err error) {
	response := CreateVpcResponse{}
	err = client.Invoke("CreateVpc", args, &response)
	if err != nil {
		return nil, err
	}
	return &response, err
}

type DeleteVpcArgs struct {
	VpcId string
}

type DeleteVpcResponse struct {
	common.Response
}

// DeleteVpc deletes Virtual Private Cloud
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vpc&deletevpc
func (client *Client) DeleteVpc(vpcId string) error {
	args := DeleteVpcArgs{
		VpcId: vpcId,
	}
	response := DeleteVpcResponse{}
	return client.Invoke("DeleteVpc", &args, &response)
}

type VpcStatus string

const (
	VpcStatusPending   = VpcStatus("Pending")
	VpcStatusAvailable = VpcStatus("Available")
)

type DescribeVpcsArgs struct {
	VpcId    string
	RegionId common.Region
	common.Pagination
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&vpcsettype
type VpcSetType struct {
	VpcId      string
	RegionId   common.Region
	Status     VpcStatus // enum Pending | Available
	VpcName    string
	VSwitchIds struct {
		VSwitchId []string
	}
	CidrBlock    string
	VRouterId    string
	Description  string
	CreationTime util.ISO6801Time
}

type DescribeVpcsResponse struct {
	common.Response
	common.PaginationResult
	Vpcs struct {
		Vpc []VpcSetType
	}
}

// DescribeInstanceStatus describes instance status
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vpc&describevpcs
func (client *Client) DescribeVpcs(args *DescribeVpcsArgs) (vpcs []VpcSetType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeVpcsResponse{}

	err = client.Invoke("DescribeVpcs", args, &response)

	if err == nil {
		return response.Vpcs.Vpc, &response.PaginationResult, nil
	}

	return nil, nil, err
}

type ModifyVpcAttributeArgs struct {
	VpcId       string
	VpcName     string
	Description string
}

type ModifyVpcAttributeResponse struct {
	common.Response
}

// ModifyVpcAttribute modifies attribute of Virtual Private Cloud
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vpc&modifyvpcattribute
func (client *Client) ModifyVpcAttribute(args *ModifyVpcAttributeArgs) error {
	response := ModifyVpcAttributeResponse{}
	return client.Invoke("ModifyVpcAttribute", args, &response)
}

// WaitForInstance waits for instance to given status
func (client *Client) WaitForVpcAvailable(regionId common.Region, vpcId string, timeout int) error {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	args := DescribeVpcsArgs{
		RegionId: regionId,
		VpcId:    vpcId,
	}
	for {
		vpcs, _, err := client.DescribeVpcs(&args)
		if err != nil {
			return err
		}
		if len(vpcs) > 0 && vpcs[0].Status == VpcStatusAvailable {
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
