package ecs

import (
	"time"

	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type CreateVSwitchArgs struct {
	ZoneId      string
	CidrBlock   string
	VpcId       string
	VSwitchName string
	Description string
	ClientToken string
}

type CreateVSwitchResponse struct {
	common.Response
	VSwitchId string
}

// CreateVSwitch creates Virtual Switch
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vswitch&createvswitch
func (client *Client) CreateVSwitch(args *CreateVSwitchArgs) (vswitchId string, err error) {
	response := CreateVSwitchResponse{}
	err = client.Invoke("CreateVSwitch", args, &response)
	if err != nil {
		return "", err
	}
	return response.VSwitchId, err
}

type DeleteVSwitchArgs struct {
	VSwitchId string
}

type DeleteVSwitchResponse struct {
	common.Response
}

// DeleteVSwitch deletes Virtual Switch
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vswitch&deletevswitch
func (client *Client) DeleteVSwitch(VSwitchId string) error {
	args := DeleteVSwitchArgs{
		VSwitchId: VSwitchId,
	}
	response := DeleteVSwitchResponse{}
	return client.Invoke("DeleteVSwitch", &args, &response)
}

type DescribeVSwitchesArgs struct {
	VpcId     string
	VSwitchId string
	ZoneId    string
	common.Pagination
}

type VSwitchStatus string

const (
	VSwitchStatusPending   = VSwitchStatus("Pending")
	VSwitchStatusAvailable = VSwitchStatus("Available")
)

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&vswitchsettype
type VSwitchSetType struct {
	VSwitchId               string
	VpcId                   string
	Status                  VSwitchStatus // enum Pending | Available
	CidrBlock               string
	ZoneId                  string
	AvailableIpAddressCount int
	Description             string
	VSwitchName             string
	CreationTime            util.ISO6801Time
}

type DescribeVSwitchesResponse struct {
	common.Response
	common.PaginationResult
	VSwitches struct {
		VSwitch []VSwitchSetType
	}
}

// DescribeVSwitches describes Virtual Switches
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vswitch&describevswitches
func (client *Client) DescribeVSwitches(args *DescribeVSwitchesArgs) (vswitches []VSwitchSetType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeVSwitchesResponse{}

	err = client.Invoke("DescribeVSwitches", args, &response)

	if err == nil {
		return response.VSwitches.VSwitch, &response.PaginationResult, nil
	}

	return nil, nil, err
}

type ModifyVSwitchAttributeArgs struct {
	VSwitchId   string
	VSwitchName string
	Description string
}

type ModifyVSwitchAttributeResponse struct {
	common.Response
}

// ModifyVSwitchAttribute modifies attribute of Virtual Private Cloud
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/vswitch&modifyvswitchattribute
func (client *Client) ModifyVSwitchAttribute(args *ModifyVSwitchAttributeArgs) error {
	response := ModifyVSwitchAttributeResponse{}
	return client.Invoke("ModifyVSwitchAttribute", args, &response)
}

// WaitForVSwitchAvailable waits for VSwitch to given status
func (client *Client) WaitForVSwitchAvailable(vpcId string, vswitchId string, timeout int) error {
	if timeout <= 0 {
		timeout = DefaultTimeout
	}
	args := DescribeVSwitchesArgs{
		VpcId:     vpcId,
		VSwitchId: vswitchId,
	}
	for {
		vswitches, _, err := client.DescribeVSwitches(&args)
		if err != nil {
			return err
		}
		if len(vswitches) == 0 {
			return common.GetClientErrorFromString("Not found")
		}
		if vswitches[0].Status == VSwitchStatusAvailable {
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
