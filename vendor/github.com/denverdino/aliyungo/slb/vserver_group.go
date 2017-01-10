package slb

import (
	"github.com/denverdino/aliyungo/common"
)

type VBackendServerType struct {
	ServerId string
	Weight   int
	Port     int
}

type VServerGroup struct {
	VServerGroupName string
	VServerGroupId   string
}

type VBackendServers struct {
	BackendServer []VBackendServerType
}

type CreateVServerGroupArgs struct {
	LoadBalancerId   string
	RegionId         common.Region
	VServerGroupName string
	VServerGroupId   string
	BackendServers   string
}

type SetVServerGroupAttributeArgs struct {
	LoadBalancerId   string
	RegionId         common.Region
	VServerGroupName string
	VServerGroupId   string
	BackendServers   string
}

type AddVServerGroupBackendServersArgs CreateVServerGroupArgs
type RemoveVServerGroupBackendServersArgs CreateVServerGroupArgs
type ModifyVServerGroupBackendServersArgs struct {
	VServerGroupId    string
	RegionId       common.Region
	OldBackendServers string
	NewBackendServers string
}

type DeleteVServerGroupArgs struct {
	VServerGroupId string
	RegionId common.Region
}

type DescribeVServerGroupsArgs struct {
	LoadBalancerId string
	RegionId       common.Region
}

type DescribeVServerGroupAttributeArgs struct {
	VServerGroupId string
	RegionId       common.Region
}

type CreateVServerGroupResponse struct {
	common.Response
	VServerGroupId   string
	VServerGroupName string
	BackendServers   VBackendServers
}

type SetVServerGroupAttributeResponse struct {
	common.Response
	VServerGroupId   string
	VServerGroupName string
	BackendServers   VBackendServers
}

type AddVServerGroupBackendServersResponse CreateVServerGroupResponse
type RemoveVServerGroupBackendServersResponse CreateVServerGroupResponse
type ModifyVServerGroupBackendServersResponse CreateVServerGroupResponse
type DeleteVServerGroupResponse struct{ common.Response }
type DescribeVServerGroupsResponse struct {
	common.Response
	VServerGroups struct {
		VServerGroup []VServerGroup
	}
}
type DescribeVServerGroupAttributeResponse CreateVServerGroupResponse


func (client *Client) CreateVServerGroup(args *CreateVServerGroupArgs) (response *CreateVServerGroupResponse, err error) {
	response = &CreateVServerGroupResponse{}
	err = client.Invoke("CreateVServerGroup", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) SetVServerGroupAttribute(args *SetVServerGroupAttributeArgs) (response *SetVServerGroupAttributeResponse, err error) {
	response = &SetVServerGroupAttributeResponse{}
	err = client.Invoke("SetVServerGroupAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) AddVServerGroupBackendServers(args *AddVServerGroupBackendServersArgs) (response *AddVServerGroupBackendServersResponse, err error) {
	response = &AddVServerGroupBackendServersResponse{}
	err = client.Invoke("AddVServerGroupBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) RemoveVServerGroupBackendServers(args *RemoveVServerGroupBackendServersArgs) (response *RemoveVServerGroupBackendServersResponse, err error) {
	response = &RemoveVServerGroupBackendServersResponse{}
	err = client.Invoke("RemoveVServerGroupBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) ModifyVServerGroupBackendServers(args *ModifyVServerGroupBackendServersArgs) (response *ModifyVServerGroupBackendServersResponse, err error) {
	response = &ModifyVServerGroupBackendServersResponse{}
	err = client.Invoke("ModifyVServerGroupBackendServers", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) DeleteVServerGroup(args *DeleteVServerGroupArgs) (response *DeleteVServerGroupResponse, err error) {
	response = &DeleteVServerGroupResponse{}
	err = client.Invoke("DeleteVServerGroup", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) DescribeVServerGroups(args *DescribeVServerGroupsArgs) (response *DescribeVServerGroupsResponse, err error) {
	response = &DescribeVServerGroupsResponse{}
	err = client.Invoke("DescribeVServerGroups", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

func (client *Client) DescribeVServerGroupAttribute(args *DescribeVServerGroupAttributeArgs) (response *DescribeVServerGroupAttributeResponse, err error) {
	response = &DescribeVServerGroupAttributeResponse{}
	err = client.Invoke("DescribeVServerGroupAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}
