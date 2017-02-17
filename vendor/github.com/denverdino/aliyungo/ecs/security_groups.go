package ecs

import (
	"github.com/denverdino/aliyungo/common"
	"github.com/denverdino/aliyungo/util"
)

type NicType string

const (
	NicTypeInternet = NicType("internet")
	NicTypeIntranet = NicType("intranet")
)

type IpProtocol string

const (
	IpProtocolAll  = IpProtocol("all")
	IpProtocolTCP  = IpProtocol("tcp")
	IpProtocolUDP  = IpProtocol("udp")
	IpProtocolICMP = IpProtocol("icmp")
	IpProtocolGRE  = IpProtocol("gre")
)

type PermissionPolicy string

const (
	PermissionPolicyAccept = PermissionPolicy("accept")
	PermissionPolicyDrop   = PermissionPolicy("drop")
)

type DescribeSecurityGroupAttributeArgs struct {
	SecurityGroupId string
	RegionId        common.Region
	NicType         NicType //enum for internet (default) |intranet
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&permissiontype
type PermissionType struct {
	IpProtocol              IpProtocol
	PortRange               string
	SourceCidrIp            string
	SourceGroupId           string
	SourceGroupOwnerAccount string
	DestCidrIp              string
	DestGroupId             string
	DestGroupOwnerAccount   string
	Policy                  PermissionPolicy
	NicType                 NicType
	Priority                int
	Direction               string
	Description             string
}

type DescribeSecurityGroupAttributeResponse struct {
	common.Response

	SecurityGroupId   string
	SecurityGroupName string
	RegionId          common.Region
	Description       string
	Permissions       struct {
		Permission []PermissionType
	}
	VpcId string
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&describesecuritygroupattribute
func (client *Client) DescribeSecurityGroupAttribute(args *DescribeSecurityGroupAttributeArgs) (response *DescribeSecurityGroupAttributeResponse, err error) {
	response = &DescribeSecurityGroupAttributeResponse{}
	err = client.Invoke("DescribeSecurityGroupAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, nil
}

type DescribeSecurityGroupsArgs struct {
	RegionId common.Region
	VpcId    string
	common.Pagination
}

//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/datatype&securitygroupitemtype
type SecurityGroupItemType struct {
	SecurityGroupId   string
	SecurityGroupName string
	Description       string
	VpcId             string
	CreationTime      util.ISO6801Time
}

type DescribeSecurityGroupsResponse struct {
	common.Response
	common.PaginationResult

	RegionId       common.Region
	SecurityGroups struct {
		SecurityGroup []SecurityGroupItemType
	}
}

// DescribeSecurityGroups describes security groups
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&describesecuritygroups
func (client *Client) DescribeSecurityGroups(args *DescribeSecurityGroupsArgs) (securityGroupItems []SecurityGroupItemType, pagination *common.PaginationResult, err error) {
	args.Validate()
	response := DescribeSecurityGroupsResponse{}

	err = client.Invoke("DescribeSecurityGroups", args, &response)

	if err != nil {
		return nil, nil, err
	}

	return response.SecurityGroups.SecurityGroup, &response.PaginationResult, nil
}

type CreateSecurityGroupArgs struct {
	RegionId          common.Region
	SecurityGroupName string
	Description       string
	VpcId             string
	ClientToken       string
}

type CreateSecurityGroupResponse struct {
	common.Response

	SecurityGroupId string
}

// CreateSecurityGroup creates security group
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&createsecuritygroup
func (client *Client) CreateSecurityGroup(args *CreateSecurityGroupArgs) (securityGroupId string, err error) {
	response := CreateSecurityGroupResponse{}
	err = client.Invoke("CreateSecurityGroup", args, &response)
	if err != nil {
		return "", err
	}
	return response.SecurityGroupId, err
}

type DeleteSecurityGroupArgs struct {
	RegionId        common.Region
	SecurityGroupId string
}

type DeleteSecurityGroupResponse struct {
	common.Response
}

// DeleteSecurityGroup deletes security group
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&deletesecuritygroup
func (client *Client) DeleteSecurityGroup(regionId common.Region, securityGroupId string) error {
	args := DeleteSecurityGroupArgs{
		RegionId:        regionId,
		SecurityGroupId: securityGroupId,
	}
	response := DeleteSecurityGroupResponse{}
	err := client.Invoke("DeleteSecurityGroup", &args, &response)
	return err
}

type ModifySecurityGroupAttributeArgs struct {
	RegionId          common.Region
	SecurityGroupId   string
	SecurityGroupName string
	Description       string
}

type ModifySecurityGroupAttributeResponse struct {
	common.Response
}

// ModifySecurityGroupAttribute modifies attribute of security group
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&modifysecuritygroupattribute
func (client *Client) ModifySecurityGroupAttribute(args *ModifySecurityGroupAttributeArgs) error {
	response := ModifySecurityGroupAttributeResponse{}
	err := client.Invoke("ModifySecurityGroupAttribute", args, &response)
	return err
}

type AuthorizeSecurityGroupArgs struct {
	SecurityGroupId         string
	RegionId                common.Region
	IpProtocol              IpProtocol
	PortRange               string
	SourceGroupId           string
	SourceGroupOwnerAccount string
	SourceGroupOwnerID      string
	SourceCidrIp            string           // IPv4 only, default 0.0.0.0/0
	Policy                  PermissionPolicy // enum of accept (default) | drop
	Priority                int              // 1 - 100, default 1
	NicType                 NicType          // enum of internet | intranet (default)
}

type AuthorizeSecurityGroupResponse struct {
	common.Response
}

// AuthorizeSecurityGroup authorize permissions to security group
//
// You can read doc at http://docs.aliyun.com/#/pub/ecs/open-api/securitygroup&authorizesecuritygroup
func (client *Client) AuthorizeSecurityGroup(args *AuthorizeSecurityGroupArgs) error {
	response := AuthorizeSecurityGroupResponse{}
	err := client.Invoke("AuthorizeSecurityGroup", args, &response)
	return err
}

type RevokeSecurityGroupArgs struct {
	AuthorizeSecurityGroupArgs
}

type RevokeSecurityGroupResponse struct {
	common.Response
}

// You can read doc at https://help.aliyun.com/document_detail/25557.html?spm=5176.doc25554.6.755.O6Tjz0
func (client *Client) RevokeSecurityGroup(args *RevokeSecurityGroupArgs) error {
	response := RevokeSecurityGroupResponse{}
	err := client.Invoke("RevokeSecurityGroup", args, &response)
	return err
}

type AuthorizeSecurityGroupEgressArgs struct {
	SecurityGroupId       string
	RegionId              common.Region
	IpProtocol            IpProtocol
	PortRange             string
	DestGroupId           string
	DestGroupOwnerAccount string
	DestGroupOwnerId      string
	DestCidrIp            string           // IPv4 only, default 0.0.0.0/0
	Policy                PermissionPolicy // enum of accept (default) | drop
	Priority              int              // 1 - 100, default 1
	NicType               NicType          // enum of internet | intranet (default)
}

type AuthorizeSecurityGroupEgressResponse struct {
	common.Response
}

// AuthorizeSecurityGroup authorize permissions to security group
//
// You can read doc at https://help.aliyun.com/document_detail/25560.html
func (client *Client) AuthorizeSecurityGroupEgress(args *AuthorizeSecurityGroupEgressArgs) error {
	response := AuthorizeSecurityGroupEgressResponse{}
	err := client.Invoke("AuthorizeSecurityGroupEgress", args, &response)
	return err
}

type RevokeSecurityGroupEgressArgs struct {
	AuthorizeSecurityGroupEgressArgs
}

type RevokeSecurityGroupEgressResponse struct {
	common.Response
}

// You can read doc at https://help.aliyun.com/document_detail/25561.html?spm=5176.doc25557.6.759.qcR4Az
func (client *Client) RevokeSecurityGroupEgress(args *RevokeSecurityGroupEgressArgs) error {
	response := RevokeSecurityGroupEgressResponse{}
	err := client.Invoke("RevokeSecurityGroupEgress", args, &response)
	return err
}
