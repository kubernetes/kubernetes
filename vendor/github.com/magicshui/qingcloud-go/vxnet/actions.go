package vxnet

import (
	"github.com/magicshui/qingcloud-go"
)

type VXNET struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *VXNET {
	return &VXNET{
		Client: clt,
	}
}

type DescribeVxnetsRequest struct {
	VxnetsN    qingcloud.NumberedString
	VxnetType  qingcloud.Integer
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	OffSet     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeVxnetsResponse struct {
	VxnetSet []Vxnet `json:"vxnet_set"`
	qingcloud.CommonResponse
}

// DescribeVxnets 可根据私有网络ID作过滤条件，获取私有网络列表。 如果不指定任何过滤条件，默认返回你所拥有的所有私有网络。
func (c *VXNET) DescribeVxnets(params DescribeVxnetsRequest) (DescribeVxnetsResponse, error) {
	var result DescribeVxnetsResponse
	err := c.Get("DescribeVxnets", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateVxnetsRequest struct {
	VxnetName qingcloud.String
	VxnetType qingcloud.Integer
	Count     qingcloud.Integer
}
type CreateVxnetsResponse struct {
	Vxnets []string `json:"vxnets"`
	qingcloud.CommonResponse
}

// CreateVxnets 创建新的私有网络。
// 青云私有网络有两种类型： 受管私有网络 ( vxnet_type=1 ) 和 自管私有网络 ( vxnet_type=0 ) ，
// 受管私有网络可以使用青云路由器来配置和管理其网络，使得网络搭建更方便快捷。
// 自管私有网络需要您自行配置和管理网络，适用于对底层网络有特殊需求的用户。
func (c *VXNET) CreateVxnets(params CreateVxnetsRequest) (CreateVxnetsResponse, error) {
	var result CreateVxnetsResponse
	err := c.Get("CreateVxnets", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteVxnetsRequest struct {
	VxnetsN qingcloud.NumberedString
}
type DeleteVxnetsResponse struct {
	Vxnets []string `json:"vxnets"`
	qingcloud.CommonResponse
}

// DeleteVxnets 删除私有网络。
// 只能删除没有主机的私有网络，若删除时仍然有主机在此网络中，会返回错误信息。 可通过 LeaveVxnet 移出主机。
func (c *VXNET) DeleteVxnets(params DeleteVxnetsRequest) (DeleteVxnetsResponse, error) {
	var result DeleteVxnetsResponse
	err := c.Get("DeleteVxnets", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type JoinVxnetRequest struct {
	Vxnet      qingcloud.String
	InstancesN qingcloud.NumberedString
}
type JoinVxnetResponse qingcloud.CommonResponse

// JoinVxnet 将主机加入到私有网络。
// 警告 一台主机最多只能加入一个受管网络 ( 包括基础网络vxnet-0 )
func (c *VXNET) JoinVxnet(params JoinVxnetRequest) (JoinVxnetResponse, error) {
	var result JoinVxnetResponse
	err := c.Get("JoinVxnet", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type LeaveVxnetRequest struct {
	Vxnet      qingcloud.String
	InstancesN qingcloud.NumberedString
}
type LeaveVxnetResponse qingcloud.CommonResponse

// LeaveVxnet 将主机从私有网络中断开。
func (c *VXNET) LeaveVxnet(params LeaveVxnetRequest) (LeaveVxnetResponse, error) {
	var result LeaveVxnetResponse
	err := c.Get("LeaveVxnet", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyVxnetAttributesRequest struct {
	Vxnet       qingcloud.String
	VxnetName   qingcloud.String
	Description qingcloud.String
}
type ModifyVxnetAttributesResponse qingcloud.CommonResponse

// ModifyVxnetAttributes 修改私有网络的名称和描述。
// 一次只能修改一个私有网络。
func (c *VXNET) ModifyVxnetAttributes(params ModifyVxnetAttributesRequest) (ModifyVxnetAttributesResponse, error) {
	var result ModifyVxnetAttributesResponse
	err := c.Get("ModifyVxnetAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeVxnetInstancesRequest struct {
	Vxnet        qingcloud.String
	InstancesN   qingcloud.NumberedString
	InstanceType qingcloud.String
	Status       qingcloud.String
	Image        qingcloud.String
	OffSet       qingcloud.Integer
	Limit        qingcloud.Integer
}
type DescribeVxnetInstancesResponse struct {
	InstanceSet []Instance `json:"instance_set"`
	TotalCount  int        `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeVxnetInstances 获取私有网络中的主机。
// 可通过主机ID，映像ID，主机配置类型，主机状态作为过滤条件进行筛选。
func (c *VXNET) DescribeVxnetInstances(params DescribeVxnetInstancesRequest) (DescribeVxnetInstancesResponse, error) {
	var result DescribeVxnetInstancesResponse
	err := c.Get("DescribeVxnetInstances", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
