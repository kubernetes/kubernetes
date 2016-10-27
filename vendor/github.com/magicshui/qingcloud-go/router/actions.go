package router

import (
	"github.com/magicshui/qingcloud-go"
)

// ROUTER 路由
type ROUTER struct {
	*qingcloud.Client
}

// NewClient 创建新的路由控制器
func NewClient(clt *qingcloud.Client) *ROUTER {
	return &ROUTER{
		Client: clt,
	}
}

// DescribeRoutersRequest 创建参数
type DescribeRoutersRequest struct {
	RoutersN   qingcloud.NumberedString
	Vxnet      qingcloud.String
	StatusN    qingcloud.String
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}

// DescribeRoutersResponse 返回参数
type DescribeRoutersResponse struct {
	Action     string   `json:"action"`
	RouterSet  []Router `json:"router_set"`
	TotalCount int      `json:"total_count"`
	qingcloud.CommonResponse
}

//DescribeRouters 获取一个或多个路由器
// 可根据路由器ID，状态，路由器名称作过滤条件，来获取路由器列表。 如果不指定任何过滤条件，默认返回你所拥有的所有路由器。 如果指定不存在的路由器ID，或非法状态值，则会返回错误信息。
func (c *ROUTER) DescribeRouters(params DescribeRoutersRequest) (DescribeRoutersResponse, error) {
	var result DescribeRoutersResponse
	err := c.Get("DescribeRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateRoutersRequest struct {
	RouterName    qingcloud.String
	RouterType    qingcloud.Integer
	VpcNetwork    qingcloud.String
	Count         qingcloud.Integer
	SecurityGroup qingcloud.String
}
type CreateRoutersResponse struct {
	Routers []string `json:"routers"`
	qingcloud.CommonResponse
}

// CreateRouters  创建一台或多台路由器。路由器用于受管私有网络之间互联，并提供三项附加服务：DHCP 服务、端口转发、VPN 隧道服务。
// 这个API只负责路由器的创建工作，如果需要通过路由器将自己名下的受管私有网络连接起来，请查看 JoinRouter。
// 如果需要配置端口转发规则或打开VPN 隧道服务，请查看 AddRouterStatics 和 UpdateRouters。
func (c *ROUTER) CreateRouters(params CreateRoutersRequest) (CreateRoutersResponse, error) {
	var result CreateRoutersResponse
	err := c.Get("CreateRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteRoutersRequest struct {
	RoutersN qingcloud.NumberedString
}
type DeleteRoutersResponse qingcloud.CommonResponse

// DeleteRouters 删除一台或多台路由器。
// 销毁路由器的前提，是此路由器已建立租用信息（租用信息是在创建路由器成功后， 几秒钟内系统自动建立的）。所以正在创建的路由器（状态为 pending ）， 以及刚刚创建成功但还没有建立租用信息的路由器，是不能被销毁的。
func (c *ROUTER) DeleteRouters(params DeleteRoutersRequest) (DeleteRoutersResponse, error) {
	var result DeleteRoutersResponse
	err := c.Get("DeleteRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type UpdateRoutersRequest struct {
	RoutersN qingcloud.NumberedString
}
type UpdateRoutersResponse qingcloud.CommonResponse

// UpdateRouters 更新一台或多台路由器的配置信息。当配置发生变更之后，需要执行本操作使配置生效。
// 可以使路由器配置发生变更的操作为 AddRouterStatics 和 DeleteRouterStatics 和 ModifyRouterAttributes。
// 只有在处于 active 状态的路由器才能支持此操作，如果处于非活跃状态，则返回错误信息。
func (c *ROUTER) UpdateRouters(params UpdateRoutersRequest) (UpdateRoutersResponse, error) {
	var result UpdateRoutersResponse
	err := c.Get("UpdateRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type PowerOffRoutersRequest struct {
	RoutersN qingcloud.NumberedString
}
type PowerOffRoutersResponse qingcloud.CommonResponse

// PowerOffRouters 关闭一台或多台路由器。
// 路由器只有在运行 active 状态才能被关闭，如果处于非运行状态，则返回错误信息。
func (c *ROUTER) PowerOffRouters(params PowerOffRoutersRequest) (PowerOffRoutersResponse, error) {
	var result PowerOffRoutersResponse
	err := c.Get("PowerOffRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type PowerOnRoutersRequest struct {
	RoutersN qingcloud.NumberedString
}
type PowerOnRoutersResponse qingcloud.CommonResponse

// PowerOnRouters 启动一台或多台路由器。
// 路由器只有在关闭 poweroffed 状态才能被启动，如果处于非关闭状态，则返回错误信息。
func (c *ROUTER) PowerOnRouters(params PowerOnRoutersRequest) (PowerOnRoutersResponse, error) {
	var result PowerOnRoutersResponse
	err := c.Get("PowerOnRouters", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type JoinRouterRequest struct {
	Vxnet      qingcloud.String
	Router     qingcloud.String
	IpNetwork  qingcloud.String
	Feature    qingcloud.Integer
	ManagerIP  qingcloud.String
	DynIpStart qingcloud.String
	DynIpEnd   qingcloud.String
}
type JoinRouterResponse qingcloud.CommonResponse

// JoinRouter 将一个受管私有网络连接到一台路由器。这样受管私有网络可以被路由器管理起来， 受管私有网络里的主机也将获得 DHCP 自动分配地址的能力。
// 只有受管私有网络才能连接到路由器，一个受管私有网络可以且仅可以连接到一台路由器。 受管私有网络可以连接到状态为 active 和 poweroffed 的路由器。
func (c *ROUTER) JoinRouter(params JoinRouterRequest) (JoinRouterResponse, error) {
	var result JoinRouterResponse
	err := c.Get("JoinRouter", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type LeaveRouterRequest struct {
	VxnetsN qingcloud.NumberedString
	Router  qingcloud.String
}
type LeaveRouterResponse qingcloud.CommonResponse

// LeaveRouter 将一个或多个受管私有网络从一台路由器上断开。
func (c *ROUTER) LeaveRouter(params LeaveRouterRequest) (LeaveRouterResponse, error) {
	var result LeaveRouterResponse
	err := c.Get("LeaveRouter", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyRouterAttributesRequest struct {
	Router        qingcloud.String
	Eip           qingcloud.String
	SecurityGroup qingcloud.String
	RouterName    qingcloud.String
	Description   qingcloud.String
	DynIpStart    qingcloud.String
	DynIpEnd      qingcloud.String
}
type ModifyRouterAttributesResponse qingcloud.CommonResponse

// ModifyRouterAttributes
// 修改一台路由器的配置。在修改配置之后，为了让配置生效，你可能需要执行 UpdateRouters 或者 ApplySecurityGroup 指令。
func (c *ROUTER) ModifyRouterAttributes(params ModifyRouterAttributesRequest) (ModifyRouterAttributesResponse, error) {
	var result ModifyRouterAttributesResponse
	err := c.Get("ModifyRouterAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeRouterStaticsRequest struct {
	RouterStaticsN qingcloud.NumberedString
	Router         qingcloud.String
	Vxnet          qingcloud.String
	StaticType     qingcloud.String
	Verbose        qingcloud.Integer
	Offset         qingcloud.Integer
	Limit          qingcloud.Integer
}
type DescribeRouterStaticsReponse struct {
	RouterStaticSet []RouterStatic `json:"router_static_set"`
	TotalCount      int            `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeRouterStatics
// 获取路由器的规则。
// 可根据路由器规则ID，路由器ID，规则类型等作为过滤条件，来获取路由器规则列表。 如果不指定任何过滤条件，默认返回你所拥有的所有路由器规则。
func (c *ROUTER) DescribeRouterStatics(params DescribeRouterStaticsRequest) (DescribeRouterStaticsReponse, error) {
	var result DescribeRouterStaticsReponse
	err := c.Get("DescribeRouterStatics", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddRouterStaticsRequest struct {
	Router                   qingcloud.String
	StaticsNRouterStaticName qingcloud.NumberedString
	StaticsNStaticType       qingcloud.NumberedInteger
	StaticsNVal1             qingcloud.NumberedString
	StaticsNVal2             qingcloud.NumberedString
	StaticsNVal3             qingcloud.NumberedString
	StaticsNVal4             qingcloud.NumberedString
	StaticsNVal5             qingcloud.NumberedString
}
type AddRouterStaticsResponse struct {
	qingcloud.CommonResponse
	RouterStatics []string `json:"router_statics"`
}

// AddRouterStatics
// 增加一条或多条路由器规则，规则包括：端口转发、VPN 、DHCP 、隧道、过滤控制。 注意：在增加路由器规则后，你需要执行 UpdateRouters 才能使之生效。
func (c *ROUTER) AddRouterStatics(params AddRouterStaticsRequest) (AddRouterStaticsResponse, error) {
	var result AddRouterStaticsResponse
	err := c.Get("AddRouterStatics", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyRouterStaticAttributesRequest struct {
	RouterStatic     qingcloud.String
	RouterStaticName qingcloud.String
	Val1             qingcloud.String
	Val2             qingcloud.String
	Val3             qingcloud.String
	Val4             qingcloud.String
	Val5             qingcloud.String
	Val6             qingcloud.String
}
type ModifyRouterStaticAttributesResponse struct {
	qingcloud.CommonResponse
	RouterStaticId string `json:"router_static_id"`
}

// ModifyRouterStaticAttributes
// 修改某条路由器规则。修改规则后，需要执行 UpdateRouters 来使规则生效。
func (c *ROUTER) ModifyRouterStaticAttributes(params ModifyRouterStaticAttributesRequest) (ModifyRouterStaticAttributesResponse, error) {
	var result ModifyRouterStaticAttributesResponse
	err := c.Get("ModifyRouterStaticAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteRouterStaticsRequest struct {
	RouterStaticsN qingcloud.NumberedString
}
type DeleteRouterStaticsResponse struct {
	qingcloud.CommonResponse
	RouterStatics []string `json:"router_statics"`
}

// DeleteRouterStatics
// 删除一条或多条路由器规则。在删除路由器规则之后，你需要执行 UpdateRouters 来使规则删除生效。
func (c *ROUTER) DeleteRouterStatics(params DeleteRouterStaticsRequest) (DeleteRouterStaticsResponse, error) {
	var result DeleteRouterStaticsResponse
	err := c.Get("DeleteRouterStatics", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeRouterVxnetsReqeust struct {
	Router  qingcloud.String
	Vxnet   qingcloud.String
	Verbose qingcloud.Integer
	Offset  qingcloud.Integer
	Limit   qingcloud.Integer
}
type DescribeRouterVxnetsResponse struct {
	RouterVxnetSet []RouterVxnet `json:"router_vxnet_set"`
	TotalCount     int           `json:"total_count"`
}

// DescribeRouterVxnets
// 获取路由器管理的私有网络列表。
// 可根据路由器ID，私有网络ID，等作为过滤条件，来获取私有网络列表。
func (c *ROUTER) DescribeRouterVxnets(params DescribeRouterVxnetsReqeust) (DescribeRouterVxnetsResponse, error) {
	var result DescribeRouterVxnetsResponse
	err := c.Get("DescribeRouterVxnets", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddRouterStaticEntriesRequest struct {
	RouterStatic qingcloud.String
	EntriesNVal1 qingcloud.NumberedString
	EntriesNVal2 qingcloud.NumberedString
}
type AddRouterStaticEntriesResponse struct {
	RouterStaticEntries []string `json:"router_static_entries"`
	qingcloud.CommonResponse
}

// AddRouterStaticEntries
// 增加一条路由器规则条目，比如 PPTP 的账户信息或是隧道规则的网络地址。 注意：在增加路由器规则条目后，你需要执行 UpdateRouters 才能使之生效。
func (c *ROUTER) AddRouterStaticEntries(params AddRouterStaticEntriesRequest) (AddRouterStaticEntriesResponse, error) {
	var result AddRouterStaticEntriesResponse
	err := c.Get("AddRouterStaticEntries", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteRouterStaticEntriesReqeust struct {
	RouterStaticEntriesN qingcloud.NumberedString
}
type DeleteRouterStaticEntriesResponse struct {
	RouterStaticEntriesN qingcloud.NumberedString
	qingcloud.CommonResponse
}

// DeleteRouterStaticEntries
// 删除一条或多条路由器规则条目。在删除路由器规则条目之后，你需要执行 UpdateRouters 来使规则删除生效。
func (c *ROUTER) DeleteRouterStaticEntries(params DeleteRouterStaticEntriesReqeust) (DeleteRouterStaticEntriesResponse, error) {
	var result DeleteRouterStaticEntriesResponse
	err := c.Get("DeleteRouterStaticEntries", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyRouterStaticEntryAttributesReqeust struct {
	RouterStaticEntry     qingcloud.String
	RouterStaticEntryName qingcloud.String
	Val1                  qingcloud.String
	Val2                  qingcloud.String
}
type ModifyRouterStaticEntryAttributesResponse struct {
	RouterStaticEntry string `json:"router_static_entry"`
}

// ModifyRouterStaticEntryAttributes
// 修改路由器规则中的某条条目属性。修改后，需要执行 UpdateRouters 来使规则生效。
func (c *ROUTER) ModifyRouterStaticEntryAttributes(params ModifyRouterStaticEntryAttributesReqeust) (ModifyRouterStaticEntryAttributesResponse, error) {
	var result ModifyRouterStaticEntryAttributesResponse
	err := c.Get("ModifyRouterStaticEntryAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeRouterStaticEntriesRequest struct {
	RouterStaticEntryID qingcloud.String
	RouterStatic        qingcloud.String
	Offset              qingcloud.Integer
	Limit               qingcloud.Integer
}
type DescribeRouterStaticEntriesResponse struct {
	TotalCount           int                 `json:"total_count"`
	RouterStaticEntrySet []RouterStaticEntry `json:"router_static_entry_set"`
	qingcloud.CommonResponse
}

// DescribeRouterStaticEntries
// 获取路由器规则的条目。
// 可根据路由器规则ID作为过滤条件，来获取路由器规则中的条目列表。
func (c *ROUTER) DescribeRouterStaticEntries(params DescribeRouterStaticEntriesRequest) (DescribeRouterStaticEntriesResponse, error) {
	var result DescribeRouterStaticEntriesResponse
	err := c.Get("DescribeRouterStaticEntries", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
