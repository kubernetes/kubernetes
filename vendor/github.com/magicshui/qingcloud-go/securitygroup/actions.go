package securitygroup

import (
	"github.com/magicshui/qingcloud-go"
)

type SECURITYGROUP struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *SECURITYGROUP {
	return &SECURITYGROUP{
		Client: clt,
	}
}

type DescribeSecurityGroupsRequest struct {
	SecurityGroupsN qingcloud.NumberedString

	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeSecurityGroupsResponse struct {
	SecurityGroupSet []SecurityGroup `json:"security_group_set"`
	TotalCount       int             `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeSecurityGroups
// 获取一个或多个防火墙信息。
// 可根据防火墙ID，名称作过滤条件，来获取防火墙列表。 如果不指定任何过滤条件，默认返回你所拥有的所有防火墙。 如果指定不存在的防火墙ID，或非法状态值，则会返回错误信息。
func (c *SECURITYGROUP) DescribeSecurityGroups(params DescribeSecurityGroupsRequest) (DescribeSecurityGroupsResponse, error) {
	var result DescribeSecurityGroupsResponse
	params.Verbose.Enum(0, 1)
	err := c.Get("DescribeSecurityGroups", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateSecurityGroupRequest struct {
	SecurityGroupName qingcloud.String
}
type CreateSecurityGroupResponse struct {
	qingcloud.CommonResponse
	SecurityGroupId string `json:"security_group_id"`
}

// CreateSecurityGroup
// 创建防火墙。防火墙可用于保障主机和路由器的网络安全。
// 刚创建的防火墙不包含任何规则，即任何端口都是封闭的， 需要建立规则以打开相应的端口。
// 青云为每个用户提供了一个缺省防火墙，为了方便用户使用， 缺省防火墙默认打开了下行 icmp 协议和 tcp 22 端口。
func (c *SECURITYGROUP) CreateSecurityGroup(params CreateSecurityGroupRequest) (CreateSecurityGroupResponse, error) {
	var result CreateSecurityGroupResponse
	err := c.Get("CreateSecurityGroup", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteSecurityGroupsRequest struct {
	SecurityGroupsN qingcloud.NumberedString
}
type DeleteSecurityGroupsResponse struct {
	SecurityGroups []string `json:"security_groups"`
	qingcloud.CommonResponse
}

// DeleteSecurityGroups
// 删除一个或多个防火墙。
// 防火墙须在没有资源（主机或路由器）使用的情况下才能被删除。 已加载规则到资源的防火墙，需先将相关资源从防火墙移出后才能被删除。
// 要删除的防火墙已加载规则到主机，则需要先调用 ApplySecurityGroup 将其他防火墙的规则应用到对应主机，之后才能被删除。
// 要删除的防火墙已加载规则到路由器，则需要先调用 ModifyRouterAttributes 并 UpdateRouters 将其他防火墙的规则应用到对应路由器，之后才能被删除。
// 青云系统提供的缺省防火墙不能被删除。
func (c *SECURITYGROUP) DeleteSecurityGroups(params DeleteSecurityGroupsRequest) (DeleteSecurityGroupsResponse, error) {
	var result DeleteSecurityGroupsResponse
	err := c.Get("DeleteSecurityGroups", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ApplySecurityGroupRequest struct {
	SecurityGroup qingcloud.String
	InstancesN    qingcloud.NumberedString
}
type ApplySecurityGroupResponse qingcloud.CommonResponse

// ApplySecurityGroup
// 应用防火墙规则。当防火墙的规则发生改变后，新规则不会即刻生效 （可通过 is_applied 属性分辨），需要调用 ApplySecurityGroup 之后才生效。
// 防火墙规则可通过 AddSecurityGroupRules, DeleteSecurityGroupRules, ModifySecurityGroupRuleAttributes 修改。
// 如果请求参数中传递了 instances.n ，则表示将此防火墙的规则应用到对应的主机。 如果不传此参数，则会将最新规则更新到所有已应用此防火墙的主机。
func (c *SECURITYGROUP) ApplySecurityGroup(params ApplySecurityGroupRequest) (ApplySecurityGroupResponse, error) {
	var result ApplySecurityGroupResponse
	err := c.Get("ApplySecurityGroup", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifySecurityGroupAttributesRequest struct {
	SecurityGroup     qingcloud.String
	SecurityGroupName qingcloud.String
	Description       qingcloud.String
}
type ModifySecurityGroupAttributesResponse struct {
	qingcloud.CommonResponse
	SecurityGroupId string `json:"security_group_id"`
}

// ModifySecurityGroupAttributes
// 修改防火墙的名称和描述。
// 一次只能修改一个防火墙。
func (c *SECURITYGROUP) ModifySecurityGroupAttributes(params ModifySecurityGroupAttributesRequest) (ModifySecurityGroupAttributesResponse, error) {
	var result ModifySecurityGroupAttributesResponse
	err := c.Get("ModifySecurityGroupAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeSecurityGroupRulesRequest struct {
	SecurityGroup       qingcloud.String
	SecurityGroupRulesN qingcloud.NumberedString
	Direction           qingcloud.Integer
	Offset              qingcloud.Integer
	Limit               qingcloud.Integer
}
type DescribeSecurityGroupRulesResponse struct {
	SecurityGroupRuleSet []SecurityGroupRule `json:"security_group_rule_set"`
	TotalCount           int                 `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeSecurityGroupRules
// 获取某个防火墙的规则信息。
// 可根据防火墙ID，上行/下行，防火墙规则ID 作过滤条件，获取防火墙规则列表。 如果不指定任何过滤条件，默认返回你所拥有的所有防火墙的所有规则。
func (c *SECURITYGROUP) DescribeSecurityGroupRules(params DescribeSecurityGroupRulesRequest) (DescribeSecurityGroupRulesResponse, error) {
	var result DescribeSecurityGroupRulesResponse
	err := c.Get("DescribeSecurityGroupRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddSecurityGroupRulesRequest struct {
	SecurityGroup qingcloud.String

	RulesNProtocol              qingcloud.NumberedString
	RulesNPriority              qingcloud.NumberedInteger
	RulesNSecurityGroupRuleName qingcloud.NumberedString
	RulesNAction                qingcloud.NumberedString
	RulesNDirection             qingcloud.NumberedInteger

	RulesNVal1 qingcloud.NumberedString
	RulesNVal2 qingcloud.NumberedString
	RulesNVal3 qingcloud.NumberedString
}
type AddSecurityGroupRulesResponse struct {
	SecurityGroupRules []string `json:"security_group_rules"`
	qingcloud.CommonResponse
}

// AddSecurityGroupRules
// 给防火墙添加规则。每条规则包括的属性为：
// protocol：协议
// priority：优先级，由高到低为 0 - 100
// security_group_rule_name：规则名称
// action：操作，分为 accept 接受 和 drop 拒绝
// direction：方向，0 表示下行，1 表示上行。
// val1：如果协议为 tcp 或 udp，此值表示起始端口。 如果协议为 icmp，此值表示 ICMP 类型。 具体类型可参见 ICMP 类型及代码
// val2：如果协议为 tcp 或 udp，此值表示结束端口。 如果协议为 icmp，此值表示 ICMP 代码。 具体代码可参见 ICMP 类型及代码
// val3：源IP
func (c *SECURITYGROUP) AddSecurityGroupRules(params AddSecurityGroupRulesRequest) (AddSecurityGroupRulesResponse, error) {
	var result AddSecurityGroupRulesResponse
	err := c.Get("AddSecurityGroupRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteSecurityGroupRulesRequest struct {
	SecurityGroupRulesN qingcloud.NumberedString
}
type DeleteSecurityGroupRulesResponse struct {
	SecurityGroupRules []string `json:"security_group_rules"`
	qingcloud.CommonResponse
}

// DeleteSecurityGroupRules
// 删除防火墙规则。
func (c *SECURITYGROUP) DeleteSecurityGroupRules(params DeleteSecurityGroupRulesRequest) (DeleteSecurityGroupRulesResponse, error) {
	var result DeleteSecurityGroupRulesResponse
	err := c.Get("DeleteSecurityGroupRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifySecurityGroupRuleAttributesRequest struct {
	SecurityGroupRule     qingcloud.String
	SecurityGroupRuleName qingcloud.String
	Priority              qingcloud.Integer
	RuleAction            qingcloud.String
	Direction             qingcloud.Integer
	Protocol              qingcloud.String
	Val1                  qingcloud.Integer
	Val2                  qingcloud.Integer
	Val3                  qingcloud.Integer
}
type ModifySecurityGroupRuleAttributesResponse struct {
	qingcloud.CommonResponse
	SecurityGroupRuleId string `json:"security_group_rule_id"`
}

// ModifySecurityGroupRuleAttributes
// 修改防火墙规则的优先级。
func (c *SECURITYGROUP) ModifySecurityGroupRuleAttributes(params ModifySecurityGroupRuleAttributesRequest) (ModifySecurityGroupRuleAttributesResponse, error) {
	var result ModifySecurityGroupRuleAttributesResponse
	err := c.Get("ModifySecurityGroupRuleAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateSecurityGroupSnapshotRequest struct {
	SecurityGroup qingcloud.String
	Name          qingcloud.String
}
type CreateSecurityGroupSnapshotResponse struct {
	SecurityGroupID         string `json:"security_group_id"`
	SecurityGroupSnapshotID string `json:"security_group_snapshot_id"`
	qingcloud.CommonResponse
}

// CreateSecurityGroupSnapshot
// 根据当前的防火墙规则创建一个备份, 用于随时回滚之前的防火墙规则。
func (c *SECURITYGROUP) CreateSecurityGroupSnapshot(params CreateSecurityGroupSnapshotRequest) (CreateSecurityGroupSnapshotResponse, error) {
	var result CreateSecurityGroupSnapshotResponse
	err := c.Get("CreateSecurityGroupSnapshot", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeSecurityGroupSnapshotsRequest struct {
	SecurityGroup           qingcloud.String
	SecurityGroupSnapshotsN qingcloud.NumberedString

	Offset qingcloud.Integer
	Limit  qingcloud.Integer
}
type DescribeSecurityGroupSnapshotsResponse struct {
	qingcloud.CommonResponse
	SecurityGroupSnapshotSet []SecurityGroupSnapshot `json:"security_group_snapshot_set"`
	TotalCount               int                     `json:"total_count"`
}

// DescribeSecurityGroupSnapshots
// 获取某个防火墙的备份信息。
func (c *SECURITYGROUP) DescribeSecurityGroupSnapshots(params DescribeSecurityGroupSnapshotsRequest) (DescribeSecurityGroupSnapshotsResponse, error) {
	var result DescribeSecurityGroupSnapshotsResponse
	err := c.Get("DescribeSecurityGroupSnapshots", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteSecurityGroupSnapshotsRequest struct {
	SecurityGroupSnapshotsN qingcloud.NumberedString
}
type DeleteSecurityGroupSnapshotsResponse struct {
	SecurityGroupSnapshots []string `json:"security_group_snapshots"`

	qingcloud.CommonResponse
}

// DeleteSecurityGroupSnapshots
// 删除防火墙备份。
func (c *SECURITYGROUP) DeleteSecurityGroupSnapshots(params DeleteSecurityGroupSnapshotsRequest) (DeleteSecurityGroupSnapshotsResponse, error) {
	var result DeleteSecurityGroupSnapshotsResponse
	err := c.Get("DeleteSecurityGroupSnapshots", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type RollbackSecurityGroupRequest struct {
	SecurityGroup         qingcloud.String
	SecurityGroupSnapshot qingcloud.String
}
type RollbackSecurityGroupResponse struct {
	SecurityGroupId         string `json:"security_group_id"`
	SecurityGroupSnapshotId string `json:"security_group_snapshot_id"`
	qingcloud.CommonResponse
}

// RollbackSecurityGroup
// 使用防火墙备份回滚。
func (c *SECURITYGROUP) RollbackSecurityGroup(params RollbackSecurityGroupRequest) (RollbackSecurityGroupResponse, error) {
	var result RollbackSecurityGroupResponse
	err := c.Get("RollbackSecurityGroup", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
