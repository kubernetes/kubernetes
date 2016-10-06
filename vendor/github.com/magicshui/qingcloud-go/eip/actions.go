package eip

import (
	"github.com/magicshui/qingcloud-go"
)

type EIP struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *EIP {
	var a = &EIP{
		Client: clt,
	}
	return a
}

type DescribeEipsRequest struct {
	EipsN      qingcloud.NumberedString
	InstanceId qingcloud.NumberedString

	StatusN    qingcloud.NumberedString
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeEipsResponse struct {
	EipSet     []Eip `json:"eip_set"`
	TotalCount int   `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeEips
// 获取一个或多个公网IP
// 可根据公网IP的ID，状态，名称，分配的主机ID作过滤条件，来获取公网IP列表。 如果不指定任何过滤条件，默认返回你所拥有的所有公网IP。 如果指定不存在的公网IP，或非法状态值，则会返回错误信息。
func (c *EIP) DescribeEips(params DescribeEipsRequest) (DescribeEipsResponse, error) {
	var result DescribeEipsResponse
	err := c.Get("DescribeEips", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AllocateEipsRequest struct {
	Bandwidth   qingcloud.Integer
	BillingMode qingcloud.String
	EipName     qingcloud.String
	Count       qingcloud.Integer
	NeedIcp     qingcloud.Integer
}
type AllocateEipsResponse struct {
	Eips []string `json:"eips"`
}

// AllocateEips 从IP池中分配一个公网IP，分配时可指定带宽、数量、IP组、名称及是否需要备案。
// 分配后的公网IP可跟主机或路由器绑定。
func (c *EIP) AllocateEips(params AllocateEipsRequest) (AllocateEipsResponse, error) {
	var result AllocateEipsResponse
	err := c.Get("AllocateEips", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ReleaseEipsRequest struct {
	EipsN qingcloud.NumberedString
}
type ReleaseEipsResponse qingcloud.CommonResponse

// ReleaseEips
// 将一个或多个公网IP释放回IP池，同时相关IP的计费也会停止。
// 如果公网IP正与其他资源绑定，则需要先解绑，再释放， 保证被释放的IP处于“可用”（ available ）状态。
func (c *EIP) ReleaseEips(params ReleaseEipsRequest) (ReleaseEipsResponse, error) {
	var result ReleaseEipsResponse
	err := c.Get("ReleaseEips", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AssociateEipRequest struct {
	Eip      qingcloud.String
	Instance qingcloud.String
}
type AssociateEipResponse qingcloud.CommonResponse

// AssociateEip 将一个“可用”（ available ）状态的公网IP绑定到主机， 绑定后的主机才具有访问外网的能力。
// 不能对已绑定公网IP的主机再次绑定，如果需要更改IP， 则要先解绑之前的IP，再绑定新的。
// 如果想将公网IP绑定到路由器，请参见 ModifyRouterAttributes
func (c *EIP) AssociateEip(params AssociateEipRequest) (AssociateEipResponse, error) {
	var result AssociateEipResponse
	err := c.Get("AssociateEip", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DissociateEipsRequest struct {
	EipsN qingcloud.NumberedString
}
type DissociateEipsResponse qingcloud.CommonResponse

// DissociateEips
// 将一个或多个“绑定中”（ associated ）状态的公网IP解绑， 解绑后会变回“可用”（ available ）状态。
func (c *EIP) DissociateEips(params DissociateEipsRequest) (DissociateEipsResponse, error) {
	var result DissociateEipsResponse
	err := c.Get("DissociateEips", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ChangeEipsBandwidthRequest struct {
	Bandwidth qingcloud.Integer
	EipsN     qingcloud.NumberedString
}
type ChangeEipsBandwidthResponse qingcloud.CommonResponse

// ChangeEipsBandwidth 动态改变一个或多个公网IP的带宽，改变后计费系统会同步更新。
// 无论公网IP当前处于“可用”（ available ）还是“绑定中” （ associated ）状态，都可以随时改变带宽，并实时生效。
func (c *EIP) ChangeEipsBandwidth(params ChangeEipsBandwidthRequest) (ChangeEipsBandwidthResponse, error) {
	var result ChangeEipsBandwidthResponse
	err := c.Get("ChangeEipsBandwidth", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ChangeEipsBillingModeRequest struct {
	EipsN       qingcloud.NumberedString
	BillingMode qingcloud.String
}
type ChangeEipsBillingModeResponse qingcloud.CommonResponse

// ChangeEipsBillingMode
// 动态改变一个或多个公网IP的计费模式，改变后计费系统会及时更新。
func (c *EIP) ChangeEipsBillingMode(params ChangeEipsBillingModeRequest) (ChangeEipsBillingModeResponse, error) {
	var result ChangeEipsBillingModeResponse
	err := c.Get("ChangeEipsBillingMode", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyEipAttributesRequest struct {
	Eip         qingcloud.String
	EipName     qingcloud.String
	Description qingcloud.String
}
type ModifyEipAttributesResponse qingcloud.CommonResponse

// ModifyEipAttributes
// 修改一个公网IP的名称和描述。
// 修改时不受公网IP状态限制。
// 一次只能修改一个公网IP。
func (c *EIP) ModifyEipAttributes(params ModifyEipAttributesRequest) (ModifyEipAttributesResponse, error) {
	var result ModifyEipAttributesResponse
	err := c.Get("ModifyEipAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
