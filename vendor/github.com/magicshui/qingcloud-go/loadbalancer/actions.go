package loadbalancer

import (
	"github.com/magicshui/qingcloud-go"
)

type LOADBALANCER struct {
	*qingcloud.Client
}

func NewClient(clt *qingcloud.Client) *LOADBALANCER {
	return &LOADBALANCER{
		Client: clt,
	}
}

type CreateLoadBalancerRequest struct {
	EipsN            qingcloud.NumberedString
	Vxnet            qingcloud.String
	PrivateIp        qingcloud.String
	LoadbalancerType qingcloud.Integer
	LoadbalancerName qingcloud.String
	SecurityGroup    qingcloud.String
}

type CreateLoadBalancerResponse struct {
	qingcloud.CommonResponse
	LoadbalancerId string `json:"loadbalancer_id"`
}

// CreateLoadBalancer
// 创建一个负载均衡器。创建时需指定与此负载均衡器关联的公网IP，可支持多个IP。eips 和 vxnet 必选一个.
func (c *LOADBALANCER) CreateLoadBalancer(params CreateLoadBalancerRequest) (CreateLoadBalancerResponse, error) {
	var result CreateLoadBalancerResponse
	err := c.Get("CreateLoadBalancer", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeLoadBalancersRequest struct {
	LoadbalancersN qingcloud.NumberedString

	StatusN    qingcloud.NumberedString
	SearchWord qingcloud.String
	TagsN      qingcloud.NumberedString
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeLoadBalancersResponse struct {
	LoadbalancerSet []Loadbalancer `json:"loadbalancer_set"`
	TotalCount      int            `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeLoadBalancers
// 获取一个或多个负载均衡器。
// 可根据负载均衡器ID，状态，负载均衡器名称作过滤条件，来获取负载均衡器列表。 如果不指定任何过滤条件，默认返回你的所有负载均衡器。
func (c *LOADBALANCER) DescribeLoadBalancers(params DescribeLoadBalancersRequest) (DescribeLoadBalancersResponse, error) {
	var result DescribeLoadBalancersResponse
	err := c.Get("DescribeLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteLoadBalancersRequest struct {
	LoadbalancersN qingcloud.NumberedString
}
type DeleteLoadBalancersResponse qingcloud.CommonResponse

// DeleteLoadBalancers
// 删除一台或多台负载均衡器。
// 销毁资源的前提，是此资源已建立租用信息（租用信息是在资源创建成功后， 几秒钟内系统自动建立的）。所以正在创建的资源（状态是 pending ）， 以及刚刚创建但还没有建立租用信息的，是不能被销毁的。
// 删除负载均衡器后，与其关联的公网IP会自动解绑，变为“可用”状态。
func (c *LOADBALANCER) DeleteLoadBalancers(params DeleteLoadBalancersRequest) (DeleteLoadBalancersResponse, error) {
	var result DeleteLoadBalancersResponse
	err := c.Get("DeleteLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyLoadBalancerAttributesRequest struct {
	Loadbalancer     qingcloud.String
	LoadbalancerName qingcloud.String
	SecurityGroup    qingcloud.String
	Description      qingcloud.String
	PrivateIp        qingcloud.String
}
type ModifyLoadBalancerAttributesResponse qingcloud.CommonResponse

// ModifyLoadBalancerAttributes
// 修改一台负载均衡器的名称和描述。
func (c *LOADBALANCER) ModifyLoadBalancerAttributes(params ModifyLoadBalancerAttributesRequest) (ModifyLoadBalancerAttributesResponse, error) {
	var result ModifyLoadBalancerAttributesResponse
	err := c.Get("ModifyLoadBalancerAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StartLoadBalancersRequest struct {
	LoadbalancersN qingcloud.NumberedString
}
type StartLoadBalancersResponse qingcloud.CommonResponse

// StartLoadBalancers
// 启动一台或多台负载均衡器。
func (c *LOADBALANCER) StartLoadBalancers(params StartLoadBalancersRequest) (StartLoadBalancersResponse, error) {
	var result StartLoadBalancersResponse
	err := c.Get("StartLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type StopLoadBalancersRequest struct {
	LoadbalancersN qingcloud.NumberedString
}
type StopLoadBalancersResponse qingcloud.CommonResponse

// StopLoadBalancers
// 关闭一台或多台负载均衡器。
func (c *LOADBALANCER) StopLoadBalancers(params StopLoadBalancersRequest) (StopLoadBalancersResponse, error) {
	var result StopLoadBalancersResponse
	err := c.Get("StopLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type UpdateLoadBalancersRequest struct {
	LoadbalancersN qingcloud.NumberedString
}
type UpdateLoadBalancersResponse qingcloud.CommonResponse

// UpdateLoadBalancers
// 更新一台或多台负载均衡器的配置。在每次对负载均衡器的配置进行变更，例如”增加”或”删除”监听器或后端服务时， 需要执行该操作使配置更新生效。
func (c *LOADBALANCER) UpdateLoadBalancers(params UpdateLoadBalancersRequest) (UpdateLoadBalancersResponse, error) {
	var result UpdateLoadBalancersResponse
	err := c.Get("UpdateLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ResizeLoadBalancersRequest struct {
	LoadbalancersN   qingcloud.NumberedString
	LoadbalancerType qingcloud.Integer
}
type ResizeLoadBalancersResponse qingcloud.CommonResponse

// ResizeLoadBalancers
// 修改负载均衡器最大连接数配置。负载均衡器状态必须是关闭的 stopped ，不然会返回错误。
func (c *LOADBALANCER) ResizeLoadBalancers(params ResizeLoadBalancersRequest) (ResizeLoadBalancersResponse, error) {
	var result ResizeLoadBalancersResponse
	err := c.Get("ResizeLoadBalancers", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AssociateEipsToLoadBalancerRequest struct {
	EipsN        qingcloud.NumberedString
	Loadbalancer qingcloud.String
}
type AssociateEipsToLoadBalancerResponse qingcloud.CommonResponse

// AssociateEipsToLoadBalancer
// 将一个或多个“可用”（ available ）状态的公网IP绑定到负载均衡器。
func (c *LOADBALANCER) AssociateEipsToLoadBalancer(params AssociateEipsToLoadBalancerRequest) (AssociateEipsToLoadBalancerResponse, error) {
	var result AssociateEipsToLoadBalancerResponse
	err := c.Get("AssociateEipsToLoadBalancer", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DissociateEipsFromLoadBalancerRequest struct {
	EipsN        qingcloud.NumberedString
	Loadbalancer qingcloud.String
}
type DissociateEipsFromLoadBalancerResponse qingcloud.CommonResponse

// DissociateEipsFromLoadBalancer
// 将一个或多个“绑定中”（ associated ）状态的公网IP从负载均衡器中解绑， 解绑后会变回“可用”（ available ）状态。
func (c *LOADBALANCER) DissociateEipsFromLoadBalancer(params DissociateEipsFromLoadBalancerRequest) (DissociateEipsFromLoadBalancerResponse, error) {
	var result DissociateEipsFromLoadBalancerResponse
	err := c.Get("DissociateEipsFromLoadBalancer", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddLoadBalancerListenersRequest struct {
	Loadbalancer                       qingcloud.String
	ListenersNListenerPort             qingcloud.NumberedInteger
	ListenersNListenerProtocol         qingcloud.NumberedString
	ListenersNServerCertificateId      qingcloud.NumberedString
	ListenersNBackendProtocol          qingcloud.NumberedString
	ListenersNLoadbalancerListenerName qingcloud.NumberedString
	ListenersNBalanceMode              qingcloud.NumberedString
	ListenersNSessionSticky            qingcloud.NumberedString
	ListenersNForwardfor               qingcloud.NumberedInteger
	ListenersNHealthyCheckMethod       qingcloud.NumberedString
	ListenersNHealthyCheckOption       qingcloud.NumberedString
	ListenersNListenerOption           qingcloud.NumberedInteger
}
type AddLoadBalancerListenersResponse struct {
	LoadbalancerListeners []string `json:"loadbalancer_listeners"`
	qingcloud.CommonResponse
}

// AddLoadBalancerListeners
// 给负载均衡器添加一个或多个监听器。
func (c *LOADBALANCER) AddLoadBalancerListeners(params AddLoadBalancerListenersRequest) (AddLoadBalancerListenersResponse, error) {
	var result AddLoadBalancerListenersResponse
	err := c.Get("AddLoadBalancerListeners", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeLoadBalancerListenersRequest struct {
	LoadbalancerListenersN qingcloud.NumberedString
	Loadbalancer           qingcloud.String
	Verbose                qingcloud.Integer
	Offset                 qingcloud.Integer
	Limit                  qingcloud.Integer
}
type DescribeLoadBalancerListenersResponse struct {
	LoadbalancerListenerSet []LoadbalancerListener `json:"loadbalancer_listener_set"`
	TotalCount              int                    `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeLoadBalancerListeners
// 获取负载均衡器的监听器。
// 可根据负载均衡器ID，监听器ID 作为过滤条件获取监听器列表。 如果不指定任何过滤条件，默认返回你拥有的负载均衡器下面的所有监听器。
func (c *LOADBALANCER) DescribeLoadBalancerListeners(params DescribeLoadBalancerListenersRequest) (DescribeLoadBalancerListenersResponse, error) {
	var result DescribeLoadBalancerListenersResponse
	err := c.Get("DescribeLoadBalancerListeners", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteLoadBalancerListenersRequest struct {
	LoadbalancerListenersN qingcloud.NumberedString
}
type DeleteLoadBalancerListenersResponse struct {
	LoadbalancerListeners []string `json:"loadbalancer_listeners"`
	qingcloud.CommonResponse
}

// DeleteLoadBalancerListeners
// 删除一个或多个负载均衡器监听器。
func (c *LOADBALANCER) DeleteLoadBalancerListeners(params DeleteLoadBalancerListenersRequest) (DeleteLoadBalancerListenersResponse, error) {
	var result DeleteLoadBalancerListenersResponse
	err := c.Get("DeleteLoadBalancerListeners", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyLoadBalancerListenerAttributesRequest struct {
	LoadbalancerListener     qingcloud.String
	LoadbalancerListenerName qingcloud.String
	ServerCertificateId      qingcloud.String
	SessionSticky            qingcloud.String
	Forwardfor               qingcloud.Integer
	HealthyCheckMethod       qingcloud.String
	HealthyCheckOption       qingcloud.String
	ListenersNListenerOption qingcloud.NumberedInteger
}
type ModifyLoadBalancerListenerAttributesResponse qingcloud.CommonResponse

// ModifyLoadBalancerListenerAttributes
// 修改负载均衡器监听器的属性。
func (c *LOADBALANCER) ModifyLoadBalancerListenerAttributes(params ModifyLoadBalancerListenerAttributesRequest) (ModifyLoadBalancerListenerAttributesResponse, error) {
	var result ModifyLoadBalancerListenerAttributesResponse
	err := c.Get("ModifyLoadBalancerListenerAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddLoadBalancerBackendsRequest struct {
	LoadbalancerListener             qingcloud.String
	BackendsNResourceId              qingcloud.NumberedString
	BackendsNLoadbalancerBackendName qingcloud.NumberedString
	BackendsNLoadbalancerPolicyId    qingcloud.NumberedString
	BackendsNPort                    qingcloud.NumberedInteger
	BackendsNWeight                  qingcloud.NumberedInteger
}
type AddLoadBalancerBackendsResponse struct {
	LoadbalancerBackends []string `json:"loadbalancer_backends"`
	qingcloud.CommonResponse
}

// AddLoadBalancerBackends
// 给负载均衡器的监听器添加后端服务。后端服务资源可以是主机或路由器。
func (c *LOADBALANCER) AddLoadBalancerBackends(params AddLoadBalancerBackendsRequest) (AddLoadBalancerBackendsResponse, error) {
	var result AddLoadBalancerBackendsResponse
	err := c.Get("AddLoadBalancerBackends", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeLoadBalancerBackendsRequest struct {
	LoadbalancerBackendsN qingcloud.NumberedString
	LoadbalancerLister    qingcloud.NumberedString
	Loadbalancere         qingcloud.String
	Verbose               qingcloud.Integer
	Offset                qingcloud.Integer
	Limit                 qingcloud.Integer
}
type DescribeLoadBalancerBackendsResponse struct {
	LoadbalancerBackendSet []LoadbalancerBackend `json:"loadbalancer_backend_set"`
	TotalCount             int                   `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeLoadBalancerBackends
// 获取负载均衡器后端服务列表。
// 可根据负载均衡器ID，监听器ID 或 后端服务ID 作为过滤条件获取后端服务列表。 如果不指定任何过滤条件，默认返回你拥有的负载均衡器下面监听器的所有后端服务。
func (c *LOADBALANCER) DescribeLoadBalancerBackends(params DescribeLoadBalancerBackendsRequest) (DescribeLoadBalancerBackendsResponse, error) {
	var result DescribeLoadBalancerBackendsResponse
	err := c.Get("DescribeLoadBalancerBackends", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteLoadBalancerBackendsRequest struct {
	LoadbalancerBackendsN qingcloud.NumberedString
}
type DeleteLoadBalancerBackendsResponse struct {
	LoadbalancerBackends []string `json:"loadbalancer_backends"`
	qingcloud.CommonResponse
}

// DeleteLoadBalancerBackends
// 删除一个或多个负载均衡器后端服务。
func (c *LOADBALANCER) DeleteLoadBalancerBackends(params DeleteLoadBalancerBackendsRequest) (DeleteLoadBalancerBackendsResponse, error) {
	var result DeleteLoadBalancerBackendsResponse
	err := c.Get("DeleteLoadBalancerBackends", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyLoadBalancerBackendAttributesRequest struct {
	LoadbalancerBackend  qingcloud.String
	Port                 qingcloud.String
	Weight               qingcloud.String
	Disabled             qingcloud.Integer
	LoadbalancerPolicyId qingcloud.String
}
type ModifyLoadBalancerBackendAttributesResponse qingcloud.CommonResponse

// ModifyLoadBalancerBackendAttributes
// 修改负载均衡器后端服务的属性。
func (c *LOADBALANCER) ModifyLoadBalancerBackendAttributes(params ModifyLoadBalancerBackendAttributesRequest) (ModifyLoadBalancerBackendAttributesResponse, error) {
	var result ModifyLoadBalancerBackendAttributesResponse
	err := c.Get("ModifyLoadBalancerBackendAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateLoadBalancerPolicyRequest struct {
	Operator qingcloud.String
}
type CreateLoadBalancerPolicyResponse struct {
	// TODO: 文档又错误
	LoadbalancerPolicyId string `json:"loadbalancer_policy_id"`
	qingcloud.CommonResponse
}

// CreateLoadBalancerPolicy
// 创建一个负载均衡器转发策略，可通过自定义转发策略来进行更高级的转发控制。 每个策略可包括多条规则，规则间支持『与』和『或』关系。
func (c *LOADBALANCER) CreateLoadBalancerPolicy(params CreateLoadBalancerPolicyRequest) (CreateLoadBalancerPolicyResponse, error) {
	var result CreateLoadBalancerPolicyResponse
	err := c.Get("CreateLoadBalancerPolicy", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeLoadBalancerPoliciesRequest struct {
	LoadbalancerPoliciesN qingcloud.NumberedString
	Verbose               qingcloud.Integer
	Offset                qingcloud.Integer
	Limit                 qingcloud.Integer
}
type DescribeLoadBalancerPoliciesResponse struct {
	qingcloud.CommonResponse
	TotalCount            int                  `json:"total_count"`
	LoadbalancerPolicySet []LoadbalancerPolicy `json:"loadbalancer_policy_set"`
}

// DescribeLoadBalancerPolicies
// 获取负载均衡器的转发策略。
// 可根据负载转发策略ID，策略名称获取转发策略列表。 如果不指定过滤条件，默认返回你拥有的所有负载转发策略。
func (c *LOADBALANCER) DescribeLoadBalancerPolicies(params DescribeLoadBalancerPoliciesRequest) (DescribeLoadBalancerPoliciesResponse, error) {
	var result DescribeLoadBalancerPoliciesResponse
	err := c.Get("DescribeLoadBalancerPolicies", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyLoadBalancerPolicyAttributesRequest struct {
	LoadbalancerPolicy     qingcloud.String
	LoadbalancerPolicyName qingcloud.String
	Operator               qingcloud.String
}
type ModifyLoadBalancerPolicyAttributesResponse qingcloud.CommonResponse

// ModifyLoadBalancerPolicyAttributes
// 修改负载均衡器转发策略的属性。
func (c *LOADBALANCER) ModifyLoadBalancerPolicyAttributes(params ModifyLoadBalancerPolicyAttributesRequest) (ModifyLoadBalancerPolicyAttributesResponse, error) {
	var result ModifyLoadBalancerPolicyAttributesResponse
	err := c.Get("ModifyLoadBalancerPolicyAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ApplyLoadBalancerPolicyRequest struct {
	LoadbalancerPolicy qingcloud.String
}
type ApplyLoadBalancerPolicyResponse qingcloud.CommonResponse

// ApplyLoadBalancerPolicy
// 更新负载转发策略。在每次对转发策略、转发规则进行修改后， 都需要主动『更新修改』使改动生效，即调用此 API 。
func (c *LOADBALANCER) ApplyLoadBalancerPolicy(params ApplyLoadBalancerPolicyRequest) (ApplyLoadBalancerPolicyResponse, error) {
	var result ApplyLoadBalancerPolicyResponse
	err := c.Get("ApplyLoadBalancerPolicy", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteLoadBalancerPoliciesRequest struct {
	LoadbalancerPoliciesN qingcloud.NumberedString
}
type DeleteLoadBalancerPoliciesResponse struct {
	qingcloud.CommonResponse
	LoadbalancerPolicies []string `json:"loadbalancer_policies"`
}

// DeleteLoadBalancerPolicies
// 删除一个或多个转发策略。
func (c *LOADBALANCER) DeleteLoadBalancerPolicies(params DeleteLoadBalancerPoliciesRequest) (DeleteLoadBalancerPoliciesResponse, error) {
	var result DeleteLoadBalancerPoliciesResponse
	err := c.Get("DeleteLoadBalancerPolicies", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type AddLoadBalancerPolicyRulesRequest struct {
	LoadbalancerPolicy               qingcloud.String
	RulesNLoadbalancerPolicyRuleName qingcloud.NumberedString
	RulesNRuleType                   qingcloud.NumberedString
	RulesNVal                        qingcloud.NumberedString
}
type AddLoadBalancerPolicyRulesResponse struct {
	qingcloud.CommonResponse
	LoadbalancerPoliciyRules []string `json:"loadbalancer_policy_rules"`
}

// AddLoadBalancerPolicyRules
// 给转发策略添加多条规则。 注意：在添加之后，为了让新规则生效，你需要执行 ApplyLoadBalancerPolicy 指令。
func (c *LOADBALANCER) AddLoadBalancerPolicyRules(params AddLoadBalancerPolicyRulesRequest) (AddLoadBalancerPolicyRulesResponse, error) {
	var result AddLoadBalancerPolicyRulesResponse
	err := c.Get("AddLoadBalancerPolicyRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeLoadBalancerPolicyRulesRequest struct {
	LoadbalancerPolicyRulesN qingcloud.NumberedString
	LoadbalancerPolicy       qingcloud.String
	Offset                   qingcloud.Integer
	Limit                    qingcloud.Integer
}
type DescribeLoadBalancerPolicyRulesResponse struct {
	qingcloud.CommonResponse
	LoadbalancerPoliciyRule []LoadbalancerPoliciyRule `json:"loadbalancer_policy_rule_set"`
}

// DescribeLoadBalancerPolicyRules
// 获取转发策略规则列表。
// 可根据转发策略ID，转发策略规则ID 作为过滤条件获取转发策略规则列表。 如果不指定任何过滤条件，默认返回你拥有的所有转发策略规则。
func (c *LOADBALANCER) DescribeLoadBalancerPolicyRules(params DescribeLoadBalancerPolicyRulesRequest) (DescribeLoadBalancerPolicyRulesResponse, error) {
	var result DescribeLoadBalancerPolicyRulesResponse
	err := c.Get("DescribeLoadBalancerPolicyRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyLoadBalancerPolicyRuleAttributesRequest struct {
	LoadbalancerPolicyRule     qingcloud.String
	LoadbalancerPolicyRuleName qingcloud.String
	Val                        qingcloud.String
}
type ModifyLoadBalancerPolicyRuleAttributesResponse qingcloud.CommonResponse

// ModifyLoadBalancerPolicyRuleAttributes
// 修改负载均衡器转发策略规则的属性。 修改之后，为了让新规则生效，你需要执行 ApplyLoadBalancerPolicy 指令。
func (c *LOADBALANCER) ModifyLoadBalancerPolicyRuleAttributes(params ModifyLoadBalancerPolicyRuleAttributesRequest) (ModifyLoadBalancerPolicyRuleAttributesResponse, error) {
	var result ModifyLoadBalancerPolicyRuleAttributesResponse
	err := c.Get("ModifyLoadBalancerPolicyRuleAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteLoadBalancerPolicyRulesRequest struct {
	LoadbalancerPolicyRulesN qingcloud.NumberedString
}
type DeleteLoadBalancerPolicyRulesResponse struct {
	qingcloud.CommonResponse
	LoadbalancerPolicyRules []string `json:"loadbalancer_policy_rules"`
}

// DeleteLoadBalancerPolicyRules
// 删除一个或多个负载均衡器转发策略规则。 注意：在删除后，你需要执行 ApplyLoadBalancerPolicy 指令才会生效。
func (c *LOADBALANCER) DeleteLoadBalancerPolicyRules(params DeleteLoadBalancerPolicyRulesRequest) (DeleteLoadBalancerPolicyRulesResponse, error) {
	var result DeleteLoadBalancerPolicyRulesResponse
	err := c.Get("DeleteLoadBalancerPolicyRules", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type CreateServerCertificateRequest struct {
	ServerCertificateName qingcloud.String
	CertificateContent    qingcloud.String
	PrivateKey            qingcloud.String
}
type CreateServerCertificateResponse struct {
	qingcloud.CommonResponse
	ServerCertificateId string `json:"server_certificate_id"`
}

// CreateServerCertificate
// 此 API 需使用 POST 方法, 创建一个服务器证书。创建时需指定与此服务器证书关联的证书内容和私钥.
func (c *LOADBALANCER) CreateServerCertificate(params CreateServerCertificateRequest) (CreateServerCertificateResponse, error) {
	var result CreateServerCertificateResponse
	err := c.Post("CreateServerCertificate", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DescribeServerCertificatesRequest struct {
	ServerCertificates qingcloud.String

	SearchWord qingcloud.String
	Verbose    qingcloud.Integer
	Offset     qingcloud.Integer
	Limit      qingcloud.Integer
}
type DescribeServerCertificatesResponse struct {
	ServerCertificateSet []ServerCertificate `json:"server_certificate_set"`
	TotalCount           int                 `json:"total_count"`
	qingcloud.CommonResponse
}

// DescribeServerCertificates
// 获取一个或多个服务器证书。
// 可根据服务器证书ID，服务器证书名称作过滤条件，来获取服务器证书列表。 如果不指定任何过滤条件，默认返回你的所有服务器证书。
func (c *LOADBALANCER) DescribeServerCertificates(params DescribeServerCertificatesRequest) (DescribeServerCertificatesResponse, error) {
	var result DescribeServerCertificatesResponse
	err := c.Get("DescribeServerCertificates", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type ModifyServerCertificateAttributesRequest struct {
	ServerCertificate     qingcloud.String
	ServerCertificateName qingcloud.String
	Description           qingcloud.String
}
type ModifyServerCertificateAttributesResponse qingcloud.CommonResponse

// ModifyServerCertificateAttributes
// 修改一个服务器证书的名称和描述。
func (c *LOADBALANCER) ModifyServerCertificateAttributes(params ModifyServerCertificateAttributesRequest) (ModifyServerCertificateAttributesResponse, error) {
	var result ModifyServerCertificateAttributesResponse
	err := c.Get("ModifyServerCertificateAttributes", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}

type DeleteServerCertificatesRequest struct {
	ServerCertificatesN qingcloud.NumberedString
}
type DeleteServerCertificatesResponse qingcloud.CommonResponse

// DeleteServerCertificates
// 删除一个或多个服务器证书。
func (c *LOADBALANCER) DeleteServerCertificates(params DeleteServerCertificatesRequest) (DeleteServerCertificatesResponse, error) {
	var result DeleteServerCertificatesResponse
	err := c.Get("DeleteServerCertificates", qingcloud.TransfomRequestToParams(&params), &result)
	return result, err
}
