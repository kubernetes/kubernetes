package loadbalancer

import (
	"time"
)

type Loadbalancer struct {
	Status           string `json:"status"`
	IsApplied        int    `json:"is_applied"`
	Description      string `json:"description"`
	LoadbalancerName string `json:"loadbalancer_name"`
	TransitionStatus string `json:"transition_status"`
	Eips             []struct {
		EipId   string `json:"eip_id"`
		EipName string `json:"eip_name"`
		EipAddr string `json:"eip_addr"`
	} `json:"eips"`
	Listeners []struct {
		Forwardfor               int       `json:"forwardfor"`
		LoadbalancerListenerID   string    `json:"loadbalancer_listener_id"`
		BalanceMode              string    `json:"balance_mode"`
		ListenerProtocol         string    `json:"listener_protocol"`
		BackendProtocol          string    `json:"backend_protocol"`
		HealthyCheckMethod       string    `json:"healthy_check_method"`
		SessionSticky            string    `json:"session_sticky"`
		LoadbalancerListenerName string    `json:"loadbalancer_listener_name"`
		Controller               string    `json:"controller"`
		CreateTime               time.Time `json:"create_time"`
		HealthyCheckOption       string    `json:"healthy_check_option"`
		LoadbalancerID           string    `json:"loadbalancer_id"`
		ListenerPort             int       `json:"listener_port"`
	} `json:"listeners"`
	CreateTime      time.Time `json:"create_time"`
	StatusTime      time.Time `json:"status_time"`
	SecurityGroupID string    `json:"security_group_id"`
	LoadbalancerID  string    `json:"loadbalancer_id"`
	Vxnet           struct {
		VxnetName string `json:"vxnet_name"`
		PrivateIP string `json:"private_ip"`
		VxnetID   string `json:"vxnet_id"`
	} `json:"vxnet"`
}

type LoadbalancerListener struct {
	Forwardfor               int    `json:"forwardfor"`
	LoadbalancerListenerID   string `json:"loadbalancer_listener_id"`
	BalanceMode              string `json:"balance_mode"`
	ListenerProtocol         string `json:"listener_protocol"`
	BackendProtocol          string `json:"backend_protocol"`
	HealthyCheckMethod       string `json:"healthy_check_method"`
	HealthyCheckOption       string `json:"healthy_check_option"`
	SessionSticky            string `json:"session_sticky"`
	LoadbalancerListenerName string `json:"loadbalancer_listener_name"`
	Backends                 []struct {
		LoadbalancerBackendID   string    `json:"loadbalancer_backend_id"`
		LoadbalancerBackendName string    `json:"loadbalancer_backend_name"`
		Weight                  int       `json:"weight"`
		Port                    int       `json:"port"`
		ResourceID              string    `json:"resource_id"`
		LoadbalancerListenerID  string    `json:"loadbalancer_listener_id"`
		LoadbalancerID          string    `json:"loadbalancer_id"`
		CreateTime              time.Time `json:"create_time"`
	} `json:"backends"`
	CreateTime     time.Time `json:"create_time"`
	LoadbalancerID string    `json:"loadbalancer_id"`
	ListenerPort   int       `json:"listener_port"`
	ListenerOption int       `json:"listener_option"`
}

type LoadbalancerBackend struct {
	LoadbalancerBackendID   string      `json:"loadbalancer_backend_id"`
	Weight                  int         `json:"weight"`
	ResourceID              string      `json:"resource_id"`
	LoadbalancerBackendName interface{} `json:"loadbalancer_backend_name"`
	Port                    int         `json:"port"`
	CreateTime              time.Time   `json:"create_time"`
	LoadbalancerListenerID  string      `json:"loadbalancer_listener_id"`
	LoadbalancerID          string      `json:"loadbalancer_id"`
}

type LoadbalancerPolicy struct {
	LoadbalancerPolicyID   string    `json:"loadbalancer_policy_id"`
	LoadbalancerPolicyName string    `json:"loadbalancer_policy_name"`
	LoadbalancerIds        []string  `json:"loadbalancer_ids"`
	CreateTime             time.Time `json:"create_time"`
	IsApplied              int       `json:"is_applied"`
}

type LoadbalancerPoliciyRule struct {
	RuleType                   string      `json:"rule_type"`
	Val                        string      `json:"val"`
	LoadbalancerPolicyRuleID   string      `json:"loadbalancer_policy_rule_id"`
	LoadbalancerPolicyRuleName interface{} `json:"loadbalancer_policy_rule_name"`
	LoadbalancerPolicyID       string      `json:"loadbalancer_policy_id"`
}

type ServerCertificate struct {
	PrivateKey            string    `json:"private_key"`
	ServerCertificateID   string    `json:"server_certificate_id"`
	ServerCertificateName string    `json:"server_certificate_name"`
	Description           string    `json:"description"`
	CreateTime            time.Time `json:"create_time"`
	CertificateContent    string    `json:"certificate_content"`
}
