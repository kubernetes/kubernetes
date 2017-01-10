package slb

import (
	"fmt"
	"strings"
	"time"

	"github.com/denverdino/aliyungo/common"
)

type ListenerStatus string

const (
	Starting    = ListenerStatus("starting")
	Running     = ListenerStatus("running")
	Configuring = ListenerStatus("configuring")
	Stopping    = ListenerStatus("stopping")
	Stopped     = ListenerStatus("stopped")
)

type SchedulerType string

const (
	WRRScheduler = SchedulerType("wrr")
	WLCScheduler = SchedulerType("wlc")
)

type FlagType string

const (
	OnFlag  = FlagType("on")
	OffFlag = FlagType("off")
)

type StickySessionType string

const (
	InsertStickySessionType = StickySessionType("insert")
	ServerStickySessionType = StickySessionType("server")
)

const BackendServerPort = -520

type HealthCheckHttpCodeType string

const (
	HTTP_2XX = HealthCheckHttpCodeType("http_2xx")
	HTTP_3XX = HealthCheckHttpCodeType("http_3xx")
	HTTP_4XX = HealthCheckHttpCodeType("http_4xx")
	HTTP_5XX = HealthCheckHttpCodeType("http_5xx")
)

func EncodeHealthCheckHttpCodeType(healthCheckHttpCodes []HealthCheckHttpCodeType) (HealthCheckHttpCodeType, error) {
	code := ""

	if nil == healthCheckHttpCodes || len(healthCheckHttpCodes) < 1 {
		return "", fmt.Errorf("Invalid size of healthCheckHttpCodes")
	}

	for _, healthCheckHttpCode := range healthCheckHttpCodes {
		if strings.EqualFold(string(HTTP_2XX), string(healthCheckHttpCode)) ||
			strings.EqualFold(string(HTTP_3XX), string(healthCheckHttpCode)) ||
			strings.EqualFold(string(HTTP_4XX), string(healthCheckHttpCode)) ||
			strings.EqualFold(string(HTTP_5XX), string(healthCheckHttpCode)) {
			if "" == code {
				code = string(healthCheckHttpCode)
			} else {
				if strings.Contains(code, string(healthCheckHttpCode)) {
					return "", fmt.Errorf("Duplicates healthCheckHttpCode(%v in %v)", healthCheckHttpCode, healthCheckHttpCodes)
				}
				code += code + "," + string(healthCheckHttpCode)
			}
		} else {
			return "", fmt.Errorf("Invalid healthCheckHttpCode(%v in %v)", healthCheckHttpCode, healthCheckHttpCodes)
		}
	}
	return HealthCheckHttpCodeType(code), nil
}

type CommonLoadBalancerListenerResponse struct {
	common.Response
}

type HTTPListenerType struct {
	LoadBalancerId         string
	ListenerPort           int
	BackendServerPort      int
	Bandwidth              int
	Scheduler              SchedulerType
	StickySession          FlagType
	StickySessionType      StickySessionType
	CookieTimeout          int
	Cookie                 string
	HealthCheck            FlagType
	HealthCheckDomain      string
	HealthCheckURI         string
	HealthCheckConnectPort int
	HealthyThreshold       int
	UnhealthyThreshold     int
	HealthCheckTimeout     int
	HealthCheckInterval    int
	HealthCheckHttpCode    HealthCheckHttpCodeType
	VServerGroupId         string
	Gzip                   FlagType
}
type CreateLoadBalancerHTTPListenerArgs HTTPListenerType

// CreateLoadBalancerHTTPListener create HTTP listener on loadbalancer
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&CreateLoadBalancerHTTPListener
func (client *Client) CreateLoadBalancerHTTPListener(args *CreateLoadBalancerHTTPListenerArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("CreateLoadBalancerHTTPListener", args, response)
	return err
}

type HTTPSListenerType struct {
	HTTPListenerType
	ServerCertificateId string
}

type CreateLoadBalancerHTTPSListenerArgs HTTPSListenerType

// CreateLoadBalancerHTTPSListener create HTTPS listener on loadbalancer
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&CreateLoadBalancerHTTPSListener
func (client *Client) CreateLoadBalancerHTTPSListener(args *CreateLoadBalancerHTTPSListenerArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("CreateLoadBalancerHTTPSListener", args, response)
	return err
}

type HealthCheckType string

const (
	TCPHealthCheckType  = HealthCheckType("tcp")
	HTTPHealthCheckType = HealthCheckType("http")
)

type TCPListenerType struct {
	LoadBalancerId         string
	ListenerPort           int
	BackendServerPort      int
	Bandwidth              int
	Scheduler              SchedulerType
	PersistenceTimeout     int
	HealthCheckType        HealthCheckType
	HealthCheckDomain      string
	HealthCheckURI         string
	HealthCheckConnectPort int
	HealthyThreshold       int
	UnhealthyThreshold     int
	HealthCheckTimeout     int
	HealthCheckInterval    int
	HealthCheckHttpCode    HealthCheckHttpCodeType
	VServerGroupId         string
}

type CreateLoadBalancerTCPListenerArgs TCPListenerType

// CreateLoadBalancerTCPListener create TCP listener on loadbalancer
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&CreateLoadBalancerTCPListener
func (client *Client) CreateLoadBalancerTCPListener(args *CreateLoadBalancerTCPListenerArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("CreateLoadBalancerTCPListener", args, response)
	return err
}

type UDPListenerType struct {
	LoadBalancerId         string
	ListenerPort           int
	BackendServerPort      int
	Bandwidth              int
	Scheduler              SchedulerType
	PersistenceTimeout     int
	HealthCheckConnectPort int
	HealthyThreshold       int
	UnhealthyThreshold     int
	HealthCheckTimeout     int
	HealthCheckInterval    int
	VServerGroupId         string
}
type CreateLoadBalancerUDPListenerArgs UDPListenerType

// CreateLoadBalancerUDPListener create UDP listener on loadbalancer
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&CreateLoadBalancerUDPListener
func (client *Client) CreateLoadBalancerUDPListener(args *CreateLoadBalancerUDPListenerArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("CreateLoadBalancerUDPListener", args, response)
	return err
}

type CommonLoadBalancerListenerArgs struct {
	LoadBalancerId string
	ListenerPort   int
}

// DeleteLoadBalancerListener Delete listener
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DeleteLoadBalancerListener
func (client *Client) DeleteLoadBalancerListener(loadBalancerId string, port int) (err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("DeleteLoadBalancerListener", args, response)
	return err
}

// StartLoadBalancerListener Start listener
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&StartLoadBalancerListener
func (client *Client) StartLoadBalancerListener(loadBalancerId string, port int) (err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("StartLoadBalancerListener", args, response)
	return err
}

// StopLoadBalancerListener Stop listener
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&StopLoadBalancerListener
func (client *Client) StopLoadBalancerListener(loadBalancerId string, port int) (err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("StopLoadBalancerListener", args, response)
	return err
}

type AccessControlStatus string

const (
	OpenWhileList = AccessControlStatus("open_white_list")
	Close         = AccessControlStatus("close")
)

type SetListenerAccessControlStatusArgs struct {
	LoadBalancerId      string
	ListenerPort        int
	AccessControlStatus AccessControlStatus
}

// SetListenerAccessControlStatus Set listener access control status
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&SetListenerAccessControlStatus
func (client *Client) SetListenerAccessControlStatus(loadBalancerId string, port int, status AccessControlStatus) (err error) {
	args := &SetListenerAccessControlStatusArgs{
		LoadBalancerId:      loadBalancerId,
		ListenerPort:        port,
		AccessControlStatus: status,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("SetListenerAccessControlStatus", args, response)
	return err
}

type CommonListenerWhiteListItemArgs struct {
	LoadBalancerId string
	ListenerPort   int
	SourceItems    string
}

// AddListenerWhiteListItem Add listener white-list item
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&AddListenerWhiteListItem
func (client *Client) AddListenerWhiteListItem(loadBalancerId string, port int, sourceItems string) (err error) {
	args := &CommonListenerWhiteListItemArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
		SourceItems:    sourceItems,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("AddListenerWhiteListItem", args, response)
	return err
}

// RemoveListenerWhiteListItem Remove listener white-list item
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&RemoveListenerWhiteListItem
func (client *Client) RemoveListenerWhiteListItem(loadBalancerId string, port int, sourceItems string) (err error) {
	args := &CommonListenerWhiteListItemArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
		SourceItems:    sourceItems,
	}
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("RemoveListenerWhiteListItem", args, response)
	return err
}

type SetLoadBalancerHTTPListenerAttributeArgs CreateLoadBalancerHTTPListenerArgs

// SetLoadBalancerHTTPListenerAttribute Set HTTP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&SetLoadBalancerHTTPListenerAttribute
func (client *Client) SetLoadBalancerHTTPListenerAttribute(args *SetLoadBalancerHTTPListenerAttributeArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("SetLoadBalancerHTTPListenerAttribute", args, response)
	return err
}

type SetLoadBalancerHTTPSListenerAttributeArgs CreateLoadBalancerHTTPSListenerArgs

// SetLoadBalancerHTTPSListenerAttribute Set HTTPS listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&SetLoadBalancerHTTPSListenerAttribute
func (client *Client) SetLoadBalancerHTTPSListenerAttribute(args *SetLoadBalancerHTTPSListenerAttributeArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("SetLoadBalancerHTTPSListenerAttribute", args, response)
	return err
}

type SetLoadBalancerTCPListenerAttributeArgs CreateLoadBalancerTCPListenerArgs

// SetLoadBalancerTCPListenerAttribute Set TCP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&SetLoadBalancerTCPListenerAttribute
func (client *Client) SetLoadBalancerTCPListenerAttribute(args *SetLoadBalancerTCPListenerAttributeArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("SetLoadBalancerTCPListenerAttribute", args, response)
	return err
}

type SetLoadBalancerUDPListenerAttributeArgs CreateLoadBalancerUDPListenerArgs

// SetLoadBalancerUDPListenerAttribute Set UDP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&SetLoadBalancerUDPListenerAttribute
func (client *Client) SetLoadBalancerUDPListenerAttribute(args *SetLoadBalancerUDPListenerAttributeArgs) (err error) {
	response := &CommonLoadBalancerListenerResponse{}
	err = client.Invoke("SetLoadBalancerUDPListenerAttribute", args, response)
	return err
}

type DescribeLoadBalancerListenerAttributeResponse struct {
	common.Response
	Status ListenerStatus
}

type DescribeLoadBalancerHTTPListenerAttributeResponse struct {
	DescribeLoadBalancerListenerAttributeResponse
	HTTPListenerType
}

// DescribeLoadBalancerHTTPListenerAttribute Describe HTTP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DescribeLoadBalancerHTTPListenerAttribute
func (client *Client) DescribeLoadBalancerHTTPListenerAttribute(loadBalancerId string, port int) (response *DescribeLoadBalancerHTTPListenerAttributeResponse, err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response = &DescribeLoadBalancerHTTPListenerAttributeResponse{}
	err = client.Invoke("DescribeLoadBalancerHTTPListenerAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

type DescribeLoadBalancerHTTPSListenerAttributeResponse struct {
	DescribeLoadBalancerListenerAttributeResponse
	HTTPSListenerType
}

// DescribeLoadBalancerHTTPSListenerAttribute Describe HTTPS listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DescribeLoadBalancerHTTPSListenerAttribute
func (client *Client) DescribeLoadBalancerHTTPSListenerAttribute(loadBalancerId string, port int) (response *DescribeLoadBalancerHTTPSListenerAttributeResponse, err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response = &DescribeLoadBalancerHTTPSListenerAttributeResponse{}
	err = client.Invoke("DescribeLoadBalancerHTTPSListenerAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

type DescribeLoadBalancerTCPListenerAttributeResponse struct {
	DescribeLoadBalancerListenerAttributeResponse
	TCPListenerType
}

// DescribeLoadBalancerTCPListenerAttribute Describe TCP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DescribeLoadBalancerTCPListenerAttribute
func (client *Client) DescribeLoadBalancerTCPListenerAttribute(loadBalancerId string, port int) (response *DescribeLoadBalancerTCPListenerAttributeResponse, err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response = &DescribeLoadBalancerTCPListenerAttributeResponse{}
	err = client.Invoke("DescribeLoadBalancerTCPListenerAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

type DescribeLoadBalancerUDPListenerAttributeResponse struct {
	DescribeLoadBalancerListenerAttributeResponse
	UDPListenerType
}

// DescribeLoadBalancerUDPListenerAttribute Describe UDP listener attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DescribeLoadBalancerUDPListenerAttribute
func (client *Client) DescribeLoadBalancerUDPListenerAttribute(loadBalancerId string, port int) (response *DescribeLoadBalancerUDPListenerAttributeResponse, err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response = &DescribeLoadBalancerUDPListenerAttributeResponse{}
	err = client.Invoke("DescribeLoadBalancerUDPListenerAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}

type ListenerType string

const (
	UDP   = ListenerType("UDP")
	TCP   = ListenerType("TCP")
	HTTP  = ListenerType("HTTP")
	HTTPS = ListenerType("HTTPS")
)

const DefaultWaitForInterval = 5 //5 seconds
const DefaultTimeout = 60        //60 seconds

// WaitForListener waits for listener to given status
func (client *Client) WaitForListener(loadBalancerId string, port int, listenerType ListenerType) (status ListenerStatus, err error) {
	timeout := DefaultTimeout

	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}

	method := fmt.Sprintf("DescribeLoadBalancer%sListenerAttribute", listenerType)
	response := &DescribeLoadBalancerListenerAttributeResponse{}

	for {
		timeout = timeout - DefaultWaitForInterval
		if timeout <= 0 {
			return response.Status, common.GetClientErrorFromString("Timeout")
		}
		time.Sleep(DefaultWaitForInterval * time.Second)
		//Sleep first to ensure the previous request is sent
		err = client.Invoke(method, args, response)
		if err != nil {
			return "", err
		}
		if response.Status == Running || response.Status == Stopped {
			break
		}
	}
	return response.Status, nil
}

type DescribeListenerAccessControlAttributeResponse struct {
	common.Response
	AccessControlStatus AccessControlStatus
	SourceItems         string
}

// DescribeListenerAccessControlAttribute Describe listener access control attribute
//
// You can read doc at https://docs.aliyun.com/#/pub/slb/api-reference/api-related-listener&DescribeListenerAccessControlAttribute
func (client *Client) DescribeListenerAccessControlAttribute(loadBalancerId string, port int) (response *DescribeListenerAccessControlAttributeResponse, err error) {
	args := &CommonLoadBalancerListenerArgs{
		LoadBalancerId: loadBalancerId,
		ListenerPort:   port,
	}
	response = &DescribeListenerAccessControlAttributeResponse{}
	err = client.Invoke("DescribeListenerAccessControlAttribute", args, response)
	if err != nil {
		return nil, err
	}
	return response, err
}
