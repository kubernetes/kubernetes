// The elb package provides types and functions for interaction with the AWS
// Elastic Load Balancing service (ELB)
package elb

import (
	"encoding/xml"
	"net/http"
	"net/url"
	"strconv"
	"time"

	"github.com/mitchellh/goamz/aws"
)

// The ELB type encapsulates operations operations with the elb endpoint.
type ELB struct {
	aws.Auth
	aws.Region
	httpClient *http.Client
}

const APIVersion = "2012-06-01"

// New creates a new ELB instance.
func New(auth aws.Auth, region aws.Region) *ELB {
	return NewWithClient(auth, region, aws.RetryingClient)
}

func NewWithClient(auth aws.Auth, region aws.Region, httpClient *http.Client) *ELB {
	return &ELB{auth, region, httpClient}
}

func (elb *ELB) query(params map[string]string, resp interface{}) error {
	params["Version"] = APIVersion
	params["Timestamp"] = time.Now().In(time.UTC).Format(time.RFC3339)

	endpoint, err := url.Parse(elb.Region.ELBEndpoint)
	if err != nil {
		return err
	}

	sign(elb.Auth, "GET", "/", params, endpoint.Host)
	endpoint.RawQuery = multimap(params).Encode()
	r, err := elb.httpClient.Get(endpoint.String())

	if err != nil {
		return err
	}
	defer r.Body.Close()
	if r.StatusCode > 200 {
		return buildError(r)
	}

	decoder := xml.NewDecoder(r.Body)
	decodedBody := decoder.Decode(resp)

	return decodedBody
}

func buildError(r *http.Response) error {
	var (
		err    Error
		errors xmlErrors
	)
	xml.NewDecoder(r.Body).Decode(&errors)
	if len(errors.Errors) > 0 {
		err = errors.Errors[0]
	}
	err.StatusCode = r.StatusCode
	if err.Message == "" {
		err.Message = r.Status
	}
	return &err
}

func multimap(p map[string]string) url.Values {
	q := make(url.Values, len(p))
	for k, v := range p {
		q[k] = []string{v}
	}
	return q
}

func makeParams(action string) map[string]string {
	params := make(map[string]string)
	params["Action"] = action
	return params
}

// ----------------------------------------------------------------------------
// ELB objects

// A listener attaches to an elb
type Listener struct {
	InstancePort     int64  `xml:"Listener>InstancePort"`
	InstanceProtocol string `xml:"Listener>InstanceProtocol"`
	SSLCertificateId string `xml:"Listener>SSLCertificateId"`
	LoadBalancerPort int64  `xml:"Listener>LoadBalancerPort"`
	Protocol         string `xml:"Listener>Protocol"`
}

// An Instance attaches to an elb
type Instance struct {
	InstanceId string `xml:"InstanceId"`
}

// A tag attached to an elb
type Tag struct {
	Key   string `xml:"Key"`
	Value string `xml:"Value"`
}

// An InstanceState from an elb health query
type InstanceState struct {
	InstanceId  string `xml:"InstanceId"`
	Description string `xml:"Description"`
	State       string `xml:"State"`
	ReasonCode  string `xml:"ReasonCode"`
}

// ----------------------------------------------------------------------------
// AddTags

type AddTags struct {
	LoadBalancerNames []string
	Tags              []Tag
}

type AddTagsResp struct {
	RequestId string `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) AddTags(options *AddTags) (resp *AddTagsResp, err error) {
	params := makeParams("AddTags")

	for i, v := range options.LoadBalancerNames {
		params["LoadBalancerNames.member."+strconv.Itoa(i+1)] = v
	}

	for i, v := range options.Tags {
		params["Tags.member."+strconv.Itoa(i+1)+".Key"] = v.Key
		params["Tags.member."+strconv.Itoa(i+1)+".Value"] = v.Value
	}

	resp = &AddTagsResp{}

	err = elb.query(params, resp)

	return resp, err
}

// ----------------------------------------------------------------------------
// RemoveTags

type RemoveTags struct {
	LoadBalancerNames []string
	TagKeys           []string
}

type RemoveTagsResp struct {
	RequestId string `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) RemoveTags(options *RemoveTags) (resp *RemoveTagsResp, err error) {
	params := makeParams("RemoveTags")

	for i, v := range options.LoadBalancerNames {
		params["LoadBalancerNames.member."+strconv.Itoa(i+1)] = v
	}

	for i, v := range options.TagKeys {
		params["Tags.member."+strconv.Itoa(i+1)+".Key"] = v
	}

	resp = &RemoveTagsResp{}

	err = elb.query(params, resp)

	return resp, err
}

// ----------------------------------------------------------------------------
// Create

// The CreateLoadBalancer request parameters
type CreateLoadBalancer struct {
	AvailZone        []string
	Listeners        []Listener
	LoadBalancerName string
	Internal         bool // true for vpc elbs
	SecurityGroups   []string
	Subnets          []string
	Tags             []Tag
}

type CreateLoadBalancerResp struct {
	DNSName   string `xml:"CreateLoadBalancerResult>DNSName"`
	RequestId string `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) CreateLoadBalancer(options *CreateLoadBalancer) (resp *CreateLoadBalancerResp, err error) {
	params := makeParams("CreateLoadBalancer")

	params["LoadBalancerName"] = options.LoadBalancerName

	for i, v := range options.AvailZone {
		params["AvailabilityZones.member."+strconv.Itoa(i+1)] = v
	}

	for i, v := range options.SecurityGroups {
		params["SecurityGroups.member."+strconv.Itoa(i+1)] = v
	}

	for i, v := range options.Subnets {
		params["Subnets.member."+strconv.Itoa(i+1)] = v
	}

	for i, v := range options.Listeners {
		params["Listeners.member."+strconv.Itoa(i+1)+".LoadBalancerPort"] = strconv.FormatInt(v.LoadBalancerPort, 10)
		params["Listeners.member."+strconv.Itoa(i+1)+".InstancePort"] = strconv.FormatInt(v.InstancePort, 10)
		params["Listeners.member."+strconv.Itoa(i+1)+".Protocol"] = v.Protocol
		params["Listeners.member."+strconv.Itoa(i+1)+".InstanceProtocol"] = v.InstanceProtocol
		params["Listeners.member."+strconv.Itoa(i+1)+".SSLCertificateId"] = v.SSLCertificateId
	}

	for i, v := range options.Tags {
		params["Tags.member."+strconv.Itoa(i+1)+".Key"] = v.Key
		params["Tags.member."+strconv.Itoa(i+1)+".Value"] = v.Value
	}

	if options.Internal {
		params["Scheme"] = "internal"
	}

	resp = &CreateLoadBalancerResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Destroy

// The DestroyLoadBalancer request parameters
type DeleteLoadBalancer struct {
	LoadBalancerName string
}

func (elb *ELB) DeleteLoadBalancer(options *DeleteLoadBalancer) (resp *SimpleResp, err error) {
	params := makeParams("DeleteLoadBalancer")

	params["LoadBalancerName"] = options.LoadBalancerName

	resp = &SimpleResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Describe

// An individual load balancer
type LoadBalancer struct {
	LoadBalancerName  string      `xml:"LoadBalancerName"`
	Listeners         []Listener  `xml:"ListenerDescriptions>member"`
	Instances         []Instance  `xml:"Instances>member"`
	HealthCheck       HealthCheck `xml:"HealthCheck"`
	AvailabilityZones []string    `xml:"AvailabilityZones>member"`
	HostedZoneNameID  string      `xml:"CanonicalHostedZoneNameID"`
	DNSName           string      `xml:"DNSName"`
	SecurityGroups    []string    `xml:"SecurityGroups>member"`
	Scheme            string      `xml:"Scheme"`
	Subnets           []string    `xml:"Subnets>member"`
}

// DescribeLoadBalancer request params
type DescribeLoadBalancer struct {
	Names []string
}

type DescribeLoadBalancersResp struct {
	RequestId     string         `xml:"ResponseMetadata>RequestId"`
	LoadBalancers []LoadBalancer `xml:"DescribeLoadBalancersResult>LoadBalancerDescriptions>member"`
}

func (elb *ELB) DescribeLoadBalancers(options *DescribeLoadBalancer) (resp *DescribeLoadBalancersResp, err error) {
	params := makeParams("DescribeLoadBalancers")

	for i, v := range options.Names {
		params["LoadBalancerNames.member."+strconv.Itoa(i+1)] = v
	}

	resp = &DescribeLoadBalancersResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Attributes

type AccessLog struct {
	EmitInterval   int64
	Enabled        bool
	S3BucketName   string
	S3BucketPrefix string
}

type ConnectionDraining struct {
	Enabled bool
	Timeout int64
}

type LoadBalancerAttributes struct {
	CrossZoneLoadBalancingEnabled bool
	ConnectionSettingsIdleTimeout int64
	ConnectionDraining            ConnectionDraining
	AccessLog                     AccessLog
}

type ModifyLoadBalancerAttributes struct {
	LoadBalancerName       string
	LoadBalancerAttributes LoadBalancerAttributes
}

func (elb *ELB) ModifyLoadBalancerAttributes(options *ModifyLoadBalancerAttributes) (resp *SimpleResp, err error) {
	params := makeParams("ModifyLoadBalancerAttributes")

	params["LoadBalancerName"] = options.LoadBalancerName
	params["LoadBalancerAttributes.CrossZoneLoadBalancing.Enabled"] = strconv.FormatBool(options.LoadBalancerAttributes.CrossZoneLoadBalancingEnabled)
	if options.LoadBalancerAttributes.ConnectionSettingsIdleTimeout > 0 {
		params["LoadBalancerAttributes.ConnectionSettings.IdleTimeout"] = strconv.Itoa(int(options.LoadBalancerAttributes.ConnectionSettingsIdleTimeout))
	}
	if options.LoadBalancerAttributes.ConnectionDraining.Timeout > 0 {
		params["LoadBalancerAttributes.ConnectionDraining.Timeout"] = strconv.Itoa(int(options.LoadBalancerAttributes.ConnectionDraining.Timeout))
	}
	params["LoadBalancerAttributes.ConnectionDraining.Enabled"] = strconv.FormatBool(options.LoadBalancerAttributes.ConnectionDraining.Enabled)
	params["LoadBalancerAttributes.AccessLog.Enabled"] = strconv.FormatBool(options.LoadBalancerAttributes.AccessLog.Enabled)
	if options.LoadBalancerAttributes.AccessLog.Enabled {
		params["LoadBalancerAttributes.AccessLog.EmitInterval"] = strconv.Itoa(int(options.LoadBalancerAttributes.AccessLog.EmitInterval))
		params["LoadBalancerAttributes.AccessLog.S3BucketName"] = options.LoadBalancerAttributes.AccessLog.S3BucketName
		params["LoadBalancerAttributes.AccessLog.S3BucketPrefix"] = options.LoadBalancerAttributes.AccessLog.S3BucketPrefix
	}

	resp = &SimpleResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Instance Registration / deregistration

// The RegisterInstancesWithLoadBalancer request parameters
type RegisterInstancesWithLoadBalancer struct {
	LoadBalancerName string
	Instances        []string
}

type RegisterInstancesWithLoadBalancerResp struct {
	Instances []Instance `xml:"RegisterInstancesWithLoadBalancerResult>Instances>member"`
	RequestId string     `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) RegisterInstancesWithLoadBalancer(options *RegisterInstancesWithLoadBalancer) (resp *RegisterInstancesWithLoadBalancerResp, err error) {
	params := makeParams("RegisterInstancesWithLoadBalancer")

	params["LoadBalancerName"] = options.LoadBalancerName

	for i, v := range options.Instances {
		params["Instances.member."+strconv.Itoa(i+1)+".InstanceId"] = v
	}

	resp = &RegisterInstancesWithLoadBalancerResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// The DeregisterInstancesFromLoadBalancer request parameters
type DeregisterInstancesFromLoadBalancer struct {
	LoadBalancerName string
	Instances        []string
}

type DeregisterInstancesFromLoadBalancerResp struct {
	Instances []Instance `xml:"DeregisterInstancesFromLoadBalancerResult>Instances>member"`
	RequestId string     `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) DeregisterInstancesFromLoadBalancer(options *DeregisterInstancesFromLoadBalancer) (resp *DeregisterInstancesFromLoadBalancerResp, err error) {
	params := makeParams("DeregisterInstancesFromLoadBalancer")

	params["LoadBalancerName"] = options.LoadBalancerName

	for i, v := range options.Instances {
		params["Instances.member."+strconv.Itoa(i+1)+".InstanceId"] = v
	}

	resp = &DeregisterInstancesFromLoadBalancerResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// DescribeTags

type DescribeTags struct {
	LoadBalancerNames []string
}

type LoadBalancerTag struct {
	Tags             []Tag  `xml:"Tags>member"`
	LoadBalancerName string `xml:"LoadBalancerName"`
}

type DescribeTagsResp struct {
	LoadBalancerTags []LoadBalancerTag `xml:"DescribeTagsResult>TagDescriptions>member"`
	NextToken        string            `xml:"DescribeTagsResult>NextToken"`
	RequestId        string            `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) DescribeTags(options *DescribeTags) (resp *DescribeTagsResp, err error) {
	params := makeParams("DescribeTags")

	for i, v := range options.LoadBalancerNames {
		params["LoadBalancerNames.member."+strconv.Itoa(i+1)] = v
	}

	resp = &DescribeTagsResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Health Checks

type HealthCheck struct {
	HealthyThreshold   int64  `xml:"HealthyThreshold"`
	UnhealthyThreshold int64  `xml:"UnhealthyThreshold"`
	Interval           int64  `xml:"Interval"`
	Target             string `xml:"Target"`
	Timeout            int64  `xml:"Timeout"`
}

type ConfigureHealthCheck struct {
	LoadBalancerName string
	Check            HealthCheck
}

type ConfigureHealthCheckResp struct {
	Check     HealthCheck `xml:"ConfigureHealthCheckResult>HealthCheck"`
	RequestId string      `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) ConfigureHealthCheck(options *ConfigureHealthCheck) (resp *ConfigureHealthCheckResp, err error) {
	params := makeParams("ConfigureHealthCheck")

	params["LoadBalancerName"] = options.LoadBalancerName
	params["HealthCheck.HealthyThreshold"] = strconv.Itoa(int(options.Check.HealthyThreshold))
	params["HealthCheck.UnhealthyThreshold"] = strconv.Itoa(int(options.Check.UnhealthyThreshold))
	params["HealthCheck.Interval"] = strconv.Itoa(int(options.Check.Interval))
	params["HealthCheck.Target"] = options.Check.Target
	params["HealthCheck.Timeout"] = strconv.Itoa(int(options.Check.Timeout))

	resp = &ConfigureHealthCheckResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// ----------------------------------------------------------------------------
// Instance Health

// The DescribeInstanceHealth request parameters
type DescribeInstanceHealth struct {
	LoadBalancerName string
}

type DescribeInstanceHealthResp struct {
	InstanceStates []InstanceState `xml:"DescribeInstanceHealthResult>InstanceStates>member"`
	RequestId      string          `xml:"ResponseMetadata>RequestId"`
}

func (elb *ELB) DescribeInstanceHealth(options *DescribeInstanceHealth) (resp *DescribeInstanceHealthResp, err error) {
	params := makeParams("DescribeInstanceHealth")

	params["LoadBalancerName"] = options.LoadBalancerName

	resp = &DescribeInstanceHealthResp{}

	err = elb.query(params, resp)

	if err != nil {
		resp = nil
	}

	return
}

// Responses

type SimpleResp struct {
	RequestId string `xml:"ResponseMetadata>RequestId"`
}

type xmlErrors struct {
	Errors []Error `xml:"Error"`
}

// Error encapsulates an elb error.
type Error struct {
	// HTTP status code of the error.
	StatusCode int

	// AWS code of the error.
	Code string

	// Message explaining the error.
	Message string
}

func (e *Error) Error() string {
	var prefix string
	if e.Code != "" {
		prefix = e.Code + ": "
	}
	if prefix == "" && e.StatusCode > 0 {
		prefix = strconv.Itoa(e.StatusCode) + ": "
	}
	return prefix + e.Message
}
