//
// Copyright 2016, Sander van Harmelen
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//

package cloudstack

import (
	"bytes"
	"crypto/hmac"
	"crypto/sha1"
	"crypto/tls"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/ioutil"
	"net/http"
	"net/url"
	"regexp"
	"sort"
	"strings"
	"time"
)

// UnlimitedResourceID is a special ID to define an unlimited resource
const UnlimitedResourceID = "-1"

var idRegex = regexp.MustCompile(`^([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|-1)$`)

// IsID return true if the passed ID is either a UUID or a UnlimitedResourceID
func IsID(id string) bool {
	return idRegex.MatchString(id)
}

// OptionFunc can be passed to the courtesy helper functions to set additional parameters
type OptionFunc func(*CloudStackClient, interface{}) error

type CSError struct {
	ErrorCode   int    `json:"errorcode"`
	CSErrorCode int    `json:"cserrorcode"`
	ErrorText   string `json:"errortext"`
}

func (e *CSError) Error() error {
	return fmt.Errorf("CloudStack API error %d (CSExceptionErrorCode: %d): %s", e.ErrorCode, e.CSErrorCode, e.ErrorText)
}

type CloudStackClient struct {
	HTTPGETOnly bool // If `true` only use HTTP GET calls

	client  *http.Client // The http client for communicating
	baseURL string       // The base URL of the API
	apiKey  string       // Api key
	secret  string       // Secret key
	async   bool         // Wait for async calls to finish
	timeout int64        // Max waiting timeout in seconds for async jobs to finish; defaults to 300 seconds

	APIDiscovery     *APIDiscoveryService
	Account          *AccountService
	Address          *AddressService
	AffinityGroup    *AffinityGroupService
	Alert            *AlertService
	Asyncjob         *AsyncjobService
	Authentication   *AuthenticationService
	AutoScale        *AutoScaleService
	Baremetal        *BaremetalService
	Certificate      *CertificateService
	CloudIdentifier  *CloudIdentifierService
	Cluster          *ClusterService
	Configuration    *ConfigurationService
	DiskOffering     *DiskOfferingService
	Domain           *DomainService
	Event            *EventService
	Firewall         *FirewallService
	GuestOS          *GuestOSService
	Host             *HostService
	Hypervisor       *HypervisorService
	ISO              *ISOService
	ImageStore       *ImageStoreService
	InternalLB       *InternalLBService
	LDAP             *LDAPService
	Limit            *LimitService
	LoadBalancer     *LoadBalancerService
	NAT              *NATService
	NetworkACL       *NetworkACLService
	NetworkDevice    *NetworkDeviceService
	NetworkOffering  *NetworkOfferingService
	Network          *NetworkService
	Nic              *NicService
	NiciraNVP        *NiciraNVPService
	OvsElement       *OvsElementService
	Pod              *PodService
	Pool             *PoolService
	PortableIP       *PortableIPService
	Project          *ProjectService
	Quota            *QuotaService
	Region           *RegionService
	Resourcemetadata *ResourcemetadataService
	Resourcetags     *ResourcetagsService
	Router           *RouterService
	SSH              *SSHService
	SecurityGroup    *SecurityGroupService
	ServiceOffering  *ServiceOfferingService
	Snapshot         *SnapshotService
	StoragePool      *StoragePoolService
	StratosphereSSP  *StratosphereSSPService
	Swift            *SwiftService
	SystemCapacity   *SystemCapacityService
	SystemVM         *SystemVMService
	Template         *TemplateService
	UCS              *UCSService
	Usage            *UsageService
	User             *UserService
	VLAN             *VLANService
	VMGroup          *VMGroupService
	VPC              *VPCService
	VPN              *VPNService
	VirtualMachine   *VirtualMachineService
	Volume           *VolumeService
	Zone             *ZoneService
}

// Creates a new client for communicating with CloudStack
func newClient(apiurl string, apikey string, secret string, async bool, verifyssl bool) *CloudStackClient {
	cs := &CloudStackClient{
		client: &http.Client{
			Transport: &http.Transport{
				Proxy:           http.ProxyFromEnvironment,
				TLSClientConfig: &tls.Config{InsecureSkipVerify: !verifyssl}, // If verifyssl is true, skipping the verify should be false and vice versa
			},
			Timeout: time.Duration(60 * time.Second),
		},
		baseURL: apiurl,
		apiKey:  apikey,
		secret:  secret,
		async:   async,
		timeout: 300,
	}
	cs.APIDiscovery = NewAPIDiscoveryService(cs)
	cs.Account = NewAccountService(cs)
	cs.Address = NewAddressService(cs)
	cs.AffinityGroup = NewAffinityGroupService(cs)
	cs.Alert = NewAlertService(cs)
	cs.Asyncjob = NewAsyncjobService(cs)
	cs.Authentication = NewAuthenticationService(cs)
	cs.AutoScale = NewAutoScaleService(cs)
	cs.Baremetal = NewBaremetalService(cs)
	cs.Certificate = NewCertificateService(cs)
	cs.CloudIdentifier = NewCloudIdentifierService(cs)
	cs.Cluster = NewClusterService(cs)
	cs.Configuration = NewConfigurationService(cs)
	cs.DiskOffering = NewDiskOfferingService(cs)
	cs.Domain = NewDomainService(cs)
	cs.Event = NewEventService(cs)
	cs.Firewall = NewFirewallService(cs)
	cs.GuestOS = NewGuestOSService(cs)
	cs.Host = NewHostService(cs)
	cs.Hypervisor = NewHypervisorService(cs)
	cs.ISO = NewISOService(cs)
	cs.ImageStore = NewImageStoreService(cs)
	cs.InternalLB = NewInternalLBService(cs)
	cs.LDAP = NewLDAPService(cs)
	cs.Limit = NewLimitService(cs)
	cs.LoadBalancer = NewLoadBalancerService(cs)
	cs.NAT = NewNATService(cs)
	cs.NetworkACL = NewNetworkACLService(cs)
	cs.NetworkDevice = NewNetworkDeviceService(cs)
	cs.NetworkOffering = NewNetworkOfferingService(cs)
	cs.Network = NewNetworkService(cs)
	cs.Nic = NewNicService(cs)
	cs.NiciraNVP = NewNiciraNVPService(cs)
	cs.OvsElement = NewOvsElementService(cs)
	cs.Pod = NewPodService(cs)
	cs.Pool = NewPoolService(cs)
	cs.PortableIP = NewPortableIPService(cs)
	cs.Project = NewProjectService(cs)
	cs.Quota = NewQuotaService(cs)
	cs.Region = NewRegionService(cs)
	cs.Resourcemetadata = NewResourcemetadataService(cs)
	cs.Resourcetags = NewResourcetagsService(cs)
	cs.Router = NewRouterService(cs)
	cs.SSH = NewSSHService(cs)
	cs.SecurityGroup = NewSecurityGroupService(cs)
	cs.ServiceOffering = NewServiceOfferingService(cs)
	cs.Snapshot = NewSnapshotService(cs)
	cs.StoragePool = NewStoragePoolService(cs)
	cs.StratosphereSSP = NewStratosphereSSPService(cs)
	cs.Swift = NewSwiftService(cs)
	cs.SystemCapacity = NewSystemCapacityService(cs)
	cs.SystemVM = NewSystemVMService(cs)
	cs.Template = NewTemplateService(cs)
	cs.UCS = NewUCSService(cs)
	cs.Usage = NewUsageService(cs)
	cs.User = NewUserService(cs)
	cs.VLAN = NewVLANService(cs)
	cs.VMGroup = NewVMGroupService(cs)
	cs.VPC = NewVPCService(cs)
	cs.VPN = NewVPNService(cs)
	cs.VirtualMachine = NewVirtualMachineService(cs)
	cs.Volume = NewVolumeService(cs)
	cs.Zone = NewZoneService(cs)
	return cs
}

// Default non-async client. So for async calls you need to implement and check the async job result yourself. When using
// HTTPS with a self-signed certificate to connect to your CloudStack API, you would probably want to set 'verifyssl' to
// false so the call ignores the SSL errors/warnings.
func NewClient(apiurl string, apikey string, secret string, verifyssl bool) *CloudStackClient {
	cs := newClient(apiurl, apikey, secret, false, verifyssl)
	return cs
}

// For sync API calls this client behaves exactly the same as a standard client call, but for async API calls
// this client will wait until the async job is finished or until the configured AsyncTimeout is reached. When the async
// job finishes successfully it will return actual object received from the API and nil, but when the timout is
// reached it will return the initial object containing the async job ID for the running job and a warning.
func NewAsyncClient(apiurl string, apikey string, secret string, verifyssl bool) *CloudStackClient {
	cs := newClient(apiurl, apikey, secret, true, verifyssl)
	return cs
}

// When using the async client an api call will wait for the async call to finish before returning. The default is to poll for 300 seconds
// seconds, to check if the async job is finished.
func (cs *CloudStackClient) AsyncTimeout(timeoutInSeconds int64) {
	cs.timeout = timeoutInSeconds
}

var AsyncTimeoutErr = errors.New("Timeout while waiting for async job to finish")

// A helper function that you can use to get the result of a running async job. If the job is not finished within the configured
// timeout, the async job returns a AsyncTimeoutErr.
func (cs *CloudStackClient) GetAsyncJobResult(jobid string, timeout int64) (json.RawMessage, error) {
	var timer time.Duration
	currentTime := time.Now().Unix()

	for {
		p := cs.Asyncjob.NewQueryAsyncJobResultParams(jobid)
		r, err := cs.Asyncjob.QueryAsyncJobResult(p)
		if err != nil {
			return nil, err
		}

		// Status 1 means the job is finished successfully
		if r.Jobstatus == 1 {
			return r.Jobresult, nil
		}

		// When the status is 2, the job has failed
		if r.Jobstatus == 2 {
			if r.Jobresulttype == "text" {
				return nil, fmt.Errorf(string(r.Jobresult))
			} else {
				return nil, fmt.Errorf("Undefined error: %s", string(r.Jobresult))
			}
		}

		if time.Now().Unix()-currentTime > timeout {
			return nil, AsyncTimeoutErr
		}

		// Add an (extremely simple) exponential backoff like feature to prevent
		// flooding the CloudStack API
		if timer < 15 {
			timer++
		}

		time.Sleep(timer * time.Second)
	}
}

// Execute the request against a CS API. Will return the raw JSON data returned by the API and nil if
// no error occured. If the API returns an error the result will be nil and the HTTP error code and CS
// error details. If a processing (code) error occurs the result will be nil and the generated error
func (cs *CloudStackClient) newRequest(api string, params url.Values) (json.RawMessage, error) {
	params.Set("apiKey", cs.apiKey)
	params.Set("command", api)
	params.Set("response", "json")

	// Generate signature for API call
	// * Serialize parameters, URL encoding only values and sort them by key, done by encodeValues
	// * Convert the entire argument string to lowercase
	// * Replace all instances of '+' to '%20'
	// * Calculate HMAC SHA1 of argument string with CloudStack secret
	// * URL encode the string and convert to base64
	s := encodeValues(params)
	s2 := strings.ToLower(s)
	s3 := strings.Replace(s2, "+", "%20", -1)
	mac := hmac.New(sha1.New, []byte(cs.secret))
	mac.Write([]byte(s3))
	signature := base64.StdEncoding.EncodeToString(mac.Sum(nil))

	var err error
	var resp *http.Response
	if !cs.HTTPGETOnly && (api == "deployVirtualMachine" || api == "updateVirtualMachine") {
		// The deployVirtualMachine API should be called using a POST call
		// so we don't have to worry about the userdata size

		// Add the unescaped signature to the POST params
		params.Set("signature", signature)

		// Make a POST call
		resp, err = cs.client.PostForm(cs.baseURL, params)
	} else {
		// Create the final URL before we issue the request
		url := cs.baseURL + "?" + s + "&signature=" + url.QueryEscape(signature)

		// Make a GET call
		resp, err = cs.client.Get(url)
	}
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	b, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}

	// Need to get the raw value to make the result play nice
	b, err = getRawValue(b)
	if err != nil {
		return nil, err
	}

	if resp.StatusCode != 200 {
		var e CSError
		if err := json.Unmarshal(b, &e); err != nil {
			return nil, err
		}
		return nil, e.Error()
	}
	return b, nil
}

// Custom version of net/url Encode that only URL escapes values
// Unmodified portions here remain under BSD license of The Go Authors: https://go.googlesource.com/go/+/master/LICENSE
func encodeValues(v url.Values) string {
	if v == nil {
		return ""
	}
	var buf bytes.Buffer
	keys := make([]string, 0, len(v))
	for k := range v {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	for _, k := range keys {
		vs := v[k]
		prefix := k + "="
		for _, v := range vs {
			if buf.Len() > 0 {
				buf.WriteByte('&')
			}
			buf.WriteString(prefix)
			buf.WriteString(url.QueryEscape(v))
		}
	}
	return buf.String()
}

// Generic function to get the first raw value from a response as json.RawMessage
func getRawValue(b json.RawMessage) (json.RawMessage, error) {
	var m map[string]json.RawMessage
	if err := json.Unmarshal(b, &m); err != nil {
		return nil, err
	}
	for _, v := range m {
		return v, nil
	}
	return nil, fmt.Errorf("Unable to extract the raw value from:\n\n%s\n\n", string(b))
}

// ProjectIDSetter is an interface that every type that can set a project ID must implement
type ProjectIDSetter interface {
	SetProjectid(string)
}

// WithProject takes either a project name or ID and sets the `projectid` parameter
func WithProject(project string) OptionFunc {
	return func(cs *CloudStackClient, p interface{}) error {
		ps, ok := p.(ProjectIDSetter)

		if !ok || project == "" {
			return nil
		}

		if !IsID(project) {
			id, _, err := cs.Project.GetProjectID(project)
			if err != nil {
				return err
			}
			project = id
		}

		ps.SetProjectid(project)

		return nil
	}
}

// VPCIDSetter is an interface that every type that can set a vpc ID must implement
type VPCIDSetter interface {
	SetVpcid(string)
}

// WithVPCID takes a vpc ID and sets the `vpcid` parameter
func WithVPCID(id string) OptionFunc {
	return func(cs *CloudStackClient, p interface{}) error {
		vs, ok := p.(VPCIDSetter)

		if !ok || id == "" {
			return nil
		}

		vs.SetVpcid(id)

		return nil
	}
}

type APIDiscoveryService struct {
	cs *CloudStackClient
}

func NewAPIDiscoveryService(cs *CloudStackClient) *APIDiscoveryService {
	return &APIDiscoveryService{cs: cs}
}

type AccountService struct {
	cs *CloudStackClient
}

func NewAccountService(cs *CloudStackClient) *AccountService {
	return &AccountService{cs: cs}
}

type AddressService struct {
	cs *CloudStackClient
}

func NewAddressService(cs *CloudStackClient) *AddressService {
	return &AddressService{cs: cs}
}

type AffinityGroupService struct {
	cs *CloudStackClient
}

func NewAffinityGroupService(cs *CloudStackClient) *AffinityGroupService {
	return &AffinityGroupService{cs: cs}
}

type AlertService struct {
	cs *CloudStackClient
}

func NewAlertService(cs *CloudStackClient) *AlertService {
	return &AlertService{cs: cs}
}

type AsyncjobService struct {
	cs *CloudStackClient
}

func NewAsyncjobService(cs *CloudStackClient) *AsyncjobService {
	return &AsyncjobService{cs: cs}
}

type AuthenticationService struct {
	cs *CloudStackClient
}

func NewAuthenticationService(cs *CloudStackClient) *AuthenticationService {
	return &AuthenticationService{cs: cs}
}

type AutoScaleService struct {
	cs *CloudStackClient
}

func NewAutoScaleService(cs *CloudStackClient) *AutoScaleService {
	return &AutoScaleService{cs: cs}
}

type BaremetalService struct {
	cs *CloudStackClient
}

func NewBaremetalService(cs *CloudStackClient) *BaremetalService {
	return &BaremetalService{cs: cs}
}

type CertificateService struct {
	cs *CloudStackClient
}

func NewCertificateService(cs *CloudStackClient) *CertificateService {
	return &CertificateService{cs: cs}
}

type CloudIdentifierService struct {
	cs *CloudStackClient
}

func NewCloudIdentifierService(cs *CloudStackClient) *CloudIdentifierService {
	return &CloudIdentifierService{cs: cs}
}

type ClusterService struct {
	cs *CloudStackClient
}

func NewClusterService(cs *CloudStackClient) *ClusterService {
	return &ClusterService{cs: cs}
}

type ConfigurationService struct {
	cs *CloudStackClient
}

func NewConfigurationService(cs *CloudStackClient) *ConfigurationService {
	return &ConfigurationService{cs: cs}
}

type DiskOfferingService struct {
	cs *CloudStackClient
}

func NewDiskOfferingService(cs *CloudStackClient) *DiskOfferingService {
	return &DiskOfferingService{cs: cs}
}

type DomainService struct {
	cs *CloudStackClient
}

func NewDomainService(cs *CloudStackClient) *DomainService {
	return &DomainService{cs: cs}
}

type EventService struct {
	cs *CloudStackClient
}

func NewEventService(cs *CloudStackClient) *EventService {
	return &EventService{cs: cs}
}

type FirewallService struct {
	cs *CloudStackClient
}

func NewFirewallService(cs *CloudStackClient) *FirewallService {
	return &FirewallService{cs: cs}
}

type GuestOSService struct {
	cs *CloudStackClient
}

func NewGuestOSService(cs *CloudStackClient) *GuestOSService {
	return &GuestOSService{cs: cs}
}

type HostService struct {
	cs *CloudStackClient
}

func NewHostService(cs *CloudStackClient) *HostService {
	return &HostService{cs: cs}
}

type HypervisorService struct {
	cs *CloudStackClient
}

func NewHypervisorService(cs *CloudStackClient) *HypervisorService {
	return &HypervisorService{cs: cs}
}

type ISOService struct {
	cs *CloudStackClient
}

func NewISOService(cs *CloudStackClient) *ISOService {
	return &ISOService{cs: cs}
}

type ImageStoreService struct {
	cs *CloudStackClient
}

func NewImageStoreService(cs *CloudStackClient) *ImageStoreService {
	return &ImageStoreService{cs: cs}
}

type InternalLBService struct {
	cs *CloudStackClient
}

func NewInternalLBService(cs *CloudStackClient) *InternalLBService {
	return &InternalLBService{cs: cs}
}

type LDAPService struct {
	cs *CloudStackClient
}

func NewLDAPService(cs *CloudStackClient) *LDAPService {
	return &LDAPService{cs: cs}
}

type LimitService struct {
	cs *CloudStackClient
}

func NewLimitService(cs *CloudStackClient) *LimitService {
	return &LimitService{cs: cs}
}

type LoadBalancerService struct {
	cs *CloudStackClient
}

func NewLoadBalancerService(cs *CloudStackClient) *LoadBalancerService {
	return &LoadBalancerService{cs: cs}
}

type NATService struct {
	cs *CloudStackClient
}

func NewNATService(cs *CloudStackClient) *NATService {
	return &NATService{cs: cs}
}

type NetworkACLService struct {
	cs *CloudStackClient
}

func NewNetworkACLService(cs *CloudStackClient) *NetworkACLService {
	return &NetworkACLService{cs: cs}
}

type NetworkDeviceService struct {
	cs *CloudStackClient
}

func NewNetworkDeviceService(cs *CloudStackClient) *NetworkDeviceService {
	return &NetworkDeviceService{cs: cs}
}

type NetworkOfferingService struct {
	cs *CloudStackClient
}

func NewNetworkOfferingService(cs *CloudStackClient) *NetworkOfferingService {
	return &NetworkOfferingService{cs: cs}
}

type NetworkService struct {
	cs *CloudStackClient
}

func NewNetworkService(cs *CloudStackClient) *NetworkService {
	return &NetworkService{cs: cs}
}

type NicService struct {
	cs *CloudStackClient
}

func NewNicService(cs *CloudStackClient) *NicService {
	return &NicService{cs: cs}
}

type NiciraNVPService struct {
	cs *CloudStackClient
}

func NewNiciraNVPService(cs *CloudStackClient) *NiciraNVPService {
	return &NiciraNVPService{cs: cs}
}

type OvsElementService struct {
	cs *CloudStackClient
}

func NewOvsElementService(cs *CloudStackClient) *OvsElementService {
	return &OvsElementService{cs: cs}
}

type PodService struct {
	cs *CloudStackClient
}

func NewPodService(cs *CloudStackClient) *PodService {
	return &PodService{cs: cs}
}

type PoolService struct {
	cs *CloudStackClient
}

func NewPoolService(cs *CloudStackClient) *PoolService {
	return &PoolService{cs: cs}
}

type PortableIPService struct {
	cs *CloudStackClient
}

func NewPortableIPService(cs *CloudStackClient) *PortableIPService {
	return &PortableIPService{cs: cs}
}

type ProjectService struct {
	cs *CloudStackClient
}

func NewProjectService(cs *CloudStackClient) *ProjectService {
	return &ProjectService{cs: cs}
}

type QuotaService struct {
	cs *CloudStackClient
}

func NewQuotaService(cs *CloudStackClient) *QuotaService {
	return &QuotaService{cs: cs}
}

type RegionService struct {
	cs *CloudStackClient
}

func NewRegionService(cs *CloudStackClient) *RegionService {
	return &RegionService{cs: cs}
}

type ResourcemetadataService struct {
	cs *CloudStackClient
}

func NewResourcemetadataService(cs *CloudStackClient) *ResourcemetadataService {
	return &ResourcemetadataService{cs: cs}
}

type ResourcetagsService struct {
	cs *CloudStackClient
}

func NewResourcetagsService(cs *CloudStackClient) *ResourcetagsService {
	return &ResourcetagsService{cs: cs}
}

type RouterService struct {
	cs *CloudStackClient
}

func NewRouterService(cs *CloudStackClient) *RouterService {
	return &RouterService{cs: cs}
}

type SSHService struct {
	cs *CloudStackClient
}

func NewSSHService(cs *CloudStackClient) *SSHService {
	return &SSHService{cs: cs}
}

type SecurityGroupService struct {
	cs *CloudStackClient
}

func NewSecurityGroupService(cs *CloudStackClient) *SecurityGroupService {
	return &SecurityGroupService{cs: cs}
}

type ServiceOfferingService struct {
	cs *CloudStackClient
}

func NewServiceOfferingService(cs *CloudStackClient) *ServiceOfferingService {
	return &ServiceOfferingService{cs: cs}
}

type SnapshotService struct {
	cs *CloudStackClient
}

func NewSnapshotService(cs *CloudStackClient) *SnapshotService {
	return &SnapshotService{cs: cs}
}

type StoragePoolService struct {
	cs *CloudStackClient
}

func NewStoragePoolService(cs *CloudStackClient) *StoragePoolService {
	return &StoragePoolService{cs: cs}
}

type StratosphereSSPService struct {
	cs *CloudStackClient
}

func NewStratosphereSSPService(cs *CloudStackClient) *StratosphereSSPService {
	return &StratosphereSSPService{cs: cs}
}

type SwiftService struct {
	cs *CloudStackClient
}

func NewSwiftService(cs *CloudStackClient) *SwiftService {
	return &SwiftService{cs: cs}
}

type SystemCapacityService struct {
	cs *CloudStackClient
}

func NewSystemCapacityService(cs *CloudStackClient) *SystemCapacityService {
	return &SystemCapacityService{cs: cs}
}

type SystemVMService struct {
	cs *CloudStackClient
}

func NewSystemVMService(cs *CloudStackClient) *SystemVMService {
	return &SystemVMService{cs: cs}
}

type TemplateService struct {
	cs *CloudStackClient
}

func NewTemplateService(cs *CloudStackClient) *TemplateService {
	return &TemplateService{cs: cs}
}

type UCSService struct {
	cs *CloudStackClient
}

func NewUCSService(cs *CloudStackClient) *UCSService {
	return &UCSService{cs: cs}
}

type UsageService struct {
	cs *CloudStackClient
}

func NewUsageService(cs *CloudStackClient) *UsageService {
	return &UsageService{cs: cs}
}

type UserService struct {
	cs *CloudStackClient
}

func NewUserService(cs *CloudStackClient) *UserService {
	return &UserService{cs: cs}
}

type VLANService struct {
	cs *CloudStackClient
}

func NewVLANService(cs *CloudStackClient) *VLANService {
	return &VLANService{cs: cs}
}

type VMGroupService struct {
	cs *CloudStackClient
}

func NewVMGroupService(cs *CloudStackClient) *VMGroupService {
	return &VMGroupService{cs: cs}
}

type VPCService struct {
	cs *CloudStackClient
}

func NewVPCService(cs *CloudStackClient) *VPCService {
	return &VPCService{cs: cs}
}

type VPNService struct {
	cs *CloudStackClient
}

func NewVPNService(cs *CloudStackClient) *VPNService {
	return &VPNService{cs: cs}
}

type VirtualMachineService struct {
	cs *CloudStackClient
}

func NewVirtualMachineService(cs *CloudStackClient) *VirtualMachineService {
	return &VirtualMachineService{cs: cs}
}

type VolumeService struct {
	cs *CloudStackClient
}

func NewVolumeService(cs *CloudStackClient) *VolumeService {
	return &VolumeService{cs: cs}
}

type ZoneService struct {
	cs *CloudStackClient
}

func NewZoneService(cs *CloudStackClient) *ZoneService {
	return &ZoneService{cs: cs}
}
