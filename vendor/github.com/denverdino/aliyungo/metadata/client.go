package metadata

import (
	"errors"
	"fmt"
	"github.com/denverdino/aliyungo/util"
	"io"
	"io/ioutil"
	"net"
	"net/http"
	"net/url"
	"strings"
	"time"
)

type Request struct {
}

const (
	ENDPOINT = "http://100.100.100.200"

	META_VERSION_LATEST = "latest"

	RS_TYPE_META_DATA = "meta-data"
	RS_TYPE_USER_DATA = "user-data"

	DNS_NAMESERVERS    = "dns-conf/nameservers"
	EIPV4              = "eipv4"
	HOSTNAME           = "hostname"
	IMAGE_ID           = "image-id"
	INSTANCE_ID        = "instance-id"
	MAC                = "mac"
	NETWORK_TYPE       = "network-type"
	NTP_CONF_SERVERS   = "ntp-conf/ntp-servers"
	OWNER_ACCOUNT_ID   = "owner-account-id"
	PRIVATE_IPV4       = "private-ipv4"
	REGION             = "region-id"
	SERIAL_NUMBER      = "serial-number"
	SOURCE_ADDRESS     = "source-address"
	VPC_CIDR_BLOCK     = "vpc-cidr-block"
	VPC_ID             = "vpc-id"
	VSWITCH_CIDR_BLOCK = "vswitch-cidr-block"
	VSWITCH_ID         = "vswitch-id"
)

type IMetaDataClient interface {
	Version(version string) IMetaDataClient
	ResourceType(rtype string) IMetaDataClient
	Resource(resource string) IMetaDataClient
	Go() ([]string, error)
	Url() (string, error)
}

type MetaData struct {
	c IMetaDataClient
}

func NewMetaData(client *http.Client) *MetaData {
	if client == nil {
		client = &http.Client{}
	}
	return &MetaData{
		c: &MetaDataClient{client: client},
	}
}

func (m *MetaData) HostName() (string, error) {

	hostname, err := m.c.Resource(HOSTNAME).Go()
	if err != nil {
		return "", err
	}
	return hostname[0], nil
}

func (m *MetaData) ImageID() (string, error) {

	image, err := m.c.Resource(IMAGE_ID).Go()
	if err != nil {
		return "", err
	}
	return image[0], err
}

func (m *MetaData) InstanceID() (string, error) {

	instanceid, err := m.c.Resource(INSTANCE_ID).Go()
	if err != nil {
		return "", err
	}
	return instanceid[0], err
}

func (m *MetaData) Mac() (string, error) {

	mac, err := m.c.Resource(MAC).Go()
	if err != nil {
		return "", err
	}
	return mac[0], nil
}

func (m *MetaData) NetworkType() (string, error) {

	network, err := m.c.Resource(NETWORK_TYPE).Go()
	if err != nil {
		return "", err
	}
	return network[0], nil
}

func (m *MetaData) OwnerAccountID() (string, error) {

	owner, err := m.c.Resource(OWNER_ACCOUNT_ID).Go()
	if err != nil {
		return "", err
	}
	return owner[0], nil
}

func (m *MetaData) PrivateIPv4() (string, error) {

	private, err := m.c.Resource(PRIVATE_IPV4).Go()
	if err != nil {
		return "", err
	}
	return private[0], nil
}

func (m *MetaData) Region() (string, error) {

	region, err := m.c.Resource(REGION).Go()
	if err != nil {
		return "", err
	}
	return region[0], nil
}

func (m *MetaData) SerialNumber() (string, error) {

	serial, err := m.c.Resource(SERIAL_NUMBER).Go()
	if err != nil {
		return "", err
	}
	return serial[0], nil
}

func (m *MetaData) SourceAddress() (string, error) {

	source, err := m.c.Resource(SOURCE_ADDRESS).Go()
	if err != nil {
		return "", err
	}
	return source[0], nil

}

func (m *MetaData) VpcCIDRBlock() (string, error) {

	vpcCIDR, err := m.c.Resource(VPC_CIDR_BLOCK).Go()
	if err != nil {
		return "", err
	}
	return vpcCIDR[0], err
}

func (m *MetaData) VpcID() (string, error) {

	vpcId, err := m.c.Resource(VPC_ID).Go()
	if err != nil {
		return "", err
	}
	return vpcId[0], err
}

func (m *MetaData) VswitchCIDRBlock() (string, error) {

	cidr, err := m.c.Resource(VSWITCH_CIDR_BLOCK).Go()
	if err != nil {
		return "", err
	}
	return cidr[0], err
}

func (m *MetaData) VswitchID() (string, error) {

	vswithcid, err := m.c.Resource(VSWITCH_ID).Go()
	if err != nil {
		return "", err
	}
	return vswithcid[0], err
}

func (m *MetaData) EIPv4() (string, error) {

	eip, err := m.c.Resource(EIPV4).Go()
	if err != nil {
		return "", err
	}
	return eip[0], nil
}

func (m *MetaData) DNSNameServers() ([]string, error) {

	data, err := m.c.Resource(DNS_NAMESERVERS).Go()
	if err != nil {
		return []string{}, err
	}
	return data, nil
}

func (m *MetaData) NTPConfigServers() ([]string, error) {

	data, err := m.c.Resource(NTP_CONF_SERVERS).Go()
	if err != nil {
		return []string{}, err
	}
	return data, nil
}

//
type MetaDataClient struct {
	version      string
	resourceType string
	resource     string
	client       *http.Client
}

func (vpc *MetaDataClient) Version(version string) IMetaDataClient {
	vpc.version = version
	return vpc
}

func (vpc *MetaDataClient) ResourceType(rtype string) IMetaDataClient {
	vpc.resourceType = rtype
	return vpc
}

func (vpc *MetaDataClient) Resource(resource string) IMetaDataClient {
	vpc.resource = resource
	return vpc
}

var retry = util.AttemptStrategy{
	Min:   5,
	Total: 5 * time.Second,
	Delay: 200 * time.Millisecond,
}

func (vpc *MetaDataClient) Url() (string, error) {
	if vpc.version == "" {
		vpc.version = "latest"
	}
	if vpc.resourceType == "" {
		vpc.resourceType = "meta-data"
	}
	if vpc.resource == "" {
		return "", errors.New("the resource you want to visit must not be nil!")
	}
	return fmt.Sprintf("%s/%s/%s/%s", ENDPOINT, vpc.version, vpc.resourceType, vpc.resource), nil
}

func (vpc *MetaDataClient) Go() (resu []string, err error) {
	for r := retry.Start(); r.Next(); {
		resu, err = vpc.send()
		if !shouldRetry(err) {
			break
		}
	}
	return resu, err
}

func (vpc *MetaDataClient) send() ([]string, error) {
	url, err := vpc.Url()
	if err != nil {
		return []string{}, err
	}
	requ, err := http.NewRequest(http.MethodGet, url, nil)

	if err != nil {
		return []string{}, err
	}
	resp, err := vpc.client.Do(requ)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != 200 {
		return nil, err
	}
	defer resp.Body.Close()

	data, err := ioutil.ReadAll(resp.Body)
	if err != nil {
		return []string{}, err
	}
	if string(data) == "" {
		return []string{""}, nil
	}
	return strings.Split(string(data), "\n"), nil
}

type TimeoutError interface {
	error
	Timeout() bool // Is the error a timeout?
}

func shouldRetry(err error) bool {
	if err == nil {
		return false
	}

	_, ok := err.(TimeoutError)
	if ok {
		return true
	}

	switch err {
	case io.ErrUnexpectedEOF, io.EOF:
		return true
	}
	switch e := err.(type) {
	case *net.DNSError:
		return true
	case *net.OpError:
		switch e.Op {
		case "read", "write":
			return true
		}
	case *url.Error:
		// url.Error can be returned either by net/url if a URL cannot be
		// parsed, or by net/http if the response is closed before the headers
		// are received or parsed correctly. In that later case, e.Op is set to
		// the HTTP method name with the first letter uppercased. We don't want
		// to retry on POST operations, since those are not idempotent, all the
		// other ones should be safe to retry.
		switch e.Op {
		case "Get", "Put", "Delete", "Head":
			return shouldRetry(e.Err)
		default:
			return false
		}
	}
	return false
}
