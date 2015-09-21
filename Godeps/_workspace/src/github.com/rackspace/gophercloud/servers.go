// TODO(sfalvo): Remove Rackspace-specific Server structure fields and refactor them into a provider-specific access method.
// Be sure to update godocs accordingly.

package gophercloud

import (
	"fmt"
	"net/url"
	"strings"

	"github.com/mitchellh/mapstructure"
	"github.com/racker/perigee"
)

// genericServersProvider structures provide the implementation for generic OpenStack-compatible
// CloudServersProvider interfaces.
type genericServersProvider struct {
	// endpoint refers to the provider's API endpoint base URL.  This will be used to construct
	// and issue queries.
	endpoint string

	// Test context (if any) in which to issue requests.
	context *Context

	// access associates this API provider with a set of credentials,
	// which may be automatically renewed if they near expiration.
	access AccessProvider
}

// See the CloudServersProvider interface for details.
func (gcp *genericServersProvider) ListServersByFilter(filter url.Values) ([]Server, error) {
	var ss []Server

	err := gcp.context.WithReauth(gcp.access, func() error {
		url := gcp.endpoint + "/servers/detail?" + filter.Encode()
		return perigee.Get(url, perigee.Options{
			CustomClient: gcp.context.httpClient,
			Results:      &struct{ Servers *[]Server }{&ss},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gcp.access.AuthToken(),
			},
		})
	})
	return ss, err
}

// See the CloudServersProvider interface for details.
func (gcp *genericServersProvider) ListServersLinksOnly() ([]Server, error) {
	var ss []Server

	err := gcp.context.WithReauth(gcp.access, func() error {
		url := gcp.endpoint + "/servers"
		return perigee.Get(url, perigee.Options{
			CustomClient: gcp.context.httpClient,
			Results:      &struct{ Servers *[]Server }{&ss},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gcp.access.AuthToken(),
			},
		})
	})
	return ss, err
}

// See the CloudServersProvider interface for details.
func (gcp *genericServersProvider) ListServers() ([]Server, error) {
	var ss []Server

	err := gcp.context.WithReauth(gcp.access, func() error {
		url := gcp.endpoint + "/servers/detail"
		return perigee.Get(url, perigee.Options{
			CustomClient: gcp.context.httpClient,
			Results:      &struct{ Servers *[]Server }{&ss},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gcp.access.AuthToken(),
			},
		})
	})

	// Compatibility with v0.0.x -- we "map" our public and private
	// addresses into a legacy structure field for the benefit of
	// earlier software.

	if err != nil {
		return ss, err
	}

	for _, s := range ss {
		err = mapstructure.Decode(s.RawAddresses, &s.Addresses)
		if err != nil {
			return ss, err
		}
	}

	return ss, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ServerById(id string) (*Server, error) {
	var s *Server

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/servers/" + id
		return perigee.Get(url, perigee.Options{
			Results: &struct{ Server **Server }{&s},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{200},
		})
	})

	// Compatibility with v0.0.x -- we "map" our public and private
	// addresses into a legacy structure field for the benefit of
	// earlier software.

	if err != nil {
		return s, err
	}

	err = mapstructure.Decode(s.RawAddresses, &s.Addresses)

	return s, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) CreateServer(ns NewServer) (*NewServer, error) {
	var s *NewServer

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := gsp.endpoint + "/servers"
		return perigee.Post(ep, perigee.Options{
			ReqBody: &struct {
				Server *NewServer `json:"server"`
			}{&ns},
			Results: &struct{ Server **NewServer }{&s},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})

	return s, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) DeleteServerById(id string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/servers/" + id
		return perigee.Delete(url, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{204},
		})
	})
	return err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) SetAdminPassword(id, pw string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				ChangePassword struct {
					AdminPass string `json:"adminPass"`
				} `json:"changePassword"`
			}{
				struct {
					AdminPass string `json:"adminPass"`
				}{pw},
			},
			OkCodes: []int{202},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ResizeServer(id, newName, newFlavor, newDiskConfig string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		rr := ResizeRequest{
			Name:       newName,
			FlavorRef:  newFlavor,
			DiskConfig: newDiskConfig,
		}
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				Resize ResizeRequest `json:"resize"`
			}{rr},
			OkCodes: []int{202},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) RevertResize(id string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				RevertResize *int `json:"revertResize"`
			}{nil},
			OkCodes: []int{202},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ConfirmResize(id string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				ConfirmResize *int `json:"confirmResize"`
			}{nil},
			OkCodes: []int{204},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return err
}

// See the CloudServersProvider interface for details
func (gsp *genericServersProvider) RebootServer(id string, hard bool) error {
	return gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		types := map[bool]string{false: "SOFT", true: "HARD"}
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				Reboot struct {
					Type string `json:"type"`
				} `json:"reboot"`
			}{
				struct {
					Type string `json:"type"`
				}{types[hard]},
			},
			OkCodes: []int{202},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
}

// See the CloudServersProvider interface for details
func (gsp *genericServersProvider) RescueServer(id string) (string, error) {
	var pw *string

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				Rescue string `json:"rescue"`
			}{"none"},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				AdminPass **string `json:"adminPass"`
			}{&pw},
		})
	})
	return *pw, err
}

// See the CloudServersProvider interface for details
func (gsp *genericServersProvider) UnrescueServer(id string) error {
	return gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(url, perigee.Options{
			ReqBody: &struct {
				Unrescue *int `json:"unrescue"`
			}{nil},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})
}

// See the CloudServersProvider interface for details
func (gsp *genericServersProvider) UpdateServer(id string, changes NewServerSettings) (*Server, error) {
	var svr *Server
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := fmt.Sprintf("%s/servers/%s", gsp.endpoint, id)
		return perigee.Put(url, perigee.Options{
			ReqBody: &struct {
				Server NewServerSettings `json:"server"`
			}{changes},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				Server **Server `json:"server"`
			}{&svr},
		})
	})
	return svr, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) RebuildServer(id string, ns NewServer) (*Server, error) {
	var s *Server

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Post(ep, perigee.Options{
			ReqBody: &struct {
				Rebuild *NewServer `json:"rebuild"`
			}{&ns},
			Results: &struct{ Server **Server }{&s},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})

	return s, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListAddresses(id string) (AddressSet, error) {
	var pas *AddressSet
	var statusCode int

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/servers/%s/ips", gsp.endpoint, id)
		return perigee.Get(ep, perigee.Options{
			Results: &struct{ Addresses **AddressSet }{&pas},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes:    []int{200, 203},
			StatusCode: &statusCode,
		})
	})

	if err != nil {
		if statusCode == 203 {
			err = WarnUnauthoritative
		}
	}

	return *pas, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListAddressesByNetwork(id, networkLabel string) (NetworkAddress, error) {
	pas := make(NetworkAddress)
	var statusCode int

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/servers/%s/ips/%s", gsp.endpoint, id, networkLabel)
		return perigee.Get(ep, perigee.Options{
			Results: &pas,
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes:    []int{200, 203},
			StatusCode: &statusCode,
		})
	})

	if err != nil {
		if statusCode == 203 {
			err = WarnUnauthoritative
		}
	}

	return pas, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) CreateImage(id string, ci CreateImage) (string, error) {
	response, err := gsp.context.ResponseWithReauth(gsp.access, func() (*perigee.Response, error) {
		ep := fmt.Sprintf("%s/servers/%s/action", gsp.endpoint, id)
		return perigee.Request("POST", ep, perigee.Options{
			ReqBody: &struct {
				CreateImage *CreateImage `json:"createImage"`
			}{&ci},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{200, 202},
		})
	})

	if err != nil {
		return "", err
	}
	location, err := response.HttpResponse.Location()
	if err != nil {
		return "", err
	}

	// Return the last element of the location which is the image id
	locationArr := strings.Split(location.Path, "/")
	return locationArr[len(locationArr)-1], err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListSecurityGroups() ([]SecurityGroup, error) {
	var sgs []SecurityGroup

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-groups", gsp.endpoint)
		return perigee.Get(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				SecurityGroups *[]SecurityGroup `json:"security_groups"`
			}{&sgs},
		})
	})
	return sgs, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) CreateSecurityGroup(desired SecurityGroup) (*SecurityGroup, error) {
	var actual *SecurityGroup

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-groups", gsp.endpoint)
		return perigee.Post(ep, perigee.Options{
			ReqBody: struct {
				AddSecurityGroup SecurityGroup `json:"security_group"`
			}{desired},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				SecurityGroup **SecurityGroup `json:"security_group"`
			}{&actual},
		})
	})
	return actual, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListSecurityGroupsByServerId(id string) ([]SecurityGroup, error) {
	var sgs []SecurityGroup

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/servers/%s/os-security-groups", gsp.endpoint, id)
		return perigee.Get(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				SecurityGroups *[]SecurityGroup `json:"security_groups"`
			}{&sgs},
		})
	})
	return sgs, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) SecurityGroupById(id int) (*SecurityGroup, error) {
	var actual *SecurityGroup

	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-groups/%d", gsp.endpoint, id)
		return perigee.Get(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct {
				SecurityGroup **SecurityGroup `json:"security_group"`
			}{&actual},
		})
	})
	return actual, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) DeleteSecurityGroupById(id int) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-groups/%d", gsp.endpoint, id)
		return perigee.Delete(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			OkCodes: []int{202},
		})
	})
	return err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) ListDefaultSGRules() ([]SGRule, error) {
	var sgrs []SGRule
	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-group-default-rules", gsp.endpoint)
		return perigee.Get(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct{ Security_group_default_rules *[]SGRule }{&sgrs},
		})
	})
	return sgrs, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) CreateDefaultSGRule(r SGRule) (*SGRule, error) {
	var sgr *SGRule
	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-group-default-rules", gsp.endpoint)
		return perigee.Post(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct{ Security_group_default_rule **SGRule }{&sgr},
			ReqBody: struct {
				Security_group_default_rule SGRule `json:"security_group_default_rule"`
			}{r},
		})
	})
	return sgr, err
}

// See the CloudServersProvider interface for details.
func (gsp *genericServersProvider) GetSGRule(id string) (*SGRule, error) {
	var sgr *SGRule
	err := gsp.context.WithReauth(gsp.access, func() error {
		ep := fmt.Sprintf("%s/os-security-group-default-rules/%s", gsp.endpoint, id)
		return perigee.Get(ep, perigee.Options{
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
			Results: &struct{ Security_group_default_rule **SGRule }{&sgr},
		})
	})
	return sgr, err
}

// SecurityGroup provides a description of a security group, including all its rules.
type SecurityGroup struct {
	Description string   `json:"description,omitempty"`
	Id          int      `json:"id,omitempty"`
	Name        string   `json:"name,omitempty"`
	Rules       []SGRule `json:"rules,omitempty"`
	TenantId    string   `json:"tenant_id,omitempty"`
}

// SGRule encapsulates a single rule which applies to a security group.
// This definition is just a guess, based on the documentation found in another extension here: http://docs.openstack.org/api/openstack-compute/2/content/GET_os-security-group-default-rules-v2_listSecGroupDefaultRules_v2__tenant_id__os-security-group-rules_ext-os-security-group-default-rules.html
type SGRule struct {
	FromPort   int                    `json:"from_port,omitempty"`
	Id         int                    `json:"id,omitempty"`
	IpProtocol string                 `json:"ip_protocol,omitempty"`
	IpRange    map[string]interface{} `json:"ip_range,omitempty"`
	ToPort     int                    `json:"to_port,omitempty"`
}

// RaxBandwidth provides measurement of server bandwidth consumed over a given audit interval.
type RaxBandwidth struct {
	AuditPeriodEnd    string `json:"audit_period_end"`
	AuditPeriodStart  string `json:"audit_period_start"`
	BandwidthInbound  int64  `json:"bandwidth_inbound"`
	BandwidthOutbound int64  `json:"bandwidth_outbound"`
	Interface         string `json:"interface"`
}

// A VersionedAddress denotes either an IPv4 or IPv6 (depending on version indicated)
// address.
type VersionedAddress struct {
	Addr    string `json:"addr"`
	Version int    `json:"version"`
}

// An AddressSet provides a set of public and private IP addresses for a resource.
// Each address has a version to identify if IPv4 or IPv6.
type AddressSet struct {
	Public  []VersionedAddress `json:"public"`
	Private []VersionedAddress `json:"private"`
}

type NetworkAddress map[string][]VersionedAddress

// Server records represent (virtual) hardware instances (not configurations) accessible by the user.
//
// The AccessIPv4 / AccessIPv6 fields provides IP addresses for the server in the IPv4 or IPv6 format, respectively.
//
// Addresses provides addresses for any attached isolated networks.
// The version field indicates whether the IP address is version 4 or 6.
// Note: only public and private pools appear here.
// To get the complete set, use the AllAddressPools() method instead.
//
// Created tells when the server entity was created.
//
// The Flavor field includes the flavor ID and flavor links.
//
// The compute provisioning algorithm has an anti-affinity property that
// attempts to spread customer VMs across hosts.
// Under certain situations,
// VMs from the same customer might be placed on the same host.
// The HostId field represents the host your server runs on and
// can be used to determine this scenario if it is relevant to your application.
// Note that HostId is unique only per account; it is not globally unique.
//
// Id provides the server's unique identifier.
// This field must be treated opaquely.
//
// Image indicates which image is installed on the server.
//
// Links provides one or more means of accessing the server.
//
// Metadata provides a small key-value store for application-specific information.
//
// Name provides a human-readable name for the server.
//
// Progress indicates how far along it is towards being provisioned.
// 100 represents complete, while 0 represents just beginning.
//
// Status provides an indication of what the server's doing at the moment.
// A server will be in ACTIVE state if it's ready for use.
//
// OsDcfDiskConfig indicates the server's boot volume configuration.
// Valid values are:
//     AUTO
//     ----
//     The server is built with a single partition the size of the target flavor disk.
//     The file system is automatically adjusted to fit the entire partition.
//     This keeps things simple and automated.
//     AUTO is valid only for images and servers with a single partition that use the EXT3 file system.
//     This is the default setting for applicable Rackspace base images.
//
//     MANUAL
//     ------
//     The server is built using whatever partition scheme and file system is in the source image.
//     If the target flavor disk is larger,
//     the remaining disk space is left unpartitioned.
//     This enables images to have non-EXT3 file systems, multiple partitions, and so on,
//     and enables you to manage the disk configuration.
//
// RaxBandwidth provides measures of the server's inbound and outbound bandwidth per interface.
//
// OsExtStsPowerState provides an indication of the server's power.
// This field appears to be a set of flag bits:
//
//           ... 4  3   2   1   0
//         +--//--+---+---+---+---+
//         | .... | 0 | S | 0 | I |
//         +--//--+---+---+---+---+
//                      |       |
//                      |       +---  0=Instance is down.
//                      |             1=Instance is up.
//                      |
//                      +-----------  0=Server is switched ON.
//                                    1=Server is switched OFF.
//                                    (note reverse logic.)
//
// Unused bits should be ignored when read, and written as 0 for future compatibility.
//
// OsExtStsTaskState and OsExtStsVmState work together
// to provide visibility in the provisioning process for the instance.
// Consult Rackspace documentation at
// http://docs.rackspace.com/servers/api/v2/cs-devguide/content/ch_extensions.html#ext_status
// for more details.  It's too lengthy to include here.
type Server struct {
	AccessIPv4         string `json:"accessIPv4"`
	AccessIPv6         string `json:"accessIPv6"`
	Addresses          AddressSet
	Created            string            `json:"created"`
	Flavor             FlavorLink        `json:"flavor"`
	HostId             string            `json:"hostId"`
	Id                 string            `json:"id"`
	Image              ImageLink         `json:"image"`
	Links              []Link            `json:"links"`
	Metadata           map[string]string `json:"metadata"`
	Name               string            `json:"name"`
	Progress           int               `json:"progress"`
	Status             string            `json:"status"`
	TenantId           string            `json:"tenant_id"`
	Updated            string            `json:"updated"`
	UserId             string            `json:"user_id"`
	OsDcfDiskConfig    string            `json:"OS-DCF:diskConfig"`
	RaxBandwidth       []RaxBandwidth    `json:"rax-bandwidth:bandwidth"`
	OsExtStsPowerState int               `json:"OS-EXT-STS:power_state"`
	OsExtStsTaskState  string            `json:"OS-EXT-STS:task_state"`
	OsExtStsVmState    string            `json:"OS-EXT-STS:vm_state"`

	RawAddresses map[string]interface{} `json:"addresses"`
}

// AllAddressPools returns a complete set of address pools available on the server.
// The name of each pool supported keys the map.
// The value of the map contains the addresses provided in the corresponding pool.
func (s *Server) AllAddressPools() (map[string][]VersionedAddress, error) {
	pools := make(map[string][]VersionedAddress, 0)
	for pool, subtree := range s.RawAddresses {
		addresses := make([]VersionedAddress, 0)
		err := mapstructure.Decode(subtree, &addresses)
		if err != nil {
			return nil, err
		}
		pools[pool] = addresses
	}
	return pools, nil
}

// NewServerSettings structures record those fields of the Server structure to change
// when updating a server (see UpdateServer method).
type NewServerSettings struct {
	Name       string `json:"name,omitempty"`
	AccessIPv4 string `json:"accessIPv4,omitempty"`
	AccessIPv6 string `json:"accessIPv6,omitempty"`
}

// NewServer structures are used for both requests and responses.
// The fields discussed below are relevent for server-creation purposes.
//
// The Name field contains the desired name of the server.
// Note that (at present) Rackspace permits more than one server with the same name;
// however, software should not depend on this.
// Not only will Rackspace support thank you, so will your own devops engineers.
// A name is required.
//
// The ImageRef field contains the ID of the desired software image to place on the server.
// This ID must be found in the image slice returned by the Images() function.
// This field is required.
//
// The FlavorRef field contains the ID of the server configuration desired for deployment.
// This ID must be found in the flavor slice returned by the Flavors() function.
// This field is required.
//
// For OsDcfDiskConfig, refer to the Image or Server structure documentation.
// This field defaults to "AUTO" if not explicitly provided.
//
// Metadata contains a small key/value association of arbitrary data.
// Neither Rackspace nor OpenStack places significance on this field in any way.
// This field defaults to an empty map if not provided.
//
// Personality specifies the contents of certain files in the server's filesystem.
// The files and their contents are mapped through a slice of FileConfig structures.
// If not provided, all filesystem entities retain their image-specific configuration.
//
// Networks specifies an affinity for the server's various networks and interfaces.
// Networks are identified through UUIDs; see NetworkConfig structure documentation for more details.
// If not provided, network affinity is determined automatically.
//
// The AdminPass field may be used to provide a root- or administrator-password
// during the server provisioning process.
// If not provided, a random password will be automatically generated and returned in this field.
//
// The following fields are intended to be used to communicate certain results about the server being provisioned.
// When attempting to create a new server, these fields MUST not be provided.
// They'll be filled in by the response received from the Rackspace APIs.
//
// The Id field contains the server's unique identifier.
// The identifier's scope is best assumed to be bound by the user's account, unless other arrangements have been made with Rackspace.
//
// The SecurityGroup field allows the user to specify a security group at launch.
//
// Any Links provided are used to refer to the server specifically by URL.
// These links are useful for making additional REST calls not explicitly supported by Gorax.
type NewServer struct {
	Name            string                   `json:"name,omitempty"`
	ImageRef        string                   `json:"imageRef,omitempty"`
	FlavorRef       string                   `json:"flavorRef,omitempty"`
	Metadata        map[string]string        `json:"metadata,omitempty"`
	Personality     []FileConfig             `json:"personality,omitempty"`
	Networks        []NetworkConfig          `json:"networks,omitempty"`
	AdminPass       string                   `json:"adminPass,omitempty"`
	KeyPairName     string                   `json:"key_name,omitempty"`
	Id              string                   `json:"id,omitempty"`
	Links           []Link                   `json:"links,omitempty"`
	OsDcfDiskConfig string                   `json:"OS-DCF:diskConfig,omitempty"`
	SecurityGroup   []map[string]interface{} `json:"security_groups,omitempty"`
	ConfigDrive     bool                     `json:"config_drive"`
	UserData        string                   `json:"user_data"`
}

// ResizeRequest structures are used internally to encode to JSON the parameters required to resize a server instance.
// Client applications will not use this structure (no API accepts an instance of this structure).
// See the Region method ResizeServer() for more details on how to resize a server.
type ResizeRequest struct {
	Name       string `json:"name,omitempty"`
	FlavorRef  string `json:"flavorRef"`
	DiskConfig string `json:"OS-DCF:diskConfig,omitempty"`
}

type CreateImage struct {
	Name     string            `json:"name"`
	Metadata map[string]string `json:"metadata,omitempty"`
}
