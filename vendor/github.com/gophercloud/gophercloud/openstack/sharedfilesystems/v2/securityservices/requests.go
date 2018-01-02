package securityservices

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

type SecurityServiceType string

// Valid security service types
const (
	LDAP            SecurityServiceType = "ldap"
	Kerberos        SecurityServiceType = "kerberos"
	ActiveDirectory SecurityServiceType = "active_directory"
)

// CreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type CreateOptsBuilder interface {
	ToSecurityServiceCreateMap() (map[string]interface{}, error)
}

// CreateOpts contains options for creating a SecurityService. This object is
// passed to the securityservices.Create function. For more information about
// these parameters, see the SecurityService object.
type CreateOpts struct {
	// The security service type. A valid value is ldap, kerberos, or active_directory
	Type SecurityServiceType `json:"type" required:"true"`
	// The security service name
	Name string `json:"name,omitempty"`
	// The security service description
	Description string `json:"description,omitempty"`
	// The DNS IP address that is used inside the tenant network
	DNSIP string `json:"dns_ip,omitempty"`
	// The security service user or group name that is used by the tenant
	User string `json:"user,omitempty"`
	// The user password, if you specify a user
	Password string `json:"password,omitempty"`
	// The security service domain
	Domain string `json:"domain,omitempty"`
	// The security service host name or IP address
	Server string `json:"server,omitempty"`
}

// ToSecurityServicesCreateMap assembles a request body based on the contents of a
// CreateOpts.
func (opts CreateOpts) ToSecurityServiceCreateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "security_service")
}

// Create will create a new SecurityService based on the values in CreateOpts. To
// extract the SecurityService object from the response, call the Extract method
// on the CreateResult.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToSecurityServiceCreateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}

// Delete will delete the existing SecurityService with the provided ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// ListOptsBuilder allows extensions to add additional parameters to the List
// request.
type ListOptsBuilder interface {
	ToSecurityServiceListQuery() (string, error)
}

// ListOpts holds options for listing SecurityServices. It is passed to the
// securityservices.List function.
type ListOpts struct {
	// admin-only option. Set it to true to see all tenant security services.
	AllTenants bool `q:"all_tenants"`
	// The security service ID
	ID string `q:"id"`
	// The security service domain
	Domain string `q:"domain"`
	// The security service type. A valid value is ldap, kerberos, or active_directory
	Type SecurityServiceType `q:"type"`
	// The security service name
	Name string `q:"name"`
	// The DNS IP address that is used inside the tenant network
	DNSIP string `q:"dns_ip"`
	// The security service user or group name that is used by the tenant
	User string `q:"user"`
	// The security service host name or IP address
	Server string `q:"server"`
	// The ID of the share network using security services
	ShareNetworkID string `q:"share_network_id"`
}

// ToSecurityServiceListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToSecurityServiceListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List returns SecurityServices optionally limited by the conditions provided in ListOpts.
func List(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(client)
	if opts != nil {
		query, err := opts.ToSecurityServiceListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return SecurityServicePage{pagination.SinglePageBase(r)}
	})
}

// Get retrieves the SecurityService with the provided ID. To extract the SecurityService
// object from the response, call the Extract method on the GetResult.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToSecurityServiceUpdateMap() (map[string]interface{}, error)
}

// UpdateOpts contain options for updating an existing SecurityService. This object is passed
// to the securityservices.Update function. For more information about the parameters, see
// the SecurityService object.
type UpdateOpts struct {
	// The security service name
	Name string `json:"name"`
	// The security service description
	Description string `json:"description,omitempty"`
	// The security service type. A valid value is ldap, kerberos, or active_directory
	Type string `json:"type,omitempty"`
	// The DNS IP address that is used inside the tenant network
	DNSIP string `json:"dns_ip,omitempty"`
	// The security service user or group name that is used by the tenant
	User string `json:"user,omitempty"`
	// The user password, if you specify a user
	Password string `json:"password,omitempty"`
	// The security service domain
	Domain string `json:"domain,omitempty"`
	// The security service host name or IP address
	Server string `json:"server,omitempty"`
}

// ToSecurityServiceUpdateMap assembles a request body based on the contents of an
// UpdateOpts.
func (opts UpdateOpts) ToSecurityServiceUpdateMap() (map[string]interface{}, error) {
	return gophercloud.BuildRequestBody(opts, "security_service")
}

// Update will update the SecurityService with provided information. To extract the updated
// SecurityService from the response, call the Extract method on the UpdateResult.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToSecurityServiceUpdateMap()
	if err != nil {
		r.Err = err
		return
	}
	_, r.Err = client.Put(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes: []int{200},
	})
	return
}
