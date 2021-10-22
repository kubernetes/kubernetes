package securityservices

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// SecurityService contains all the information associated with an OpenStack
// SecurityService.
type SecurityService struct {
	// The security service ID
	ID string `json:"id"`
	// The UUID of the project where the security service was created
	ProjectID string `json:"project_id"`
	// The security service domain
	Domain string `json:"domain"`
	// The security service status
	Status string `json:"status"`
	// The security service type. A valid value is ldap, kerberos, or active_directory
	Type string `json:"type"`
	// The security service name
	Name string `json:"name"`
	// The security service description
	Description string `json:"description"`
	// The DNS IP address that is used inside the tenant network
	DNSIP string `json:"dns_ip"`
	// The security service organizational unit (OU)
	OU string `json:"ou"`
	// The security service user or group name that is used by the tenant
	User string `json:"user"`
	// The user password, if you specify a user
	Password string `json:"password"`
	// The security service host name or IP address
	Server string `json:"server"`
	// The date and time stamp when the security service was created
	CreatedAt time.Time `json:"-"`
	// The date and time stamp when the security service was updated
	UpdatedAt time.Time `json:"-"`
}

func (r *SecurityService) UnmarshalJSON(b []byte) error {
	type tmp SecurityService
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = SecurityService(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// SecurityServicePage is a pagination.pager that is returned from a call to the List function.
type SecurityServicePage struct {
	pagination.SinglePageBase
}

// IsEmpty returns true if a ListResult contains no SecurityServices.
func (r SecurityServicePage) IsEmpty() (bool, error) {
	securityServices, err := ExtractSecurityServices(r)
	return len(securityServices) == 0, err
}

// ExtractSecurityServices extracts and returns SecurityServices. It is used while
// iterating over a securityservices.List call.
func ExtractSecurityServices(r pagination.Page) ([]SecurityService, error) {
	var s struct {
		SecurityServices []SecurityService `json:"security_services"`
	}
	err := (r.(SecurityServicePage)).ExtractInto(&s)
	return s.SecurityServices, err
}

// Extract will get the SecurityService object out of the commonResult object.
func (r commonResult) Extract() (*SecurityService, error) {
	var s struct {
		SecurityService *SecurityService `json:"security_service"`
	}
	err := r.ExtractInto(&s)
	return s.SecurityService, err
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// DeleteResult contains the response body and error from a Delete request.
type DeleteResult struct {
	gophercloud.ErrResult
}

// GetResult contains the response body and error from a Get request.
type GetResult struct {
	commonResult
}

// UpdateResult contains the response body and error from an Update request.
type UpdateResult struct {
	commonResult
}
