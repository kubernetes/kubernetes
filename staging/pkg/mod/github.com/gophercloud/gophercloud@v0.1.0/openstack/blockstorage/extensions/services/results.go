package services

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// Service represents a Blockstorage service in the OpenStack cloud.
type Service struct {
	// The binary name of the service.
	Binary string `json:"binary"`

	// The reason for disabling a service.
	DisabledReason string `json:"disabled_reason"`

	// The name of the host.
	Host string `json:"host"`

	// The state of the service. One of up or down.
	State string `json:"state"`

	// The status of the service. One of available or unavailable.
	Status string `json:"status"`

	// The date and time stamp when the extension was last updated.
	UpdatedAt time.Time `json:"-"`

	// The availability zone name.
	Zone string `json:"zone"`

	// The following fields are optional

	// The host is frozen or not. Only in cinder-volume service.
	Frozen bool `json:"frozen"`

	// The cluster name. Only in cinder-volume service.
	Cluster string `json:"cluster"`

	// The volume service replication status. Only in cinder-volume service.
	ReplicationStatus string `json:"replication_status"`

	// The ID of active storage backend. Only in cinder-volume service.
	ActiveBackendID string `json:"active_backend_id"`
}

// UnmarshalJSON to override default
func (r *Service) UnmarshalJSON(b []byte) error {
	type tmp Service
	var s struct {
		tmp
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Service(s.tmp)

	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}

// ServicePage represents a single page of all Services from a List request.
type ServicePage struct {
	pagination.SinglePageBase
}

// IsEmpty determines whether or not a page of Services contains any results.
func (page ServicePage) IsEmpty() (bool, error) {
	services, err := ExtractServices(page)
	return len(services) == 0, err
}

func ExtractServices(r pagination.Page) ([]Service, error) {
	var s struct {
		Service []Service `json:"services"`
	}
	err := (r.(ServicePage)).ExtractInto(&s)
	return s.Service, err
}
