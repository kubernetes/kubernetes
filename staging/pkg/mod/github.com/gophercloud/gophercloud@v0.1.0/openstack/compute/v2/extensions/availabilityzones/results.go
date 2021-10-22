package availabilityzones

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ServerAvailabilityZoneExt is an extension to the base Server object.
type ServerAvailabilityZoneExt struct {
	// AvailabilityZone is the availabilty zone the server is in.
	AvailabilityZone string `json:"OS-EXT-AZ:availability_zone"`
}

// ServiceState represents the state of a service in an AvailabilityZone.
type ServiceState struct {
	Active    bool      `json:"active"`
	Available bool      `json:"available"`
	UpdatedAt time.Time `json:"-"`
}

// UnmarshalJSON to override default
func (r *ServiceState) UnmarshalJSON(b []byte) error {
	type tmp ServiceState
	var s struct {
		tmp
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = ServiceState(s.tmp)

	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}

// Services is a map of services contained in an AvailabilityZone.
type Services map[string]ServiceState

// Hosts is map of hosts/nodes contained in an AvailabilityZone.
// Each host can have multiple services.
type Hosts map[string]Services

// ZoneState represents the current state of the availability zone.
type ZoneState struct {
	// Returns true if the availability zone is available
	Available bool `json:"available"`
}

// AvailabilityZone contains all the information associated with an OpenStack
// AvailabilityZone.
type AvailabilityZone struct {
	Hosts Hosts `json:"hosts"`
	// The availability zone name
	ZoneName  string    `json:"zoneName"`
	ZoneState ZoneState `json:"zoneState"`
}

type AvailabilityZonePage struct {
	pagination.SinglePageBase
}

// ExtractAvailabilityZones returns a slice of AvailabilityZones contained in a
// single page of results.
func ExtractAvailabilityZones(r pagination.Page) ([]AvailabilityZone, error) {
	var s struct {
		AvailabilityZoneInfo []AvailabilityZone `json:"availabilityZoneInfo"`
	}
	err := (r.(AvailabilityZonePage)).ExtractInto(&s)
	return s.AvailabilityZoneInfo, err
}
