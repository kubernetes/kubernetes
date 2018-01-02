package availabilityzones

import (
	"encoding/json"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// AvailabilityZone contains all the information associated with an OpenStack
// AvailabilityZone.
type AvailabilityZone struct {
	// The availability zone ID.
	ID string `json:"id"`
	// The name of the availability zone.
	Name string `json:"name"`
	// The date and time stamp when the availability zone was created.
	CreatedAt time.Time `json:"-"`
	// The date and time stamp when the availability zone was updated.
	UpdatedAt time.Time `json:"-"`
}

type commonResult struct {
	gophercloud.Result
}

// ListResult contains the response body and error from a List request.
type AvailabilityZonePage struct {
	pagination.SinglePageBase
}

// ExtractAvailabilityZones will get the AvailabilityZone objects out of the shareTypeAccessResult object.
func ExtractAvailabilityZones(r pagination.Page) ([]AvailabilityZone, error) {
	var a struct {
		AvailabilityZone []AvailabilityZone `json:"availability_zones"`
	}
	err := (r.(AvailabilityZonePage)).ExtractInto(&a)
	return a.AvailabilityZone, err
}

func (r *AvailabilityZone) UnmarshalJSON(b []byte) error {
	type tmp AvailabilityZone
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = AvailabilityZone(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}
