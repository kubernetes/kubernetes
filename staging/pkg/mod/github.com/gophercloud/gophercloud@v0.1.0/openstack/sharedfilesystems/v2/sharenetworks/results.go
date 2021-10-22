package sharenetworks

import (
	"encoding/json"
	"net/url"
	"strconv"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ShareNetwork contains all the information associated with an OpenStack
// ShareNetwork.
type ShareNetwork struct {
	// The Share Network ID
	ID string `json:"id"`
	// The UUID of the project where the share network was created
	ProjectID string `json:"project_id"`
	// The neutron network ID
	NeutronNetID string `json:"neutron_net_id"`
	// The neutron subnet ID
	NeutronSubnetID string `json:"neutron_subnet_id"`
	// The nova network ID
	NovaNetID string `json:"nova_net_id"`
	// The network type. A valid value is VLAN, VXLAN, GRE or flat
	NetworkType string `json:"network_type"`
	// The segmentation ID
	SegmentationID int `json:"segmentation_id"`
	// The IP block from which to allocate the network, in CIDR notation
	CIDR string `json:"cidr"`
	// The IP version of the network. A valid value is 4 or 6
	IPVersion int `json:"ip_version"`
	// The Share Network name
	Name string `json:"name"`
	// The Share Network description
	Description string `json:"description"`
	// The date and time stamp when the Share Network was created
	CreatedAt time.Time `json:"-"`
	// The date and time stamp when the Share Network was updated
	UpdatedAt time.Time `json:"-"`
}

func (r *ShareNetwork) UnmarshalJSON(b []byte) error {
	type tmp ShareNetwork
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
		UpdatedAt gophercloud.JSONRFC3339MilliNoZ `json:"updated_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = ShareNetwork(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)
	r.UpdatedAt = time.Time(s.UpdatedAt)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// ShareNetworkPage is a pagination.pager that is returned from a call to the List function.
type ShareNetworkPage struct {
	pagination.MarkerPageBase
}

// NextPageURL generates the URL for the page of results after this one.
func (r ShareNetworkPage) NextPageURL() (string, error) {
	currentURL := r.URL
	mark, err := r.Owner.LastMarker()
	if err != nil {
		return "", err
	}

	q := currentURL.Query()
	q.Set("offset", mark)
	currentURL.RawQuery = q.Encode()
	return currentURL.String(), nil
}

// LastMarker returns the last offset in a ListResult.
func (r ShareNetworkPage) LastMarker() (string, error) {
	maxInt := strconv.Itoa(int(^uint(0) >> 1))
	shareNetworks, err := ExtractShareNetworks(r)
	if err != nil {
		return maxInt, err
	}
	if len(shareNetworks) == 0 {
		return maxInt, nil
	}

	u, err := url.Parse(r.URL.String())
	if err != nil {
		return maxInt, err
	}
	queryParams := u.Query()
	offset := queryParams.Get("offset")
	limit := queryParams.Get("limit")

	// Limit is not present, only one page required
	if limit == "" {
		return maxInt, nil
	}

	iOffset := 0
	if offset != "" {
		iOffset, err = strconv.Atoi(offset)
		if err != nil {
			return maxInt, err
		}
	}
	iLimit, err := strconv.Atoi(limit)
	if err != nil {
		return maxInt, err
	}
	iOffset = iOffset + iLimit
	offset = strconv.Itoa(iOffset)

	return offset, nil
}

// IsEmpty satisifies the IsEmpty method of the Page interface
func (r ShareNetworkPage) IsEmpty() (bool, error) {
	shareNetworks, err := ExtractShareNetworks(r)
	return len(shareNetworks) == 0, err
}

// ExtractShareNetworks extracts and returns ShareNetworks. It is used while
// iterating over a sharenetworks.List call.
func ExtractShareNetworks(r pagination.Page) ([]ShareNetwork, error) {
	var s struct {
		ShareNetworks []ShareNetwork `json:"share_networks"`
	}
	err := (r.(ShareNetworkPage)).ExtractInto(&s)
	return s.ShareNetworks, err
}

// Extract will get the ShareNetwork object out of the commonResult object.
func (r commonResult) Extract() (*ShareNetwork, error) {
	var s struct {
		ShareNetwork *ShareNetwork `json:"share_network"`
	}
	err := r.ExtractInto(&s)
	return s.ShareNetwork, err
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

// AddSecurityServiceResult contains the response body and error from a security
// service addition request.
type AddSecurityServiceResult struct {
	commonResult
}

// RemoveSecurityServiceResult contains the response body and error from a security
// service removal request.
type RemoveSecurityServiceResult struct {
	commonResult
}
