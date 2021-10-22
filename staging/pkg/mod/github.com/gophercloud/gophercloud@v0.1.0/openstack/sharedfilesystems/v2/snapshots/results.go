package snapshots

import (
	"encoding/json"
	"net/url"
	"strconv"
	"time"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

const (
	invalidMarker = "-1"
)

// Snapshot contains all information associated with an OpenStack Snapshot
type Snapshot struct {
	// The UUID of the snapshot
	ID string `json:"id"`
	// The name of the snapshot
	Name string `json:"name,omitempty"`
	// A description of the snapshot
	Description string `json:"description,omitempty"`
	// UUID of the share from which the snapshot was created
	ShareID string `json:"share_id"`
	// The shared file system protocol
	ShareProto string `json:"share_proto"`
	// Size of the snapshot share in GB
	ShareSize int `json:"share_size"`
	// Size of the snapshot in GB
	Size int `json:"size"`
	// The snapshot status
	Status string `json:"status"`
	// The UUID of the project in which the snapshot was created
	ProjectID string `json:"project_id"`
	// Timestamp when the snapshot was created
	CreatedAt time.Time `json:"-"`
	// Snapshot links for pagination
	Links []map[string]string `json:"links"`
}

func (r *Snapshot) UnmarshalJSON(b []byte) error {
	type tmp Snapshot
	var s struct {
		tmp
		CreatedAt gophercloud.JSONRFC3339MilliNoZ `json:"created_at"`
	}
	err := json.Unmarshal(b, &s)
	if err != nil {
		return err
	}
	*r = Snapshot(s.tmp)

	r.CreatedAt = time.Time(s.CreatedAt)

	return nil
}

type commonResult struct {
	gophercloud.Result
}

// Extract will get the Snapshot object from the commonResult
func (r commonResult) Extract() (*Snapshot, error) {
	var s struct {
		Snapshot *Snapshot `json:"snapshot"`
	}
	err := r.ExtractInto(&s)
	return s.Snapshot, err
}

// CreateResult contains the response body and error from a Create request.
type CreateResult struct {
	commonResult
}

// SnapshotPage is a pagination.pager that is returned from a call to the List function.
type SnapshotPage struct {
	pagination.MarkerPageBase
}

// NextPageURL generates the URL for the page of results after this one.
func (r SnapshotPage) NextPageURL() (string, error) {
	currentURL := r.URL
	mark, err := r.Owner.LastMarker()
	if err != nil {
		return "", err
	}
	if mark == invalidMarker {
		return "", nil
	}

	q := currentURL.Query()
	q.Set("offset", mark)
	currentURL.RawQuery = q.Encode()
	return currentURL.String(), nil
}

// LastMarker returns the last offset in a ListResult.
func (r SnapshotPage) LastMarker() (string, error) {
	snapshots, err := ExtractSnapshots(r)
	if err != nil {
		return invalidMarker, err
	}
	if len(snapshots) == 0 {
		return invalidMarker, nil
	}

	u, err := url.Parse(r.URL.String())
	if err != nil {
		return invalidMarker, err
	}
	queryParams := u.Query()
	offset := queryParams.Get("offset")
	limit := queryParams.Get("limit")

	// Limit is not present, only one page required
	if limit == "" {
		return invalidMarker, nil
	}

	iOffset := 0
	if offset != "" {
		iOffset, err = strconv.Atoi(offset)
		if err != nil {
			return invalidMarker, err
		}
	}
	iLimit, err := strconv.Atoi(limit)
	if err != nil {
		return invalidMarker, err
	}
	iOffset = iOffset + iLimit
	offset = strconv.Itoa(iOffset)

	return offset, nil
}

// IsEmpty satisifies the IsEmpty method of the Page interface
func (r SnapshotPage) IsEmpty() (bool, error) {
	snapshots, err := ExtractSnapshots(r)
	return len(snapshots) == 0, err
}

// ExtractSnapshots extracts and returns a Snapshot slice. It is used while
// iterating over a snapshots.List call.
func ExtractSnapshots(r pagination.Page) ([]Snapshot, error) {
	var s struct {
		Snapshots []Snapshot `json:"snapshots"`
	}

	err := (r.(SnapshotPage)).ExtractInto(&s)

	return s.Snapshots, err
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

// IDFromName is a convenience function that returns a snapshot's ID given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	r, err := ListDetail(client, &ListOpts{Name: name}).AllPages()
	if err != nil {
		return "", err
	}

	ss, err := ExtractSnapshots(r)
	if err != nil {
		return "", err
	}

	switch len(ss) {
	case 0:
		return "", gophercloud.ErrResourceNotFound{Name: name, ResourceType: "snapshot"}
	case 1:
		return ss[0].ID, nil
	default:
		return "", gophercloud.ErrMultipleResourcesFound{Name: name, Count: len(ss), ResourceType: "snapshot"}
	}
}
