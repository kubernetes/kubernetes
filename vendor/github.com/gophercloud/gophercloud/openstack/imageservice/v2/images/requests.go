package images

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToImageListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Filtering is achieved by passing in struct field values that map to
// the server attributes you want to see returned. Marker and Limit are used
// for pagination.
//
// http://developer.openstack.org/api-ref-image-v2.html
type ListOpts struct {
	// Integer value for the limit of values to return.
	Limit int `q:"limit"`

	// UUID of the server at which you want to set a marker.
	Marker string `q:"marker"`

	// Name filters on the name of the image.
	Name string `q:"name"`

	// Visibility filters on the visibility of the image.
	Visibility ImageVisibility `q:"visibility"`

	// MemberStatus filters on the member status of the image.
	MemberStatus ImageMemberStatus `q:"member_status"`

	// Owner filters on the project ID of the image.
	Owner string `q:"owner"`

	// Status filters on the status of the image.
	Status ImageStatus `q:"status"`

	// SizeMin filters on the size_min image property.
	SizeMin int64 `q:"size_min"`

	// SizeMax filters on the size_max image property.
	SizeMax int64 `q:"size_max"`

	// Sort sorts the results using the new style of sorting. See the OpenStack
	// Image API reference for the exact syntax.
	//
	// Sort cannot be used with the classic sort options (sort_key and sort_dir).
	Sort string `q:"sort"`

	// SortKey will sort the results based on a specified image property.
	SortKey string `q:"sort_key"`

	// SortDir will sort the list results either ascending or decending.
	SortDir string `q:"sort_dir"`
	Tag     string `q:"tag"`
}

// ToImageListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToImageListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// List implements image list request.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToImageListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(c, url, func(r pagination.PageResult) pagination.Page {
		return ImagePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// CreateOptsBuilder allows extensions to add parameters to the Create request.
type CreateOptsBuilder interface {
	// Returns value that can be passed to json.Marshal
	ToImageCreateMap() (map[string]interface{}, error)
}

// CreateOpts represents options used to create an image.
type CreateOpts struct {
	// Name is the name of the new image.
	Name string `json:"name" required:"true"`

	// Id is the the image ID.
	ID string `json:"id,omitempty"`

	// Visibility defines who can see/use the image.
	Visibility *ImageVisibility `json:"visibility,omitempty"`

	// Tags is a set of image tags.
	Tags []string `json:"tags,omitempty"`

	// ContainerFormat is the format of the
	// container. Valid values are ami, ari, aki, bare, and ovf.
	ContainerFormat string `json:"container_format,omitempty"`

	// DiskFormat is the format of the disk. If set,
	// valid values are ami, ari, aki, vhd, vmdk, raw, qcow2, vdi,
	// and iso.
	DiskFormat string `json:"disk_format,omitempty"`

	// MinDisk is the amount of disk space in
	// GB that is required to boot the image.
	MinDisk int `json:"min_disk,omitempty"`

	// MinRAM is the amount of RAM in MB that
	// is required to boot the image.
	MinRAM int `json:"min_ram,omitempty"`

	// protected is whether the image is not deletable.
	Protected *bool `json:"protected,omitempty"`

	// properties is a set of properties, if any, that
	// are associated with the image.
	Properties map[string]string `json:"-"`
}

// ToImageCreateMap assembles a request body based on the contents of
// a CreateOpts.
func (opts CreateOpts) ToImageCreateMap() (map[string]interface{}, error) {
	b, err := gophercloud.BuildRequestBody(opts, "")
	if err != nil {
		return nil, err
	}

	if opts.Properties != nil {
		for k, v := range opts.Properties {
			b[k] = v
		}
	}
	return b, nil
}

// Create implements create image request.
func Create(client *gophercloud.ServiceClient, opts CreateOptsBuilder) (r CreateResult) {
	b, err := opts.ToImageCreateMap()
	if err != nil {
		r.Err = err
		return r
	}
	_, r.Err = client.Post(createURL(client), b, &r.Body, &gophercloud.RequestOpts{OkCodes: []int{201}})
	return
}

// Delete implements image delete request.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// Get implements image get request.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// Update implements image updated request.
func Update(client *gophercloud.ServiceClient, id string, opts UpdateOptsBuilder) (r UpdateResult) {
	b, err := opts.ToImageUpdateMap()
	if err != nil {
		r.Err = err
		return r
	}
	_, r.Err = client.Patch(updateURL(client, id), b, &r.Body, &gophercloud.RequestOpts{
		OkCodes:     []int{200},
		MoreHeaders: map[string]string{"Content-Type": "application/openstack-images-v2.1-json-patch"},
	})
	return
}

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	// returns value implementing json.Marshaler which when marshaled matches
	// the patch schema:
	// http://specs.openstack.org/openstack/glance-specs/specs/api/v2/http-patch-image-api-v2.html
	ToImageUpdateMap() ([]interface{}, error)
}

// UpdateOpts implements UpdateOpts
type UpdateOpts []Patch

// ToImageUpdateMap assembles a request body based on the contents of
// UpdateOpts.
func (opts UpdateOpts) ToImageUpdateMap() ([]interface{}, error) {
	m := make([]interface{}, len(opts))
	for i, patch := range opts {
		patchJSON := patch.ToImagePatchMap()
		m[i] = patchJSON
	}
	return m, nil
}

// Patch represents a single update to an existing image. Multiple updates
// to an image can be submitted at the same time.
type Patch interface {
	ToImagePatchMap() map[string]interface{}
}

// UpdateVisibility represents an updated visibility property request.
type UpdateVisibility struct {
	Visibility ImageVisibility
}

// ToImagePatchMap assembles a request body based on UpdateVisibility.
func (u UpdateVisibility) ToImagePatchMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  "/visibility",
		"value": u.Visibility,
	}
}

// ReplaceImageName represents an updated image_name property request.
type ReplaceImageName struct {
	NewName string
}

// ToImagePatchMap assembles a request body based on ReplaceImageName.
func (r ReplaceImageName) ToImagePatchMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  "/name",
		"value": r.NewName,
	}
}

// ReplaceImageChecksum represents an updated checksum property request.
type ReplaceImageChecksum struct {
	Checksum string
}

// ReplaceImageChecksum assembles a request body based on ReplaceImageChecksum.
func (rc ReplaceImageChecksum) ToImagePatchMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  "/checksum",
		"value": rc.Checksum,
	}
}

// ReplaceImageTags represents an updated tags property request.
type ReplaceImageTags struct {
	NewTags []string
}

// ToImagePatchMap assembles a request body based on ReplaceImageTags.
func (r ReplaceImageTags) ToImagePatchMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  "/tags",
		"value": r.NewTags,
	}
}
