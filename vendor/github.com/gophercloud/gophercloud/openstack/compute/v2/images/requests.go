package images

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// ListDetail request.
type ListOptsBuilder interface {
	ToImageListQuery() (string, error)
}

// ListOpts contain options filtering Images returned from a call to ListDetail.
type ListOpts struct {
	// ChangesSince filters Images based on the last changed status (in date-time
	// format).
	ChangesSince string `q:"changes-since"`

	// Limit limits the number of Images to return.
	Limit int `q:"limit"`

	// Mark is an Image UUID at which to set a marker.
	Marker string `q:"marker"`

	// Name is the name of the Image.
	Name string `q:"name"`

	// Server is the name of the Server (in URL format).
	Server string `q:"server"`

	// Status is the current status of the Image.
	Status string `q:"status"`

	// Type is the type of image (e.g. BASE, SERVER, ALL).
	Type string `q:"type"`
}

// ToImageListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToImageListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	return q.String(), err
}

// ListDetail enumerates the available images.
func ListDetail(client *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listDetailURL(client)
	if opts != nil {
		query, err := opts.ToImageListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}
	return pagination.NewPager(client, url, func(r pagination.PageResult) pagination.Page {
		return ImagePage{pagination.LinkedPageBase{PageResult: r}}
	})
}

// Get returns data about a specific image by its ID.
func Get(client *gophercloud.ServiceClient, id string) (r GetResult) {
	_, r.Err = client.Get(getURL(client, id), &r.Body, nil)
	return
}

// Delete deletes the specified image ID.
func Delete(client *gophercloud.ServiceClient, id string) (r DeleteResult) {
	_, r.Err = client.Delete(deleteURL(client, id), nil)
	return
}

// IDFromName is a convienience function that returns an image's ID given its
// name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	count := 0
	id := ""
	allPages, err := ListDetail(client, nil).AllPages()
	if err != nil {
		return "", err
	}

	all, err := ExtractImages(allPages)
	if err != nil {
		return "", err
	}

	for _, f := range all {
		if f.Name == name {
			count++
			id = f.ID
		}
	}

	switch count {
	case 0:
		err := &gophercloud.ErrResourceNotFound{}
		err.ResourceType = "image"
		err.Name = name
		return "", err
	case 1:
		return id, nil
	default:
		err := &gophercloud.ErrMultipleResourcesFound{}
		err.ResourceType = "image"
		err.Name = name
		err.Count = count
		return "", err
	}
}
