package images

import (
	"fmt"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToImageListQuery() (string, error)
}

// ListOpts contain options for limiting the number of Images returned from a call to ListDetail.
type ListOpts struct {
	// When the image last changed status (in date-time format).
	ChangesSince string `q:"changes-since"`
	// The number of Images to return.
	Limit int `q:"limit"`
	// UUID of the Image at which to set a marker.
	Marker string `q:"marker"`
	// The name of the Image.
	Name string `q:"name"`
	// The name of the Server (in URL format).
	Server string `q:"server"`
	// The current status of the Image.
	Status string `q:"status"`
	// The value of the type of image (e.g. BASE, SERVER, ALL)
	Type string `q:"type"`
}

// ToImageListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToImageListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
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

	createPage := func(r pagination.PageResult) pagination.Page {
		return ImagePage{pagination.LinkedPageBase{PageResult: r}}
	}

	return pagination.NewPager(client, url, createPage)
}

// Get acquires additional detail about a specific image by ID.
// Use ExtractImage() to interpret the result as an openstack Image.
func Get(client *gophercloud.ServiceClient, id string) GetResult {
	var result GetResult
	_, result.Err = client.Get(getURL(client, id), &result.Body, nil)
	return result
}

// Delete deletes the specified image ID.
func Delete(client *gophercloud.ServiceClient, id string) DeleteResult {
	var result DeleteResult
	_, result.Err = client.Delete(deleteURL(client, id), nil)
	return result
}

// IDFromName is a convienience function that returns an image's ID given its name.
func IDFromName(client *gophercloud.ServiceClient, name string) (string, error) {
	imageCount := 0
	imageID := ""
	if name == "" {
		return "", fmt.Errorf("An image name must be provided.")
	}
	pager := ListDetail(client, &ListOpts{
		Name: name,
	})
	pager.EachPage(func(page pagination.Page) (bool, error) {
		imageList, err := ExtractImages(page)
		if err != nil {
			return false, err
		}

		for _, i := range imageList {
			if i.Name == name {
				imageCount++
				imageID = i.ID
			}
		}
		return true, nil
	})

	switch imageCount {
	case 0:
		return "", fmt.Errorf("Unable to find image: %s", name)
	case 1:
		return imageID, nil
	default:
		return "", fmt.Errorf("Found %d images matching %s", imageCount, name)
	}
}
