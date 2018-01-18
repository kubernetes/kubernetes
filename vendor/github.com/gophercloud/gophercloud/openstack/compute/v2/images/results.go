package images

import (
	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/pagination"
)

// GetResult is the response from a Get operation. Call its Extract method to
// interpret it as an Image.
type GetResult struct {
	gophercloud.Result
}

// DeleteResult is the result from a Delete operation. Call its ExtractErr
// method to determine if the call succeeded or failed.
type DeleteResult struct {
	gophercloud.ErrResult
}

// Extract interprets a GetResult as an Image.
func (r GetResult) Extract() (*Image, error) {
	var s struct {
		Image *Image `json:"image"`
	}
	err := r.ExtractInto(&s)
	return s.Image, err
}

// Image represents an Image returned by the Compute API.
type Image struct {
	// ID is the unique ID of an image.
	ID string

	// Created is the date when the image was created.
	Created string

	// MinDisk is the minimum amount of disk a flavor must have to be able
	// to create a server based on the image, measured in GB.
	MinDisk int

	// MinRAM is the minimum amount of RAM a flavor must have to be able
	// to create a server based on the image, measured in MB.
	MinRAM int

	// Name provides a human-readable moniker for the OS image.
	Name string

	// The Progress and Status fields indicate image-creation status.
	Progress int

	// Status is the current status of the image.
	Status string

	// Update is the date when the image was updated.
	Updated string

	// Metadata provides free-form key/value pairs that further describe the
	// image.
	Metadata map[string]interface{}
}

// ImagePage contains a single page of all Images returne from a ListDetail
// operation. Use ExtractImages to convert it into a slice of usable structs.
type ImagePage struct {
	pagination.LinkedPageBase
}

// IsEmpty returns true if an ImagePage contains no Image results.
func (page ImagePage) IsEmpty() (bool, error) {
	images, err := ExtractImages(page)
	return len(images) == 0, err
}

// NextPageURL uses the response's embedded link reference to navigate to the
// next page of results.
func (page ImagePage) NextPageURL() (string, error) {
	var s struct {
		Links []gophercloud.Link `json:"images_links"`
	}
	err := page.ExtractInto(&s)
	if err != nil {
		return "", err
	}
	return gophercloud.ExtractNextURL(s.Links)
}

// ExtractImages converts a page of List results into a slice of usable Image
// structs.
func ExtractImages(r pagination.Page) ([]Image, error) {
	var s struct {
		Images []Image `json:"images"`
	}
	err := (r.(ImagePage)).ExtractInto(&s)
	return s.Images, err
}
