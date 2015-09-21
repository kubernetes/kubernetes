package gophercloud

import (
	"github.com/racker/perigee"
)

// See the CloudImagesProvider interface for details.
func (gsp *genericServersProvider) ListImages() ([]Image, error) {
	var is []Image

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/images/detail"
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results:      &struct{ Images *[]Image }{&is},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return is, err
}

func (gsp *genericServersProvider) ImageById(id string) (*Image, error) {
	var is *Image

	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/images/" + id
		return perigee.Get(url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			Results:      &struct{ Image **Image }{&is},
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
	})
	return is, err
}

func (gsp *genericServersProvider) DeleteImageById(id string) error {
	err := gsp.context.WithReauth(gsp.access, func() error {
		url := gsp.endpoint + "/images/" + id
		_, err := perigee.Request("DELETE", url, perigee.Options{
			CustomClient: gsp.context.httpClient,
			MoreHeaders: map[string]string{
				"X-Auth-Token": gsp.access.AuthToken(),
			},
		})
		return err
	})
	return err
}

// ImageLink provides a reference to a image by either ID or by direct URL.
// Some services use just the ID, others use just the URL.
// This structure provides a common means of expressing both in a single field.
type ImageLink struct {
	Id    string `json:"id"`
	Links []Link `json:"links"`
}

// Image is used for JSON (un)marshalling.
// It provides a description of an OS image.
//
// The Id field contains the image's unique identifier.
// For example, this identifier will be useful for specifying which operating system to install on a new server instance.
//
// The MinDisk and MinRam fields specify the minimum resources a server must provide to be able to install the image.
//
// The Name field provides a human-readable moniker for the OS image.
//
// The Progress and Status fields indicate image-creation status.
// Any usable image will have 100% progress.
//
// The Updated field indicates the last time this image was changed.
//
// OsDcfDiskConfig indicates the server's boot volume configuration.
// Valid values are:
//     AUTO
//     ----
//     The server is built with a single partition the size of the target flavor disk.
//     The file system is automatically adjusted to fit the entire partition.
//     This keeps things simple and automated.
//     AUTO is valid only for images and servers with a single partition that use the EXT3 file system.
//     This is the default setting for applicable Rackspace base images.
//
//     MANUAL
//     ------
//     The server is built using whatever partition scheme and file system is in the source image.
//     If the target flavor disk is larger,
//     the remaining disk space is left unpartitioned.
//     This enables images to have non-EXT3 file systems, multiple partitions, and so on,
//     and enables you to manage the disk configuration.
//
type Image struct {
	Created         string `json:"created"`
	Id              string `json:"id"`
	Links           []Link `json:"links"`
	MinDisk         int    `json:"minDisk"`
	MinRam          int    `json:"minRam"`
	Name            string `json:"name"`
	Progress        int    `json:"progress"`
	Status          string `json:"status"`
	Updated         string `json:"updated"`
	OsDcfDiskConfig string `json:"OS-DCF:diskConfig"`
}
