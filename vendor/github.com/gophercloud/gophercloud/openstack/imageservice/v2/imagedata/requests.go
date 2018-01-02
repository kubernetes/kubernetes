package imagedata

import (
	"io"
	"net/http"

	"github.com/gophercloud/gophercloud"
)

// Upload uploads an image file.
func Upload(client *gophercloud.ServiceClient, id string, data io.Reader) (r UploadResult) {
	_, r.Err = client.Put(uploadURL(client, id), data, nil, &gophercloud.RequestOpts{
		MoreHeaders: map[string]string{"Content-Type": "application/octet-stream"},
		OkCodes:     []int{204},
	})
	return
}

// Download retrieves an image.
func Download(client *gophercloud.ServiceClient, id string) (r DownloadResult) {
	var resp *http.Response
	resp, r.Err = client.Get(downloadURL(client, id), nil, nil)
	if resp != nil {
		r.Body = resp.Body
		r.Header = resp.Header
	}
	return
}
