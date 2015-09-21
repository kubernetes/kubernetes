package serviceassets

import (
	"strings"

	"github.com/rackspace/gophercloud"
)

// DeleteOptsBuilder allows extensions to add additional parameters to the Delete
// request.
type DeleteOptsBuilder interface {
	ToCDNAssetDeleteParams() (string, error)
}

// DeleteOpts is a structure that holds options for deleting CDN service assets.
type DeleteOpts struct {
	// If all is set to true, specifies that the delete occurs against all of the
	// assets for the service.
	All bool `q:"all"`
	// Specifies the relative URL of the asset to be deleted.
	URL string `q:"url"`
}

// ToCDNAssetDeleteParams formats a DeleteOpts into a query string.
func (opts DeleteOpts) ToCDNAssetDeleteParams() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// Delete accepts a unique service ID or URL and deletes the CDN service asset associated with
// it. For example, both "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0" and
// "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
// are valid options for idOrURL.
func Delete(c *gophercloud.ServiceClient, idOrURL string, opts DeleteOptsBuilder) DeleteResult {
	var url string
	if strings.Contains(idOrURL, "/") {
		url = idOrURL
	} else {
		url = deleteURL(c, idOrURL)
	}

	var res DeleteResult
	_, res.Err = c.Delete(url, nil)
	return res
}
