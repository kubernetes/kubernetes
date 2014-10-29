package bulk

import (
	"net/url"
	"strings"

	"github.com/racker/perigee"
	"github.com/rackspace/gophercloud"
)

// DeleteOptsBuilder allows extensions to add additional parameters to the
// Delete request.
type DeleteOptsBuilder interface {
	ToBulkDeleteBody() (string, error)
}

// DeleteOpts is a structure that holds parameters for deleting an object.
type DeleteOpts []string

// ToBulkDeleteBody formats a DeleteOpts into a request body.
func (opts DeleteOpts) ToBulkDeleteBody() (string, error) {
	return url.QueryEscape(strings.Join(opts, "\n")), nil
}

// Delete will delete objects or containers in bulk.
func Delete(c *gophercloud.ServiceClient, opts DeleteOptsBuilder) DeleteResult {
	var res DeleteResult

	if opts == nil {
		return res
	}

	reqString, err := opts.ToBulkDeleteBody()
	if err != nil {
		res.Err = err
		return res
	}

	reqBody := strings.NewReader(reqString)

	resp, err := perigee.Request("DELETE", deleteURL(c), perigee.Options{
		ContentType: "text/plain",
		MoreHeaders: c.AuthenticatedHeaders(),
		OkCodes:     []int{200},
		ReqBody:     reqBody,
		Results:     &res.Body,
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}
