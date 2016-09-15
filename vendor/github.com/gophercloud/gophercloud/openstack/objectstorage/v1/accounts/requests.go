package accounts

import "github.com/gophercloud/gophercloud"

// GetOptsBuilder allows extensions to add additional headers to the Get
// request.
type GetOptsBuilder interface {
	ToAccountGetMap() (map[string]string, error)
}

// GetOpts is a structure that contains parameters for getting an account's
// metadata.
type GetOpts struct {
	Newest bool `h:"X-Newest"`
}

// ToAccountGetMap formats a GetOpts into a map[string]string of headers.
func (opts GetOpts) ToAccountGetMap() (map[string]string, error) {
	return gophercloud.BuildHeaders(opts)
}

// Get is a function that retrieves an account's metadata. To extract just the
// custom metadata, call the ExtractMetadata method on the GetResult. To extract
// all the headers that are returned (including the metadata), call the
// ExtractHeader method on the GetResult.
func Get(c *gophercloud.ServiceClient, opts GetOptsBuilder) (r GetResult) {
	h := make(map[string]string)
	if opts != nil {
		headers, err := opts.ToAccountGetMap()
		if err != nil {
			r.Err = err
			return
		}
		for k, v := range headers {
			h[k] = v
		}
	}
	resp, err := c.Request("HEAD", getURL(c), &gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{204},
	})
	if resp != nil {
		r.Header = resp.Header
	}
	r.Err = err
	return
}

// UpdateOptsBuilder allows extensions to add additional headers to the Update
// request.
type UpdateOptsBuilder interface {
	ToAccountUpdateMap() (map[string]string, error)
}

// UpdateOpts is a structure that contains parameters for updating, creating, or
// deleting an account's metadata.
type UpdateOpts struct {
	Metadata          map[string]string
	ContentType       string `h:"Content-Type"`
	DetectContentType bool   `h:"X-Detect-Content-Type"`
	TempURLKey        string `h:"X-Account-Meta-Temp-URL-Key"`
	TempURLKey2       string `h:"X-Account-Meta-Temp-URL-Key-2"`
}

// ToAccountUpdateMap formats an UpdateOpts into a map[string]string of headers.
func (opts UpdateOpts) ToAccountUpdateMap() (map[string]string, error) {
	headers, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	for k, v := range opts.Metadata {
		headers["X-Account-Meta-"+k] = v
	}
	return headers, err
}

// Update is a function that creates, updates, or deletes an account's metadata.
// To extract the headers returned, call the Extract method on the UpdateResult.
func Update(c *gophercloud.ServiceClient, opts UpdateOptsBuilder) (r UpdateResult) {
	h := make(map[string]string)
	if opts != nil {
		headers, err := opts.ToAccountUpdateMap()
		if err != nil {
			r.Err = err
			return
		}
		for k, v := range headers {
			h[k] = v
		}
	}
	resp, err := c.Request("POST", updateURL(c), &gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{201, 202, 204},
	})
	if resp != nil {
		r.Header = resp.Header
	}
	r.Err = err
	return
}
