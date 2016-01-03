package cdncontainers

import (
	"strconv"

	"github.com/rackspace/gophercloud"
)

// EnableOptsBuilder allows extensions to add additional parameters to the Enable
// request.
type EnableOptsBuilder interface {
	ToCDNContainerEnableMap() (map[string]string, error)
}

// EnableOpts is a structure that holds options for enabling a CDN container.
type EnableOpts struct {
	// CDNEnabled indicates whether or not the container is CDN enabled. Set to
	// `true` to enable the container. Note that changing this setting from true
	// to false will disable the container in the CDN but only after the TTL has
	// expired.
	CDNEnabled bool `h:"X-Cdn-Enabled"`
	// TTL is the time-to-live for the container (in seconds).
	TTL int `h:"X-Ttl"`
}

// ToCDNContainerEnableMap formats an EnableOpts into a map of headers.
func (opts EnableOpts) ToCDNContainerEnableMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	return h, nil
}

// Enable is a function that enables/disables a CDN container.
func Enable(c *gophercloud.ServiceClient, containerName string, opts EnableOptsBuilder) EnableResult {
	var res EnableResult
	h := c.AuthenticatedHeaders()

	if opts != nil {
		headers, err := opts.ToCDNContainerEnableMap()
		if err != nil {
			res.Err = err
			return res
		}

		for k, v := range headers {
			h[k] = v
		}
	}

	resp, err := c.Request("PUT", enableURL(c, containerName), gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{201, 202, 204},
	})
	if resp != nil {
		res.Header = resp.Header
	}
	res.Err = err
	return res
}

// Get is a function that retrieves the metadata of a container. To extract just
// the custom metadata, pass the GetResult response to the ExtractMetadata
// function.
func Get(c *gophercloud.ServiceClient, containerName string) GetResult {
	var res GetResult
	resp, err := c.Request("HEAD", getURL(c, containerName), gophercloud.RequestOpts{
		OkCodes: []int{200, 204},
	})
	if resp != nil {
		res.Header = resp.Header
	}
	res.Err = err
	return res
}

// State is the state of an option. It is a pointer to a boolean to enable checking for
// a zero-value of nil instead of false, which is a valid option.
type State *bool

var (
	iTrue  = true
	iFalse = false

	// Enabled is used for a true value for options in request bodies.
	Enabled State = &iTrue
	// Disabled is used for a false value for options in request bodies.
	Disabled State = &iFalse
)

// UpdateOptsBuilder allows extensions to add additional parameters to the
// Update request.
type UpdateOptsBuilder interface {
	ToContainerUpdateMap() (map[string]string, error)
}

// UpdateOpts is a structure that holds parameters for updating, creating, or
// deleting a container's metadata.
type UpdateOpts struct {
	// Whether or not to CDN-enable a container. Prefer using XCDNEnabled, which
	// is of type *bool underneath.
	// TODO v2.0: change type to Enabled/Disabled (*bool)
	CDNEnabled bool `h:"X-Cdn-Enabled"`
	// Whether or not to enable log retention. Prefer using XLogRetention, which
	// is of type *bool underneath.
	// TODO v2.0: change type to Enabled/Disabled (*bool)
	LogRetention  bool `h:"X-Log-Retention"`
	XCDNEnabled   *bool
	XLogRetention *bool
	TTL           int `h:"X-Ttl"`
}

// ToContainerUpdateMap formats a CreateOpts into a map of headers.
func (opts UpdateOpts) ToContainerUpdateMap() (map[string]string, error) {
	h, err := gophercloud.BuildHeaders(opts)
	if err != nil {
		return nil, err
	}
	h["X-Cdn-Enabled"] = strconv.FormatBool(opts.CDNEnabled)
	h["X-Log-Retention"] = strconv.FormatBool(opts.LogRetention)

	if opts.XCDNEnabled != nil {
		h["X-Cdn-Enabled"] = strconv.FormatBool(*opts.XCDNEnabled)
	}

	if opts.XLogRetention != nil {
		h["X-Log-Retention"] = strconv.FormatBool(*opts.XLogRetention)
	}

	return h, nil
}

// Update is a function that creates, updates, or deletes a container's
// metadata.
func Update(c *gophercloud.ServiceClient, containerName string, opts UpdateOptsBuilder) UpdateResult {
	var res UpdateResult
	h := c.AuthenticatedHeaders()

	if opts != nil {
		headers, err := opts.ToContainerUpdateMap()
		if err != nil {
			res.Err = err
			return res
		}

		for k, v := range headers {
			h[k] = v
		}
	}

	resp, err := c.Request("POST", updateURL(c, containerName), gophercloud.RequestOpts{
		MoreHeaders: h,
		OkCodes:     []int{202, 204},
	})
	if resp != nil {
		res.Header = resp.Header
	}
	res.Err = err
	return res
}
