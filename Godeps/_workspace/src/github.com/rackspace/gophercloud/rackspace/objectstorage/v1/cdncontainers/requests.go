package cdncontainers

import (
	"github.com/racker/perigee"
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

	resp, err := perigee.Request("PUT", enableURL(c, containerName), perigee.Options{
		MoreHeaders: h,
		OkCodes:     []int{201, 202, 204},
	})
	res.Header = resp.HttpResponse.Header
	res.Err = err
	return res
}
