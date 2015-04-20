package services

import (
	"fmt"
	"strings"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/pagination"
)

// ListOptsBuilder allows extensions to add additional parameters to the
// List request.
type ListOptsBuilder interface {
	ToCDNServiceListQuery() (string, error)
}

// ListOpts allows the filtering and sorting of paginated collections through
// the API. Marker and Limit are used for pagination.
type ListOpts struct {
	Marker string `q:"marker"`
	Limit  int    `q:"limit"`
}

// ToCDNServiceListQuery formats a ListOpts into a query string.
func (opts ListOpts) ToCDNServiceListQuery() (string, error) {
	q, err := gophercloud.BuildQueryString(opts)
	if err != nil {
		return "", err
	}
	return q.String(), nil
}

// List returns a Pager which allows you to iterate over a collection of
// CDN services. It accepts a ListOpts struct, which allows for pagination via
// marker and limit.
func List(c *gophercloud.ServiceClient, opts ListOptsBuilder) pagination.Pager {
	url := listURL(c)
	if opts != nil {
		query, err := opts.ToCDNServiceListQuery()
		if err != nil {
			return pagination.Pager{Err: err}
		}
		url += query
	}

	createPage := func(r pagination.PageResult) pagination.Page {
		p := ServicePage{pagination.MarkerPageBase{PageResult: r}}
		p.MarkerPageBase.Owner = p
		return p
	}

	pager := pagination.NewPager(c, url, createPage)
	return pager
}

// CreateOptsBuilder is the interface options structs have to satisfy in order
// to be used in the main Create operation in this package. Since many
// extensions decorate or modify the common logic, it is useful for them to
// satisfy a basic interface in order for them to be used.
type CreateOptsBuilder interface {
	ToCDNServiceCreateMap() (map[string]interface{}, error)
}

// CreateOpts is the common options struct used in this package's Create
// operation.
type CreateOpts struct {
	// REQUIRED. Specifies the name of the service. The minimum length for name is
	// 3. The maximum length is 256.
	Name string
	// REQUIRED. Specifies a list of domains used by users to access their website.
	Domains []Domain
	// REQUIRED. Specifies a list of origin domains or IP addresses where the
	// original assets are stored.
	Origins []Origin
	// REQUIRED. Specifies the CDN provider flavor ID to use. For a list of
	// flavors, see the operation to list the available flavors. The minimum
	// length for flavor_id is 1. The maximum length is 256.
	FlavorID string
	// OPTIONAL. Specifies the TTL rules for the assets under this service. Supports wildcards for fine-grained control.
	Caching []CacheRule
	// OPTIONAL. Specifies the restrictions that define who can access assets (content from the CDN cache).
	Restrictions []Restriction
}

// ToCDNServiceCreateMap casts a CreateOpts struct to a map.
func (opts CreateOpts) ToCDNServiceCreateMap() (map[string]interface{}, error) {
	s := make(map[string]interface{})

	if opts.Name == "" {
		return nil, no("Name")
	}
	s["name"] = opts.Name

	if opts.Domains == nil {
		return nil, no("Domains")
	}
	for _, domain := range opts.Domains {
		if domain.Domain == "" {
			return nil, no("Domains[].Domain")
		}
	}
	s["domains"] = opts.Domains

	if opts.Origins == nil {
		return nil, no("Origins")
	}
	for _, origin := range opts.Origins {
		if origin.Origin == "" {
			return nil, no("Origins[].Origin")
		}
		if origin.Rules == nil && len(opts.Origins) > 1 {
			return nil, no("Origins[].Rules")
		}
		for _, rule := range origin.Rules {
			if rule.Name == "" {
				return nil, no("Origins[].Rules[].Name")
			}
			if rule.RequestURL == "" {
				return nil, no("Origins[].Rules[].RequestURL")
			}
		}
	}
	s["origins"] = opts.Origins

	if opts.FlavorID == "" {
		return nil, no("FlavorID")
	}
	s["flavor_id"] = opts.FlavorID

	if opts.Caching != nil {
		for _, cache := range opts.Caching {
			if cache.Name == "" {
				return nil, no("Caching[].Name")
			}
			if cache.Rules != nil {
				for _, rule := range cache.Rules {
					if rule.Name == "" {
						return nil, no("Caching[].Rules[].Name")
					}
					if rule.RequestURL == "" {
						return nil, no("Caching[].Rules[].RequestURL")
					}
				}
			}
		}
		s["caching"] = opts.Caching
	}

	if opts.Restrictions != nil {
		for _, restriction := range opts.Restrictions {
			if restriction.Name == "" {
				return nil, no("Restrictions[].Name")
			}
			if restriction.Rules != nil {
				for _, rule := range restriction.Rules {
					if rule.Name == "" {
						return nil, no("Restrictions[].Rules[].Name")
					}
				}
			}
		}
		s["restrictions"] = opts.Restrictions
	}

	return s, nil
}

// Create accepts a CreateOpts struct and creates a new CDN service using the
// values provided.
func Create(c *gophercloud.ServiceClient, opts CreateOptsBuilder) CreateResult {
	var res CreateResult

	reqBody, err := opts.ToCDNServiceCreateMap()
	if err != nil {
		res.Err = err
		return res
	}

	// Send request to API
	resp, err := c.Post(createURL(c), &reqBody, nil, nil)
	res.Header = resp.Header
	res.Err = err
	return res
}

// Get retrieves a specific service based on its URL or its unique ID. For
// example, both "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0" and
// "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
// are valid options for idOrURL.
func Get(c *gophercloud.ServiceClient, idOrURL string) GetResult {
	var url string
	if strings.Contains(idOrURL, "/") {
		url = idOrURL
	} else {
		url = getURL(c, idOrURL)
	}

	var res GetResult
	_, res.Err = c.Get(url, &res.Body, nil)
	return res
}

// Path is a JSON pointer location that indicates which service parameter is being added, replaced,
// or removed.
type Path struct {
	baseElement string
}

func (p Path) renderRoot() string {
	return "/" + p.baseElement
}

func (p Path) renderDash() string {
	return fmt.Sprintf("/%s/-", p.baseElement)
}

func (p Path) renderIndex(index int64) string {
	return fmt.Sprintf("/%s/%d", p.baseElement, index)
}

var (
	// PathDomains indicates that an update operation is to be performed on a Domain.
	PathDomains = Path{baseElement: "domains"}

	// PathOrigins indicates that an update operation is to be performed on an Origin.
	PathOrigins = Path{baseElement: "origins"}

	// PathCaching indicates that an update operation is to be performed on a CacheRule.
	PathCaching = Path{baseElement: "caching"}
)

type value interface {
	toPatchValue() interface{}
	appropriatePath() Path
	renderRootOr(func(p Path) string) string
}

// Patch represents a single update to an existing Service. Multiple updates to a service can be
// submitted at the same time.
type Patch interface {
	ToCDNServiceUpdateMap() map[string]interface{}
}

// Insertion is a Patch that requests the addition of a value (Domain, Origin, or CacheRule) to
// a Service at a fixed index. Use an Append instead to append the new value to the end of its
// collection. Pass it to the Update function as part of the Patch slice.
type Insertion struct {
	Index int64
	Value value
}

// ToCDNServiceUpdateMap converts an Insertion into a request body fragment suitable for the
// Update call.
func (i Insertion) ToCDNServiceUpdateMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "add",
		"path":  i.Value.renderRootOr(func(p Path) string { return p.renderIndex(i.Index) }),
		"value": i.Value.toPatchValue(),
	}
}

// Append is a Patch that requests the addition of a value (Domain, Origin, or CacheRule) to a
// Service at the end of its respective collection. Use an Insertion instead to insert the value
// at a fixed index within the collection. Pass this to the Update function as part of its
// Patch slice.
type Append struct {
	Value value
}

// ToCDNServiceUpdateMap converts an Append into a request body fragment suitable for the
// Update call.
func (a Append) ToCDNServiceUpdateMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "add",
		"path":  a.Value.renderRootOr(func(p Path) string { return p.renderDash() }),
		"value": a.Value.toPatchValue(),
	}
}

// Replacement is a Patch that alters a specific service parameter (Domain, Origin, or CacheRule)
// in-place by index. Pass it to the Update function as part of the Patch slice.
type Replacement struct {
	Value value
	Index int64
}

// ToCDNServiceUpdateMap converts a Replacement into a request body fragment suitable for the
// Update call.
func (r Replacement) ToCDNServiceUpdateMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  r.Value.renderRootOr(func(p Path) string { return p.renderIndex(r.Index) }),
		"value": r.Value.toPatchValue(),
	}
}

// NameReplacement specifically updates the Service name. Pass it to the Update function as part
// of the Patch slice.
type NameReplacement struct {
	NewName string
}

// ToCDNServiceUpdateMap converts a NameReplacement into a request body fragment suitable for the
// Update call.
func (r NameReplacement) ToCDNServiceUpdateMap() map[string]interface{} {
	return map[string]interface{}{
		"op":    "replace",
		"path":  "/name",
		"value": r.NewName,
	}
}

// Removal is a Patch that requests the removal of a service parameter (Domain, Origin, or
// CacheRule) by index. Pass it to the Update function as part of the Patch slice.
type Removal struct {
	Path  Path
	Index int64
	All   bool
}

// ToCDNServiceUpdateMap converts a Removal into a request body fragment suitable for the
// Update call.
func (r Removal) ToCDNServiceUpdateMap() map[string]interface{} {
	result := map[string]interface{}{"op": "remove"}
	if r.All {
		result["path"] = r.Path.renderRoot()
	} else {
		result["path"] = r.Path.renderIndex(r.Index)
	}
	return result
}

type UpdateOpts []Patch

// Update accepts a slice of Patch operations (Insertion, Append, Replacement or Removal) and
// updates an existing CDN service using the values provided. idOrURL can be either the service's
// URL or its ID. For example, both "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0" and
// "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
// are valid options for idOrURL.
func Update(c *gophercloud.ServiceClient, idOrURL string, opts UpdateOpts) UpdateResult {
	var url string
	if strings.Contains(idOrURL, "/") {
		url = idOrURL
	} else {
		url = updateURL(c, idOrURL)
	}

	reqBody := make([]map[string]interface{}, len(opts))
	for i, patch := range opts {
		reqBody[i] = patch.ToCDNServiceUpdateMap()
	}

	resp, err := c.Request("PATCH", url, gophercloud.RequestOpts{
		JSONBody: &reqBody,
		OkCodes:  []int{202},
	})
	var result UpdateResult
	result.Header = resp.Header
	result.Err = err
	return result
}

// Delete accepts a service's ID or its URL and deletes the CDN service
// associated with it. For example, both "96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0" and
// "https://global.cdn.api.rackspacecloud.com/v1.0/services/96737ae3-cfc1-4c72-be88-5d0e7cc9a3f0"
// are valid options for idOrURL.
func Delete(c *gophercloud.ServiceClient, idOrURL string) DeleteResult {
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
