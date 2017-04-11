package v2

import (
	"net/http"
	"net/url"
	"strings"

	"github.com/docker/distribution/reference"
	"github.com/gorilla/mux"
)

// URLBuilder creates registry API urls from a single base endpoint. It can be
// used to create urls for use in a registry client or server.
//
// All urls will be created from the given base, including the api version.
// For example, if a root of "/foo/" is provided, urls generated will be fall
// under "/foo/v2/...". Most application will only provide a schema, host and
// port, such as "https://localhost:5000/".
type URLBuilder struct {
	root     *url.URL // url root (ie http://localhost/)
	router   *mux.Router
	relative bool
}

// NewURLBuilder creates a URLBuilder with provided root url object.
func NewURLBuilder(root *url.URL, relative bool) *URLBuilder {
	return &URLBuilder{
		root:     root,
		router:   Router(),
		relative: relative,
	}
}

// NewURLBuilderFromString workes identically to NewURLBuilder except it takes
// a string argument for the root, returning an error if it is not a valid
// url.
func NewURLBuilderFromString(root string, relative bool) (*URLBuilder, error) {
	u, err := url.Parse(root)
	if err != nil {
		return nil, err
	}

	return NewURLBuilder(u, relative), nil
}

// NewURLBuilderFromRequest uses information from an *http.Request to
// construct the root url.
func NewURLBuilderFromRequest(r *http.Request, relative bool) *URLBuilder {
	var scheme string

	forwardedProto := r.Header.Get("X-Forwarded-Proto")

	switch {
	case len(forwardedProto) > 0:
		scheme = forwardedProto
	case r.TLS != nil:
		scheme = "https"
	case len(r.URL.Scheme) > 0:
		scheme = r.URL.Scheme
	default:
		scheme = "http"
	}

	host := r.Host
	forwardedHost := r.Header.Get("X-Forwarded-Host")
	if len(forwardedHost) > 0 {
		// According to the Apache mod_proxy docs, X-Forwarded-Host can be a
		// comma-separated list of hosts, to which each proxy appends the
		// requested host. We want to grab the first from this comma-separated
		// list.
		hosts := strings.SplitN(forwardedHost, ",", 2)
		host = strings.TrimSpace(hosts[0])
	}

	basePath := routeDescriptorsMap[RouteNameBase].Path

	requestPath := r.URL.Path
	index := strings.Index(requestPath, basePath)

	u := &url.URL{
		Scheme: scheme,
		Host:   host,
	}

	if index > 0 {
		// N.B. index+1 is important because we want to include the trailing /
		u.Path = requestPath[0 : index+1]
	}

	return NewURLBuilder(u, relative)
}

// BuildBaseURL constructs a base url for the API, typically just "/v2/".
func (ub *URLBuilder) BuildBaseURL() (string, error) {
	route := ub.cloneRoute(RouteNameBase)

	baseURL, err := route.URL()
	if err != nil {
		return "", err
	}

	return baseURL.String(), nil
}

// BuildCatalogURL constructs a url get a catalog of repositories
func (ub *URLBuilder) BuildCatalogURL(values ...url.Values) (string, error) {
	route := ub.cloneRoute(RouteNameCatalog)

	catalogURL, err := route.URL()
	if err != nil {
		return "", err
	}

	return appendValuesURL(catalogURL, values...).String(), nil
}

// BuildTagsURL constructs a url to list the tags in the named repository.
func (ub *URLBuilder) BuildTagsURL(name reference.Named) (string, error) {
	route := ub.cloneRoute(RouteNameTags)

	tagsURL, err := route.URL("name", name.Name())
	if err != nil {
		return "", err
	}

	return tagsURL.String(), nil
}

// BuildManifestURL constructs a url for the manifest identified by name and
// reference. The argument reference may be either a tag or digest.
func (ub *URLBuilder) BuildManifestURL(ref reference.Named) (string, error) {
	route := ub.cloneRoute(RouteNameManifest)

	tagOrDigest := ""
	switch v := ref.(type) {
	case reference.Tagged:
		tagOrDigest = v.Tag()
	case reference.Digested:
		tagOrDigest = v.Digest().String()
	}

	manifestURL, err := route.URL("name", ref.Name(), "reference", tagOrDigest)
	if err != nil {
		return "", err
	}

	return manifestURL.String(), nil
}

// BuildBlobURL constructs the url for the blob identified by name and dgst.
func (ub *URLBuilder) BuildBlobURL(ref reference.Canonical) (string, error) {
	route := ub.cloneRoute(RouteNameBlob)

	layerURL, err := route.URL("name", ref.Name(), "digest", ref.Digest().String())
	if err != nil {
		return "", err
	}

	return layerURL.String(), nil
}

// BuildBlobUploadURL constructs a url to begin a blob upload in the
// repository identified by name.
func (ub *URLBuilder) BuildBlobUploadURL(name reference.Named, values ...url.Values) (string, error) {
	route := ub.cloneRoute(RouteNameBlobUpload)

	uploadURL, err := route.URL("name", name.Name())
	if err != nil {
		return "", err
	}

	return appendValuesURL(uploadURL, values...).String(), nil
}

// BuildBlobUploadChunkURL constructs a url for the upload identified by uuid,
// including any url values. This should generally not be used by clients, as
// this url is provided by server implementations during the blob upload
// process.
func (ub *URLBuilder) BuildBlobUploadChunkURL(name reference.Named, uuid string, values ...url.Values) (string, error) {
	route := ub.cloneRoute(RouteNameBlobUploadChunk)

	uploadURL, err := route.URL("name", name.Name(), "uuid", uuid)
	if err != nil {
		return "", err
	}

	return appendValuesURL(uploadURL, values...).String(), nil
}

// clondedRoute returns a clone of the named route from the router. Routes
// must be cloned to avoid modifying them during url generation.
func (ub *URLBuilder) cloneRoute(name string) clonedRoute {
	route := new(mux.Route)
	root := new(url.URL)

	*route = *ub.router.GetRoute(name) // clone the route
	*root = *ub.root

	return clonedRoute{Route: route, root: root, relative: ub.relative}
}

type clonedRoute struct {
	*mux.Route
	root     *url.URL
	relative bool
}

func (cr clonedRoute) URL(pairs ...string) (*url.URL, error) {
	routeURL, err := cr.Route.URL(pairs...)
	if err != nil {
		return nil, err
	}

	if cr.relative {
		return routeURL, nil
	}

	if routeURL.Scheme == "" && routeURL.User == nil && routeURL.Host == "" {
		routeURL.Path = routeURL.Path[1:]
	}

	url := cr.root.ResolveReference(routeURL)
	url.Scheme = cr.root.Scheme
	return url, nil
}

// appendValuesURL appends the parameters to the url.
func appendValuesURL(u *url.URL, values ...url.Values) *url.URL {
	merged := u.Query()

	for _, v := range values {
		for k, vv := range v {
			merged[k] = append(merged[k], vv...)
		}
	}

	u.RawQuery = merged.Encode()
	return u
}

// appendValues appends the parameters to the url. Panics if the string is not
// a url.
func appendValues(u string, values ...url.Values) string {
	up, err := url.Parse(u)

	if err != nil {
		panic(err) // should never happen
	}

	return appendValuesURL(up, values...).String()
}
