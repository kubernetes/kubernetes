package docker

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/textproto"
	"net/url"
	"path"
	"strconv"
	"strings"
	"time"

	"github.com/containerd/containerd/images"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/reference"
	"github.com/containerd/containerd/remotes"
	digest "github.com/opencontainers/go-digest"
	ocispec "github.com/opencontainers/image-spec/specs-go/v1"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context/ctxhttp"
)

var (
	// ErrNoToken is returned if a request is successful but the body does not
	// contain an authorization token.
	ErrNoToken = errors.New("authorization server did not include a token in the response")

	// ErrInvalidAuthorization is used when credentials are passed to a server but
	// those credentials are rejected.
	ErrInvalidAuthorization = errors.New("authorization failed")
)

type dockerResolver struct {
	credentials func(string) (string, string, error)
	plainHTTP   bool
	client      *http.Client
	tracker     StatusTracker
}

// ResolverOptions are used to configured a new Docker register resolver
type ResolverOptions struct {
	// Credentials provides username and secret given a host.
	// If username is empty but a secret is given, that secret
	// is interpretted as a long lived token.
	Credentials func(string) (string, string, error)

	// PlainHTTP specifies to use plain http and not https
	PlainHTTP bool

	// Client is the http client to used when making registry requests
	Client *http.Client

	// Tracker is used to track uploads to the registry. This is used
	// since the registry does not have upload tracking and the existing
	// mechanism for getting blob upload status is expensive.
	Tracker StatusTracker
}

// NewResolver returns a new resolver to a Docker registry
func NewResolver(options ResolverOptions) remotes.Resolver {
	tracker := options.Tracker
	if tracker == nil {
		tracker = NewInMemoryTracker()
	}
	return &dockerResolver{
		credentials: options.Credentials,
		plainHTTP:   options.PlainHTTP,
		client:      options.Client,
		tracker:     tracker,
	}
}

var _ remotes.Resolver = &dockerResolver{}

func (r *dockerResolver) Resolve(ctx context.Context, ref string) (string, ocispec.Descriptor, error) {
	refspec, err := reference.Parse(ref)
	if err != nil {
		return "", ocispec.Descriptor{}, err
	}

	if refspec.Object == "" {
		return "", ocispec.Descriptor{}, reference.ErrObjectRequired
	}

	base, err := r.base(refspec)
	if err != nil {
		return "", ocispec.Descriptor{}, err
	}

	fetcher := dockerFetcher{
		dockerBase: base,
	}

	var (
		urls []string
		dgst = refspec.Digest()
	)

	if dgst != "" {
		if err := dgst.Validate(); err != nil {
			// need to fail here, since we can't actually resolve the invalid
			// digest.
			return "", ocispec.Descriptor{}, err
		}

		// turns out, we have a valid digest, make a url.
		urls = append(urls, fetcher.url("manifests", dgst.String()))

		// fallback to blobs on not found.
		urls = append(urls, fetcher.url("blobs", dgst.String()))
	} else {
		urls = append(urls, fetcher.url("manifests", refspec.Object))
	}

	ctx, err = contextWithRepositoryScope(ctx, refspec, false)
	if err != nil {
		return "", ocispec.Descriptor{}, err
	}
	for _, u := range urls {
		req, err := http.NewRequest(http.MethodHead, u, nil)
		if err != nil {
			return "", ocispec.Descriptor{}, err
		}

		// set headers for all the types we support for resolution.
		req.Header.Set("Accept", strings.Join([]string{
			images.MediaTypeDockerSchema2Manifest,
			images.MediaTypeDockerSchema2ManifestList,
			ocispec.MediaTypeImageManifest,
			ocispec.MediaTypeImageIndex, "*"}, ", "))

		log.G(ctx).Debug("resolving")
		resp, err := fetcher.doRequestWithRetries(ctx, req, nil)
		if err != nil {
			return "", ocispec.Descriptor{}, err
		}
		resp.Body.Close() // don't care about body contents.

		if resp.StatusCode > 299 {
			if resp.StatusCode == http.StatusNotFound {
				continue
			}
			return "", ocispec.Descriptor{}, errors.Errorf("unexpected status code %v: %v", u, resp.Status)
		}

		// this is the only point at which we trust the registry. we use the
		// content headers to assemble a descriptor for the name. when this becomes
		// more robust, we mostly get this information from a secure trust store.
		dgstHeader := digest.Digest(resp.Header.Get("Docker-Content-Digest"))

		if dgstHeader != "" {
			if err := dgstHeader.Validate(); err != nil {
				return "", ocispec.Descriptor{}, errors.Wrapf(err, "%q in header not a valid digest", dgstHeader)
			}
			dgst = dgstHeader
		}

		if dgst == "" {
			return "", ocispec.Descriptor{}, errors.Errorf("could not resolve digest for %v", ref)
		}

		var (
			size       int64
			sizeHeader = resp.Header.Get("Content-Length")
		)

		size, err = strconv.ParseInt(sizeHeader, 10, 64)
		if err != nil {

			return "", ocispec.Descriptor{}, errors.Wrapf(err, "invalid size header: %q", sizeHeader)
		}
		if size < 0 {
			return "", ocispec.Descriptor{}, errors.Errorf("%q in header not a valid size", sizeHeader)
		}

		desc := ocispec.Descriptor{
			Digest:    dgst,
			MediaType: resp.Header.Get("Content-Type"), // need to strip disposition?
			Size:      size,
		}

		log.G(ctx).WithField("desc.digest", desc.Digest).Debug("resolved")
		return ref, desc, nil
	}

	return "", ocispec.Descriptor{}, errors.Errorf("%v not found", ref)
}

func (r *dockerResolver) Fetcher(ctx context.Context, ref string) (remotes.Fetcher, error) {
	refspec, err := reference.Parse(ref)
	if err != nil {
		return nil, err
	}

	base, err := r.base(refspec)
	if err != nil {
		return nil, err
	}

	return dockerFetcher{
		dockerBase: base,
	}, nil
}

func (r *dockerResolver) Pusher(ctx context.Context, ref string) (remotes.Pusher, error) {
	refspec, err := reference.Parse(ref)
	if err != nil {
		return nil, err
	}

	// Manifests can be pushed by digest like any other object, but the passed in
	// reference cannot take a digest without the associated content. A tag is allowed
	// and will be used to tag pushed manifests.
	if refspec.Object != "" && strings.Contains(refspec.Object, "@") {
		return nil, errors.New("cannot use digest reference for push locator")
	}

	base, err := r.base(refspec)
	if err != nil {
		return nil, err
	}

	return dockerPusher{
		dockerBase: base,
		tag:        refspec.Object,
		tracker:    r.tracker,
	}, nil
}

type dockerBase struct {
	refspec reference.Spec
	base    url.URL
	token   string

	client   *http.Client
	useBasic bool
	username string
	secret   string
}

func (r *dockerResolver) base(refspec reference.Spec) (*dockerBase, error) {
	var (
		err              error
		base             url.URL
		username, secret string
	)

	host := refspec.Hostname()
	base.Scheme = "https"

	if host == "docker.io" {
		base.Host = "registry-1.docker.io"
	} else {
		base.Host = host

		if r.plainHTTP || strings.HasPrefix(host, "localhost:") {
			base.Scheme = "http"
		}
	}

	if r.credentials != nil {
		username, secret, err = r.credentials(base.Host)
		if err != nil {
			return nil, err
		}
	}

	prefix := strings.TrimPrefix(refspec.Locator, host+"/")
	base.Path = path.Join("/v2", prefix)

	return &dockerBase{
		refspec:  refspec,
		base:     base,
		client:   r.client,
		username: username,
		secret:   secret,
	}, nil
}

func (r *dockerBase) url(ps ...string) string {
	url := r.base
	url.Path = path.Join(url.Path, path.Join(ps...))
	return url.String()
}

func (r *dockerBase) authorize(req *http.Request) {
	if r.useBasic {
		req.SetBasicAuth(r.username, r.secret)
	} else if r.token != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", r.token))
	}
}

func (r *dockerBase) doRequest(ctx context.Context, req *http.Request) (*http.Response, error) {
	ctx = log.WithLogger(ctx, log.G(ctx).WithField("url", req.URL.String()))
	log.G(ctx).WithField("request.headers", req.Header).WithField("request.method", req.Method).Debug("Do request")
	r.authorize(req)
	resp, err := ctxhttp.Do(ctx, r.client, req)
	if err != nil {
		return nil, errors.Wrap(err, "failed to do request")
	}
	log.G(ctx).WithFields(logrus.Fields{
		"status":           resp.Status,
		"response.headers": resp.Header,
	}).Debug("fetch response received")
	return resp, nil
}

func (r *dockerBase) doRequestWithRetries(ctx context.Context, req *http.Request, responses []*http.Response) (*http.Response, error) {
	resp, err := r.doRequest(ctx, req)
	if err != nil {
		return nil, err
	}

	responses = append(responses, resp)
	req, err = r.retryRequest(ctx, req, responses)
	if err != nil {
		resp.Body.Close()
		return nil, err
	}
	if req != nil {
		resp.Body.Close()
		return r.doRequestWithRetries(ctx, req, responses)
	}
	return resp, err
}

func (r *dockerBase) retryRequest(ctx context.Context, req *http.Request, responses []*http.Response) (*http.Request, error) {
	if len(responses) > 5 {
		return nil, nil
	}
	last := responses[len(responses)-1]
	if last.StatusCode == http.StatusUnauthorized {
		log.G(ctx).WithField("header", last.Header.Get("WWW-Authenticate")).Debug("Unauthorized")
		for _, c := range parseAuthHeader(last.Header) {
			if c.scheme == bearerAuth {
				if err := invalidAuthorization(c, responses); err != nil {
					r.token = ""
					return nil, err
				}
				if err := r.setTokenAuth(ctx, c.parameters); err != nil {
					return nil, err
				}
				return copyRequest(req)
			} else if c.scheme == basicAuth {
				if r.username != "" && r.secret != "" {
					r.useBasic = true
				}
				return copyRequest(req)
			}
		}
		return nil, nil
	} else if last.StatusCode == http.StatusMethodNotAllowed && req.Method == http.MethodHead {
		// Support registries which have not properly implemented the HEAD method for
		// manifests endpoint
		if strings.Contains(req.URL.Path, "/manifests/") {
			// TODO: copy request?
			req.Method = http.MethodGet
			return copyRequest(req)
		}
	}

	// TODO: Handle 50x errors accounting for attempt history
	return nil, nil
}

func invalidAuthorization(c challenge, responses []*http.Response) error {
	errStr := c.parameters["error"]
	if errStr == "" {
		return nil
	}

	n := len(responses)
	if n == 1 || (n > 1 && !sameRequest(responses[n-2].Request, responses[n-1].Request)) {
		return nil
	}

	return errors.Wrapf(ErrInvalidAuthorization, "server message: %s", errStr)
}

func sameRequest(r1, r2 *http.Request) bool {
	if r1.Method != r2.Method {
		return false
	}
	if *r1.URL != *r2.URL {
		return false
	}
	return true
}

func copyRequest(req *http.Request) (*http.Request, error) {
	ireq := *req
	if ireq.GetBody != nil {
		var err error
		ireq.Body, err = ireq.GetBody()
		if err != nil {
			return nil, err
		}
	}
	return &ireq, nil
}

func isManifestAccept(h http.Header) bool {
	for _, ah := range h[textproto.CanonicalMIMEHeaderKey("Accept")] {
		switch ah {
		case images.MediaTypeDockerSchema2Manifest:
			fallthrough
		case images.MediaTypeDockerSchema2ManifestList:
			fallthrough
		case ocispec.MediaTypeImageManifest:
			fallthrough
		case ocispec.MediaTypeImageIndex:
			return true
		}
	}
	return false
}

func (r *dockerBase) setTokenAuth(ctx context.Context, params map[string]string) error {
	realm, ok := params["realm"]
	if !ok {
		return errors.New("no realm specified for token auth challenge")
	}

	realmURL, err := url.Parse(realm)
	if err != nil {
		return fmt.Errorf("invalid token auth challenge realm: %s", err)
	}

	to := tokenOptions{
		realm:   realmURL.String(),
		service: params["service"],
	}

	to.scopes = getTokenScopes(ctx, params)
	if len(to.scopes) == 0 {
		return errors.Errorf("no scope specified for token auth challenge")
	}
	if r.secret != "" {
		// Credential information is provided, use oauth POST endpoint
		r.token, err = r.fetchTokenWithOAuth(ctx, to)
		if err != nil {
			return errors.Wrap(err, "failed to fetch oauth token")
		}
	} else {
		// Do request anonymously
		r.token, err = r.getToken(ctx, to)
		if err != nil {
			return errors.Wrap(err, "failed to fetch anonymous token")
		}
	}

	return nil
}

type tokenOptions struct {
	realm   string
	service string
	scopes  []string
}

type postTokenResponse struct {
	AccessToken  string    `json:"access_token"`
	RefreshToken string    `json:"refresh_token"`
	ExpiresIn    int       `json:"expires_in"`
	IssuedAt     time.Time `json:"issued_at"`
	Scope        string    `json:"scope"`
}

func (r *dockerBase) fetchTokenWithOAuth(ctx context.Context, to tokenOptions) (string, error) {
	form := url.Values{}
	form.Set("scope", strings.Join(to.scopes, " "))
	form.Set("service", to.service)
	// TODO: Allow setting client_id
	form.Set("client_id", "containerd-dist-tool")

	if r.username == "" {
		form.Set("grant_type", "refresh_token")
		form.Set("refresh_token", r.secret)
	} else {
		form.Set("grant_type", "password")
		form.Set("username", r.username)
		form.Set("password", r.secret)
	}

	resp, err := ctxhttp.PostForm(ctx, r.client, to.realm, form)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	// Registries without support for POST may return 404 for POST /v2/token.
	// As of September 2017, GCR is known to return 404.
	if (resp.StatusCode == 405 && r.username != "") || resp.StatusCode == 404 {
		return r.getToken(ctx, to)
	} else if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		b, _ := ioutil.ReadAll(io.LimitReader(resp.Body, 64000)) // 64KB
		log.G(ctx).WithFields(logrus.Fields{
			"status": resp.Status,
			"body":   string(b),
		}).Debugf("token request failed")
		// TODO: handle error body and write debug output
		return "", errors.Errorf("unexpected status: %s", resp.Status)
	}

	decoder := json.NewDecoder(resp.Body)

	var tr postTokenResponse
	if err = decoder.Decode(&tr); err != nil {
		return "", fmt.Errorf("unable to decode token response: %s", err)
	}

	return tr.AccessToken, nil
}

type getTokenResponse struct {
	Token        string    `json:"token"`
	AccessToken  string    `json:"access_token"`
	ExpiresIn    int       `json:"expires_in"`
	IssuedAt     time.Time `json:"issued_at"`
	RefreshToken string    `json:"refresh_token"`
}

// getToken fetches a token using a GET request
func (r *dockerBase) getToken(ctx context.Context, to tokenOptions) (string, error) {
	req, err := http.NewRequest("GET", to.realm, nil)
	if err != nil {
		return "", err
	}

	reqParams := req.URL.Query()

	if to.service != "" {
		reqParams.Add("service", to.service)
	}

	for _, scope := range to.scopes {
		reqParams.Add("scope", scope)
	}

	if r.secret != "" {
		req.SetBasicAuth(r.username, r.secret)
	}

	req.URL.RawQuery = reqParams.Encode()

	resp, err := ctxhttp.Do(ctx, r.client, req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 400 {
		// TODO: handle error body and write debug output
		return "", errors.Errorf("unexpected status: %s", resp.Status)
	}

	decoder := json.NewDecoder(resp.Body)

	var tr getTokenResponse
	if err = decoder.Decode(&tr); err != nil {
		return "", fmt.Errorf("unable to decode token response: %s", err)
	}

	// `access_token` is equivalent to `token` and if both are specified
	// the choice is undefined.  Canonicalize `access_token` by sticking
	// things in `token`.
	if tr.AccessToken != "" {
		tr.Token = tr.AccessToken
	}

	if tr.Token == "" {
		return "", ErrNoToken
	}

	return tr.Token, nil
}
