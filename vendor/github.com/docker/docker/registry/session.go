package registry

import (
	"bytes"
	"crypto/sha256"
	"errors"
	"sync"
	// this is required for some certificates
	_ "crypto/sha512"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"net/http"
	"net/http/cookiejar"
	"net/url"
	"strconv"
	"strings"

	"github.com/docker/distribution/reference"
	"github.com/docker/distribution/registry/api/errcode"
	"github.com/docker/docker/api/types"
	registrytypes "github.com/docker/docker/api/types/registry"
	"github.com/docker/docker/pkg/ioutils"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/tarsum"
	"github.com/docker/docker/registry/resumable"
	"github.com/sirupsen/logrus"
)

var (
	// ErrRepoNotFound is returned if the repository didn't exist on the
	// remote side
	ErrRepoNotFound = errors.New("Repository not found")
)

// A Session is used to communicate with a V1 registry
type Session struct {
	indexEndpoint *V1Endpoint
	client        *http.Client
	// TODO(tiborvass): remove authConfig
	authConfig *types.AuthConfig
	id         string
}

type authTransport struct {
	http.RoundTripper
	*types.AuthConfig

	alwaysSetBasicAuth bool
	token              []string

	mu     sync.Mutex                      // guards modReq
	modReq map[*http.Request]*http.Request // original -> modified
}

// AuthTransport handles the auth layer when communicating with a v1 registry (private or official)
//
// For private v1 registries, set alwaysSetBasicAuth to true.
//
// For the official v1 registry, if there isn't already an Authorization header in the request,
// but there is an X-Docker-Token header set to true, then Basic Auth will be used to set the Authorization header.
// After sending the request with the provided base http.RoundTripper, if an X-Docker-Token header, representing
// a token, is present in the response, then it gets cached and sent in the Authorization header of all subsequent
// requests.
//
// If the server sends a token without the client having requested it, it is ignored.
//
// This RoundTripper also has a CancelRequest method important for correct timeout handling.
func AuthTransport(base http.RoundTripper, authConfig *types.AuthConfig, alwaysSetBasicAuth bool) http.RoundTripper {
	if base == nil {
		base = http.DefaultTransport
	}
	return &authTransport{
		RoundTripper:       base,
		AuthConfig:         authConfig,
		alwaysSetBasicAuth: alwaysSetBasicAuth,
		modReq:             make(map[*http.Request]*http.Request),
	}
}

// cloneRequest returns a clone of the provided *http.Request.
// The clone is a shallow copy of the struct and its Header map.
func cloneRequest(r *http.Request) *http.Request {
	// shallow copy of the struct
	r2 := new(http.Request)
	*r2 = *r
	// deep copy of the Header
	r2.Header = make(http.Header, len(r.Header))
	for k, s := range r.Header {
		r2.Header[k] = append([]string(nil), s...)
	}

	return r2
}

// RoundTrip changes an HTTP request's headers to add the necessary
// authentication-related headers
func (tr *authTransport) RoundTrip(orig *http.Request) (*http.Response, error) {
	// Authorization should not be set on 302 redirect for untrusted locations.
	// This logic mirrors the behavior in addRequiredHeadersToRedirectedRequests.
	// As the authorization logic is currently implemented in RoundTrip,
	// a 302 redirect is detected by looking at the Referrer header as go http package adds said header.
	// This is safe as Docker doesn't set Referrer in other scenarios.
	if orig.Header.Get("Referer") != "" && !trustedLocation(orig) {
		return tr.RoundTripper.RoundTrip(orig)
	}

	req := cloneRequest(orig)
	tr.mu.Lock()
	tr.modReq[orig] = req
	tr.mu.Unlock()

	if tr.alwaysSetBasicAuth {
		if tr.AuthConfig == nil {
			return nil, errors.New("unexpected error: empty auth config")
		}
		req.SetBasicAuth(tr.Username, tr.Password)
		return tr.RoundTripper.RoundTrip(req)
	}

	// Don't override
	if req.Header.Get("Authorization") == "" {
		if req.Header.Get("X-Docker-Token") == "true" && tr.AuthConfig != nil && len(tr.Username) > 0 {
			req.SetBasicAuth(tr.Username, tr.Password)
		} else if len(tr.token) > 0 {
			req.Header.Set("Authorization", "Token "+strings.Join(tr.token, ","))
		}
	}
	resp, err := tr.RoundTripper.RoundTrip(req)
	if err != nil {
		delete(tr.modReq, orig)
		return nil, err
	}
	if len(resp.Header["X-Docker-Token"]) > 0 {
		tr.token = resp.Header["X-Docker-Token"]
	}
	resp.Body = &ioutils.OnEOFReader{
		Rc: resp.Body,
		Fn: func() {
			tr.mu.Lock()
			delete(tr.modReq, orig)
			tr.mu.Unlock()
		},
	}
	return resp, nil
}

// CancelRequest cancels an in-flight request by closing its connection.
func (tr *authTransport) CancelRequest(req *http.Request) {
	type canceler interface {
		CancelRequest(*http.Request)
	}
	if cr, ok := tr.RoundTripper.(canceler); ok {
		tr.mu.Lock()
		modReq := tr.modReq[req]
		delete(tr.modReq, req)
		tr.mu.Unlock()
		cr.CancelRequest(modReq)
	}
}

func authorizeClient(client *http.Client, authConfig *types.AuthConfig, endpoint *V1Endpoint) error {
	var alwaysSetBasicAuth bool

	// If we're working with a standalone private registry over HTTPS, send Basic Auth headers
	// alongside all our requests.
	if endpoint.String() != IndexServer && endpoint.URL.Scheme == "https" {
		info, err := endpoint.Ping()
		if err != nil {
			return err
		}
		if info.Standalone && authConfig != nil {
			logrus.Debugf("Endpoint %s is eligible for private registry. Enabling decorator.", endpoint.String())
			alwaysSetBasicAuth = true
		}
	}

	// Annotate the transport unconditionally so that v2 can
	// properly fallback on v1 when an image is not found.
	client.Transport = AuthTransport(client.Transport, authConfig, alwaysSetBasicAuth)

	jar, err := cookiejar.New(nil)
	if err != nil {
		return errors.New("cookiejar.New is not supposed to return an error")
	}
	client.Jar = jar

	return nil
}

func newSession(client *http.Client, authConfig *types.AuthConfig, endpoint *V1Endpoint) *Session {
	return &Session{
		authConfig:    authConfig,
		client:        client,
		indexEndpoint: endpoint,
		id:            stringid.GenerateRandomID(),
	}
}

// NewSession creates a new session
// TODO(tiborvass): remove authConfig param once registry client v2 is vendored
func NewSession(client *http.Client, authConfig *types.AuthConfig, endpoint *V1Endpoint) (*Session, error) {
	if err := authorizeClient(client, authConfig, endpoint); err != nil {
		return nil, err
	}

	return newSession(client, authConfig, endpoint), nil
}

// ID returns this registry session's ID.
func (r *Session) ID() string {
	return r.id
}

// GetRemoteHistory retrieves the history of a given image from the registry.
// It returns a list of the parent's JSON files (including the requested image).
func (r *Session) GetRemoteHistory(imgID, registry string) ([]string, error) {
	res, err := r.client.Get(registry + "images/" + imgID + "/ancestry")
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		if res.StatusCode == 401 {
			return nil, errcode.ErrorCodeUnauthorized.WithArgs()
		}
		return nil, newJSONError(fmt.Sprintf("Server error: %d trying to fetch remote history for %s", res.StatusCode, imgID), res)
	}

	var history []string
	if err := json.NewDecoder(res.Body).Decode(&history); err != nil {
		return nil, fmt.Errorf("Error while reading the http response: %v", err)
	}

	logrus.Debugf("Ancestry: %v", history)
	return history, nil
}

// LookupRemoteImage checks if an image exists in the registry
func (r *Session) LookupRemoteImage(imgID, registry string) error {
	res, err := r.client.Get(registry + "images/" + imgID + "/json")
	if err != nil {
		return err
	}
	res.Body.Close()
	if res.StatusCode != 200 {
		return newJSONError(fmt.Sprintf("HTTP code %d", res.StatusCode), res)
	}
	return nil
}

// GetRemoteImageJSON retrieves an image's JSON metadata from the registry.
func (r *Session) GetRemoteImageJSON(imgID, registry string) ([]byte, int64, error) {
	res, err := r.client.Get(registry + "images/" + imgID + "/json")
	if err != nil {
		return nil, -1, fmt.Errorf("Failed to download json: %s", err)
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, -1, newJSONError(fmt.Sprintf("HTTP code %d", res.StatusCode), res)
	}
	// if the size header is not present, then set it to '-1'
	imageSize := int64(-1)
	if hdr := res.Header.Get("X-Docker-Size"); hdr != "" {
		imageSize, err = strconv.ParseInt(hdr, 10, 64)
		if err != nil {
			return nil, -1, err
		}
	}

	jsonString, err := ioutil.ReadAll(res.Body)
	if err != nil {
		return nil, -1, fmt.Errorf("Failed to parse downloaded json: %v (%s)", err, jsonString)
	}
	return jsonString, imageSize, nil
}

// GetRemoteImageLayer retrieves an image layer from the registry
func (r *Session) GetRemoteImageLayer(imgID, registry string, imgSize int64) (io.ReadCloser, error) {
	var (
		statusCode = 0
		res        *http.Response
		err        error
		imageURL   = fmt.Sprintf("%simages/%s/layer", registry, imgID)
	)

	req, err := http.NewRequest("GET", imageURL, nil)
	if err != nil {
		return nil, fmt.Errorf("Error while getting from the server: %v", err)
	}

	res, err = r.client.Do(req)
	if err != nil {
		logrus.Debugf("Error contacting registry %s: %v", registry, err)
		// the only case err != nil && res != nil is https://golang.org/src/net/http/client.go#L515
		if res != nil {
			if res.Body != nil {
				res.Body.Close()
			}
			statusCode = res.StatusCode
		}
		return nil, fmt.Errorf("Server error: Status %d while fetching image layer (%s)",
			statusCode, imgID)
	}

	if res.StatusCode != 200 {
		res.Body.Close()
		return nil, fmt.Errorf("Server error: Status %d while fetching image layer (%s)",
			res.StatusCode, imgID)
	}

	if res.Header.Get("Accept-Ranges") == "bytes" && imgSize > 0 {
		logrus.Debug("server supports resume")
		return resumable.NewRequestReaderWithInitialResponse(r.client, req, 5, imgSize, res), nil
	}
	logrus.Debug("server doesn't support resume")
	return res.Body, nil
}

// GetRemoteTag retrieves the tag named in the askedTag argument from the given
// repository. It queries each of the registries supplied in the registries
// argument, and returns data from the first one that answers the query
// successfully.
func (r *Session) GetRemoteTag(registries []string, repositoryRef reference.Named, askedTag string) (string, error) {
	repository := reference.Path(repositoryRef)

	if strings.Count(repository, "/") == 0 {
		// This will be removed once the registry supports auto-resolution on
		// the "library" namespace
		repository = "library/" + repository
	}
	for _, host := range registries {
		endpoint := fmt.Sprintf("%srepositories/%s/tags/%s", host, repository, askedTag)
		res, err := r.client.Get(endpoint)
		if err != nil {
			return "", err
		}

		logrus.Debugf("Got status code %d from %s", res.StatusCode, endpoint)
		defer res.Body.Close()

		if res.StatusCode == 404 {
			return "", ErrRepoNotFound
		}
		if res.StatusCode != 200 {
			continue
		}

		var tagID string
		if err := json.NewDecoder(res.Body).Decode(&tagID); err != nil {
			return "", err
		}
		return tagID, nil
	}
	return "", fmt.Errorf("Could not reach any registry endpoint")
}

// GetRemoteTags retrieves all tags from the given repository. It queries each
// of the registries supplied in the registries argument, and returns data from
// the first one that answers the query successfully. It returns a map with
// tag names as the keys and image IDs as the values.
func (r *Session) GetRemoteTags(registries []string, repositoryRef reference.Named) (map[string]string, error) {
	repository := reference.Path(repositoryRef)

	if strings.Count(repository, "/") == 0 {
		// This will be removed once the registry supports auto-resolution on
		// the "library" namespace
		repository = "library/" + repository
	}
	for _, host := range registries {
		endpoint := fmt.Sprintf("%srepositories/%s/tags", host, repository)
		res, err := r.client.Get(endpoint)
		if err != nil {
			return nil, err
		}

		logrus.Debugf("Got status code %d from %s", res.StatusCode, endpoint)
		defer res.Body.Close()

		if res.StatusCode == 404 {
			return nil, ErrRepoNotFound
		}
		if res.StatusCode != 200 {
			continue
		}

		result := make(map[string]string)
		if err := json.NewDecoder(res.Body).Decode(&result); err != nil {
			return nil, err
		}
		return result, nil
	}
	return nil, fmt.Errorf("Could not reach any registry endpoint")
}

func buildEndpointsList(headers []string, indexEp string) ([]string, error) {
	var endpoints []string
	parsedURL, err := url.Parse(indexEp)
	if err != nil {
		return nil, err
	}
	var urlScheme = parsedURL.Scheme
	// The registry's URL scheme has to match the Index'
	for _, ep := range headers {
		epList := strings.Split(ep, ",")
		for _, epListElement := range epList {
			endpoints = append(
				endpoints,
				fmt.Sprintf("%s://%s/v1/", urlScheme, strings.TrimSpace(epListElement)))
		}
	}
	return endpoints, nil
}

// GetRepositoryData returns lists of images and endpoints for the repository
func (r *Session) GetRepositoryData(name reference.Named) (*RepositoryData, error) {
	repositoryTarget := fmt.Sprintf("%srepositories/%s/images", r.indexEndpoint.String(), reference.Path(name))

	logrus.Debugf("[registry] Calling GET %s", repositoryTarget)

	req, err := http.NewRequest("GET", repositoryTarget, nil)
	if err != nil {
		return nil, err
	}
	// this will set basic auth in r.client.Transport and send cached X-Docker-Token headers for all subsequent requests
	req.Header.Set("X-Docker-Token", "true")
	res, err := r.client.Do(req)
	if err != nil {
		// check if the error is because of i/o timeout
		// and return a non-obtuse error message for users
		// "Get https://index.docker.io/v1/repositories/library/busybox/images: i/o timeout"
		// was a top search on the docker user forum
		if isTimeout(err) {
			return nil, fmt.Errorf("Network timed out while trying to connect to %s. You may want to check your internet connection or if you are behind a proxy.", repositoryTarget)
		}
		return nil, fmt.Errorf("Error while pulling image: %v", err)
	}
	defer res.Body.Close()
	if res.StatusCode == 401 {
		return nil, errcode.ErrorCodeUnauthorized.WithArgs()
	}
	// TODO: Right now we're ignoring checksums in the response body.
	// In the future, we need to use them to check image validity.
	if res.StatusCode == 404 {
		return nil, newJSONError(fmt.Sprintf("HTTP code: %d", res.StatusCode), res)
	} else if res.StatusCode != 200 {
		errBody, err := ioutil.ReadAll(res.Body)
		if err != nil {
			logrus.Debugf("Error reading response body: %s", err)
		}
		return nil, newJSONError(fmt.Sprintf("Error: Status %d trying to pull repository %s: %q", res.StatusCode, reference.Path(name), errBody), res)
	}

	var endpoints []string
	if res.Header.Get("X-Docker-Endpoints") != "" {
		endpoints, err = buildEndpointsList(res.Header["X-Docker-Endpoints"], r.indexEndpoint.String())
		if err != nil {
			return nil, err
		}
	} else {
		// Assume the endpoint is on the same host
		endpoints = append(endpoints, fmt.Sprintf("%s://%s/v1/", r.indexEndpoint.URL.Scheme, req.URL.Host))
	}

	remoteChecksums := []*ImgData{}
	if err := json.NewDecoder(res.Body).Decode(&remoteChecksums); err != nil {
		return nil, err
	}

	// Forge a better object from the retrieved data
	imgsData := make(map[string]*ImgData, len(remoteChecksums))
	for _, elem := range remoteChecksums {
		imgsData[elem.ID] = elem
	}

	return &RepositoryData{
		ImgList:   imgsData,
		Endpoints: endpoints,
	}, nil
}

// PushImageChecksumRegistry uploads checksums for an image
func (r *Session) PushImageChecksumRegistry(imgData *ImgData, registry string) error {
	u := registry + "images/" + imgData.ID + "/checksum"

	logrus.Debugf("[registry] Calling PUT %s", u)

	req, err := http.NewRequest("PUT", u, nil)
	if err != nil {
		return err
	}
	req.Header.Set("X-Docker-Checksum", imgData.Checksum)
	req.Header.Set("X-Docker-Checksum-Payload", imgData.ChecksumPayload)

	res, err := r.client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed to upload metadata: %v", err)
	}
	defer res.Body.Close()
	if len(res.Cookies()) > 0 {
		r.client.Jar.SetCookies(req.URL, res.Cookies())
	}
	if res.StatusCode != 200 {
		errBody, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return fmt.Errorf("HTTP code %d while uploading metadata and error when trying to parse response body: %s", res.StatusCode, err)
		}
		var jsonBody map[string]string
		if err := json.Unmarshal(errBody, &jsonBody); err != nil {
			errBody = []byte(err.Error())
		} else if jsonBody["error"] == "Image already exists" {
			return ErrAlreadyExists
		}
		return fmt.Errorf("HTTP code %d while uploading metadata: %q", res.StatusCode, errBody)
	}
	return nil
}

// PushImageJSONRegistry pushes JSON metadata for a local image to the registry
func (r *Session) PushImageJSONRegistry(imgData *ImgData, jsonRaw []byte, registry string) error {

	u := registry + "images/" + imgData.ID + "/json"

	logrus.Debugf("[registry] Calling PUT %s", u)

	req, err := http.NewRequest("PUT", u, bytes.NewReader(jsonRaw))
	if err != nil {
		return err
	}
	req.Header.Add("Content-type", "application/json")

	res, err := r.client.Do(req)
	if err != nil {
		return fmt.Errorf("Failed to upload metadata: %s", err)
	}
	defer res.Body.Close()
	if res.StatusCode == 401 && strings.HasPrefix(registry, "http://") {
		return newJSONError("HTTP code 401, Docker will not send auth headers over HTTP.", res)
	}
	if res.StatusCode != 200 {
		errBody, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return newJSONError(fmt.Sprintf("HTTP code %d while uploading metadata and error when trying to parse response body: %s", res.StatusCode, err), res)
		}
		var jsonBody map[string]string
		if err := json.Unmarshal(errBody, &jsonBody); err != nil {
			errBody = []byte(err.Error())
		} else if jsonBody["error"] == "Image already exists" {
			return ErrAlreadyExists
		}
		return newJSONError(fmt.Sprintf("HTTP code %d while uploading metadata: %q", res.StatusCode, errBody), res)
	}
	return nil
}

// PushImageLayerRegistry sends the checksum of an image layer to the registry
func (r *Session) PushImageLayerRegistry(imgID string, layer io.Reader, registry string, jsonRaw []byte) (checksum string, checksumPayload string, err error) {
	u := registry + "images/" + imgID + "/layer"

	logrus.Debugf("[registry] Calling PUT %s", u)

	tarsumLayer, err := tarsum.NewTarSum(layer, false, tarsum.Version0)
	if err != nil {
		return "", "", err
	}
	h := sha256.New()
	h.Write(jsonRaw)
	h.Write([]byte{'\n'})
	checksumLayer := io.TeeReader(tarsumLayer, h)

	req, err := http.NewRequest("PUT", u, checksumLayer)
	if err != nil {
		return "", "", err
	}
	req.Header.Add("Content-Type", "application/octet-stream")
	req.ContentLength = -1
	req.TransferEncoding = []string{"chunked"}
	res, err := r.client.Do(req)
	if err != nil {
		return "", "", fmt.Errorf("Failed to upload layer: %v", err)
	}
	if rc, ok := layer.(io.Closer); ok {
		if err := rc.Close(); err != nil {
			return "", "", err
		}
	}
	defer res.Body.Close()

	if res.StatusCode != 200 {
		errBody, err := ioutil.ReadAll(res.Body)
		if err != nil {
			return "", "", newJSONError(fmt.Sprintf("HTTP code %d while uploading metadata and error when trying to parse response body: %s", res.StatusCode, err), res)
		}
		return "", "", newJSONError(fmt.Sprintf("Received HTTP code %d while uploading layer: %q", res.StatusCode, errBody), res)
	}

	checksumPayload = "sha256:" + hex.EncodeToString(h.Sum(nil))
	return tarsumLayer.Sum(jsonRaw), checksumPayload, nil
}

// PushRegistryTag pushes a tag on the registry.
// Remote has the format '<user>/<repo>
func (r *Session) PushRegistryTag(remote reference.Named, revision, tag, registry string) error {
	// "jsonify" the string
	revision = "\"" + revision + "\""
	path := fmt.Sprintf("repositories/%s/tags/%s", reference.Path(remote), tag)

	req, err := http.NewRequest("PUT", registry+path, strings.NewReader(revision))
	if err != nil {
		return err
	}
	req.Header.Add("Content-type", "application/json")
	req.ContentLength = int64(len(revision))
	res, err := r.client.Do(req)
	if err != nil {
		return err
	}
	res.Body.Close()
	if res.StatusCode != 200 && res.StatusCode != 201 {
		return newJSONError(fmt.Sprintf("Internal server error: %d trying to push tag %s on %s", res.StatusCode, tag, reference.Path(remote)), res)
	}
	return nil
}

// PushImageJSONIndex uploads an image list to the repository
func (r *Session) PushImageJSONIndex(remote reference.Named, imgList []*ImgData, validate bool, regs []string) (*RepositoryData, error) {
	cleanImgList := []*ImgData{}
	if validate {
		for _, elem := range imgList {
			if elem.Checksum != "" {
				cleanImgList = append(cleanImgList, elem)
			}
		}
	} else {
		cleanImgList = imgList
	}

	imgListJSON, err := json.Marshal(cleanImgList)
	if err != nil {
		return nil, err
	}
	var suffix string
	if validate {
		suffix = "images"
	}
	u := fmt.Sprintf("%srepositories/%s/%s", r.indexEndpoint.String(), reference.Path(remote), suffix)
	logrus.Debugf("[registry] PUT %s", u)
	logrus.Debugf("Image list pushed to index:\n%s", imgListJSON)
	headers := map[string][]string{
		"Content-type": {"application/json"},
		// this will set basic auth in r.client.Transport and send cached X-Docker-Token headers for all subsequent requests
		"X-Docker-Token": {"true"},
	}
	if validate {
		headers["X-Docker-Endpoints"] = regs
	}

	// Redirect if necessary
	var res *http.Response
	for {
		if res, err = r.putImageRequest(u, headers, imgListJSON); err != nil {
			return nil, err
		}
		if !shouldRedirect(res) {
			break
		}
		res.Body.Close()
		u = res.Header.Get("Location")
		logrus.Debugf("Redirected to %s", u)
	}
	defer res.Body.Close()

	if res.StatusCode == 401 {
		return nil, errcode.ErrorCodeUnauthorized.WithArgs()
	}

	var tokens, endpoints []string
	if !validate {
		if res.StatusCode != 200 && res.StatusCode != 201 {
			errBody, err := ioutil.ReadAll(res.Body)
			if err != nil {
				logrus.Debugf("Error reading response body: %s", err)
			}
			return nil, newJSONError(fmt.Sprintf("Error: Status %d trying to push repository %s: %q", res.StatusCode, reference.Path(remote), errBody), res)
		}
		tokens = res.Header["X-Docker-Token"]
		logrus.Debugf("Auth token: %v", tokens)

		if res.Header.Get("X-Docker-Endpoints") == "" {
			return nil, fmt.Errorf("Index response didn't contain any endpoints")
		}
		endpoints, err = buildEndpointsList(res.Header["X-Docker-Endpoints"], r.indexEndpoint.String())
		if err != nil {
			return nil, err
		}
	} else {
		if res.StatusCode != 204 {
			errBody, err := ioutil.ReadAll(res.Body)
			if err != nil {
				logrus.Debugf("Error reading response body: %s", err)
			}
			return nil, newJSONError(fmt.Sprintf("Error: Status %d trying to push checksums %s: %q", res.StatusCode, reference.Path(remote), errBody), res)
		}
	}

	return &RepositoryData{
		Endpoints: endpoints,
	}, nil
}

func (r *Session) putImageRequest(u string, headers map[string][]string, body []byte) (*http.Response, error) {
	req, err := http.NewRequest("PUT", u, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.ContentLength = int64(len(body))
	for k, v := range headers {
		req.Header[k] = v
	}
	response, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	return response, nil
}

func shouldRedirect(response *http.Response) bool {
	return response.StatusCode >= 300 && response.StatusCode < 400
}

// SearchRepositories performs a search against the remote repository
func (r *Session) SearchRepositories(term string, limit int) (*registrytypes.SearchResults, error) {
	if limit < 1 || limit > 100 {
		return nil, fmt.Errorf("Limit %d is outside the range of [1, 100]", limit)
	}
	logrus.Debugf("Index server: %s", r.indexEndpoint)
	u := r.indexEndpoint.String() + "search?q=" + url.QueryEscape(term) + "&n=" + url.QueryEscape(fmt.Sprintf("%d", limit))

	req, err := http.NewRequest("GET", u, nil)
	if err != nil {
		return nil, fmt.Errorf("Error while getting from the server: %v", err)
	}
	// Have the AuthTransport send authentication, when logged in.
	req.Header.Set("X-Docker-Token", "true")
	res, err := r.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer res.Body.Close()
	if res.StatusCode != 200 {
		return nil, newJSONError(fmt.Sprintf("Unexpected status code %d", res.StatusCode), res)
	}
	result := new(registrytypes.SearchResults)
	return result, json.NewDecoder(res.Body).Decode(result)
}

func isTimeout(err error) bool {
	type timeout interface {
		Timeout() bool
	}
	e := err
	switch urlErr := err.(type) {
	case *url.Error:
		e = urlErr.Err
	}
	t, ok := e.(timeout)
	return ok && t.Timeout()
}

func newJSONError(msg string, res *http.Response) error {
	return &jsonmessage.JSONError{
		Message: msg,
		Code:    res.StatusCode,
	}
}
