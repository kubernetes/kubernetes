package gophercloud

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"io"
	"io/ioutil"
	"net/http"
	"strings"
	"sync"
)

// DefaultUserAgent is the default User-Agent string set in the request header.
const DefaultUserAgent = "gophercloud/2.0.0"

// UserAgent represents a User-Agent header.
type UserAgent struct {
	// prepend is the slice of User-Agent strings to prepend to DefaultUserAgent.
	// All the strings to prepend are accumulated and prepended in the Join method.
	prepend []string
}

// Prepend prepends a user-defined string to the default User-Agent string. Users
// may pass in one or more strings to prepend.
func (ua *UserAgent) Prepend(s ...string) {
	ua.prepend = append(s, ua.prepend...)
}

// Join concatenates all the user-defined User-Agend strings with the default
// Gophercloud User-Agent string.
func (ua *UserAgent) Join() string {
	uaSlice := append(ua.prepend, DefaultUserAgent)
	return strings.Join(uaSlice, " ")
}

// ProviderClient stores details that are required to interact with any
// services within a specific provider's API.
//
// Generally, you acquire a ProviderClient by calling the NewClient method in
// the appropriate provider's child package, providing whatever authentication
// credentials are required.
type ProviderClient struct {
	// IdentityBase is the base URL used for a particular provider's identity
	// service - it will be used when issuing authenticatation requests. It
	// should point to the root resource of the identity service, not a specific
	// identity version.
	IdentityBase string

	// IdentityEndpoint is the identity endpoint. This may be a specific version
	// of the identity service. If this is the case, this endpoint is used rather
	// than querying versions first.
	IdentityEndpoint string

	// TokenID is the ID of the most recently issued valid token.
	// NOTE: Aside from within a custom ReauthFunc, this field shouldn't be set by an application.
	// To safely read or write this value, call `Token` or `SetToken`, respectively
	TokenID string

	// EndpointLocator describes how this provider discovers the endpoints for
	// its constituent services.
	EndpointLocator EndpointLocator

	// HTTPClient allows users to interject arbitrary http, https, or other transit behaviors.
	HTTPClient http.Client

	// UserAgent represents the User-Agent header in the HTTP request.
	UserAgent UserAgent

	// ReauthFunc is the function used to re-authenticate the user if the request
	// fails with a 401 HTTP response code. This a needed because there may be multiple
	// authentication functions for different Identity service versions.
	ReauthFunc func() error

	// Throwaway determines whether if this client is a throw-away client. It's a copy of user's provider client
	// with the token and reauth func zeroed. Such client can be used to perform reauthorization.
	Throwaway bool

	// Context is the context passed to the HTTP request.
	Context context.Context

	// mut is a mutex for the client. It protects read and write access to client attributes such as getting
	// and setting the TokenID.
	mut *sync.RWMutex

	// reauthmut is a mutex for reauthentication it attempts to ensure that only one reauthentication
	// attempt happens at one time.
	reauthmut *reauthlock

	authResult AuthResult
}

// reauthlock represents a set of attributes used to help in the reauthentication process.
type reauthlock struct {
	sync.RWMutex
	reauthing    bool
	reauthingErr error
	done         *sync.Cond
}

// AuthenticatedHeaders returns a map of HTTP headers that are common for all
// authenticated service requests. Blocks if Reauthenticate is in progress.
func (client *ProviderClient) AuthenticatedHeaders() (m map[string]string) {
	if client.IsThrowaway() {
		return
	}
	if client.reauthmut != nil {
		client.reauthmut.Lock()
		for client.reauthmut.reauthing {
			client.reauthmut.done.Wait()
		}
		client.reauthmut.Unlock()
	}
	t := client.Token()
	if t == "" {
		return
	}
	return map[string]string{"X-Auth-Token": t}
}

// UseTokenLock creates a mutex that is used to allow safe concurrent access to the auth token.
// If the application's ProviderClient is not used concurrently, this doesn't need to be called.
func (client *ProviderClient) UseTokenLock() {
	client.mut = new(sync.RWMutex)
	client.reauthmut = new(reauthlock)
}

// GetAuthResult returns the result from the request that was used to obtain a
// provider client's Keystone token.
//
// The result is nil when authentication has not yet taken place, when the token
// was set manually with SetToken(), or when a ReauthFunc was used that does not
// record the AuthResult.
func (client *ProviderClient) GetAuthResult() AuthResult {
	if client.mut != nil {
		client.mut.RLock()
		defer client.mut.RUnlock()
	}
	return client.authResult
}

// Token safely reads the value of the auth token from the ProviderClient. Applications should
// call this method to access the token instead of the TokenID field
func (client *ProviderClient) Token() string {
	if client.mut != nil {
		client.mut.RLock()
		defer client.mut.RUnlock()
	}
	return client.TokenID
}

// SetToken safely sets the value of the auth token in the ProviderClient. Applications may
// use this method in a custom ReauthFunc.
//
// WARNING: This function is deprecated. Use SetTokenAndAuthResult() instead.
func (client *ProviderClient) SetToken(t string) {
	if client.mut != nil {
		client.mut.Lock()
		defer client.mut.Unlock()
	}
	client.TokenID = t
	client.authResult = nil
}

// SetTokenAndAuthResult safely sets the value of the auth token in the
// ProviderClient and also records the AuthResult that was returned from the
// token creation request. Applications may call this in a custom ReauthFunc.
func (client *ProviderClient) SetTokenAndAuthResult(r AuthResult) error {
	tokenID := ""
	var err error
	if r != nil {
		tokenID, err = r.ExtractTokenID()
		if err != nil {
			return err
		}
	}

	if client.mut != nil {
		client.mut.Lock()
		defer client.mut.Unlock()
	}
	client.TokenID = tokenID
	client.authResult = r
	return nil
}

// CopyTokenFrom safely copies the token from another ProviderClient into the
// this one.
func (client *ProviderClient) CopyTokenFrom(other *ProviderClient) {
	if client.mut != nil {
		client.mut.Lock()
		defer client.mut.Unlock()
	}
	if other.mut != nil && other.mut != client.mut {
		other.mut.RLock()
		defer other.mut.RUnlock()
	}
	client.TokenID = other.TokenID
	client.authResult = other.authResult
}

// IsThrowaway safely reads the value of the client Throwaway field.
func (client *ProviderClient) IsThrowaway() bool {
	if client.reauthmut != nil {
		client.reauthmut.RLock()
		defer client.reauthmut.RUnlock()
	}
	return client.Throwaway
}

// SetThrowaway safely sets the value of the client Throwaway field.
func (client *ProviderClient) SetThrowaway(v bool) {
	if client.reauthmut != nil {
		client.reauthmut.Lock()
		defer client.reauthmut.Unlock()
	}
	client.Throwaway = v
}

// Reauthenticate calls client.ReauthFunc in a thread-safe way. If this is
// called because of a 401 response, the caller may pass the previous token. In
// this case, the reauthentication can be skipped if another thread has already
// reauthenticated in the meantime. If no previous token is known, an empty
// string should be passed instead to force unconditional reauthentication.
func (client *ProviderClient) Reauthenticate(previousToken string) (err error) {
	if client.ReauthFunc == nil {
		return nil
	}

	if client.reauthmut == nil {
		return client.ReauthFunc()
	}

	client.reauthmut.Lock()
	if client.reauthmut.reauthing {
		for !client.reauthmut.reauthing {
			client.reauthmut.done.Wait()
		}
		err = client.reauthmut.reauthingErr
		client.reauthmut.Unlock()
		return err
	}
	client.reauthmut.Unlock()

	client.reauthmut.Lock()
	client.reauthmut.reauthing = true
	client.reauthmut.done = sync.NewCond(client.reauthmut)
	client.reauthmut.reauthingErr = nil
	client.reauthmut.Unlock()

	if previousToken == "" || client.TokenID == previousToken {
		err = client.ReauthFunc()
	}

	client.reauthmut.Lock()
	client.reauthmut.reauthing = false
	client.reauthmut.reauthingErr = err
	client.reauthmut.done.Broadcast()
	client.reauthmut.Unlock()
	return
}

// RequestOpts customizes the behavior of the provider.Request() method.
type RequestOpts struct {
	// JSONBody, if provided, will be encoded as JSON and used as the body of the HTTP request. The
	// content type of the request will default to "application/json" unless overridden by MoreHeaders.
	// It's an error to specify both a JSONBody and a RawBody.
	JSONBody interface{}
	// RawBody contains an io.Reader that will be consumed by the request directly. No content-type
	// will be set unless one is provided explicitly by MoreHeaders.
	RawBody io.Reader
	// JSONResponse, if provided, will be populated with the contents of the response body parsed as
	// JSON.
	JSONResponse interface{}
	// OkCodes contains a list of numeric HTTP status codes that should be interpreted as success. If
	// the response has a different code, an error will be returned.
	OkCodes []int
	// MoreHeaders specifies additional HTTP headers to be provide on the request. If a header is
	// provided with a blank value (""), that header will be *omitted* instead: use this to suppress
	// the default Accept header or an inferred Content-Type, for example.
	MoreHeaders map[string]string
	// ErrorContext specifies the resource error type to return if an error is encountered.
	// This lets resources override default error messages based on the response status code.
	ErrorContext error
}

var applicationJSON = "application/json"

// Request performs an HTTP request using the ProviderClient's current HTTPClient. An authentication
// header will automatically be provided.
func (client *ProviderClient) Request(method, url string, options *RequestOpts) (*http.Response, error) {
	var body io.Reader
	var contentType *string

	// Derive the content body by either encoding an arbitrary object as JSON, or by taking a provided
	// io.ReadSeeker as-is. Default the content-type to application/json.
	if options.JSONBody != nil {
		if options.RawBody != nil {
			return nil, errors.New("please provide only one of JSONBody or RawBody to gophercloud.Request()")
		}

		rendered, err := json.Marshal(options.JSONBody)
		if err != nil {
			return nil, err
		}

		body = bytes.NewReader(rendered)
		contentType = &applicationJSON
	}

	if options.RawBody != nil {
		body = options.RawBody
	}

	// Construct the http.Request.
	req, err := http.NewRequest(method, url, body)
	if err != nil {
		return nil, err
	}
	if client.Context != nil {
		req = req.WithContext(client.Context)
	}

	// Populate the request headers. Apply options.MoreHeaders last, to give the caller the chance to
	// modify or omit any header.
	if contentType != nil {
		req.Header.Set("Content-Type", *contentType)
	}
	req.Header.Set("Accept", applicationJSON)

	// Set the User-Agent header
	req.Header.Set("User-Agent", client.UserAgent.Join())

	if options.MoreHeaders != nil {
		for k, v := range options.MoreHeaders {
			if v != "" {
				req.Header.Set(k, v)
			} else {
				req.Header.Del(k)
			}
		}
	}

	// get latest token from client
	for k, v := range client.AuthenticatedHeaders() {
		req.Header.Set(k, v)
	}

	// Set connection parameter to close the connection immediately when we've got the response
	req.Close = true

	prereqtok := req.Header.Get("X-Auth-Token")

	// Issue the request.
	resp, err := client.HTTPClient.Do(req)
	if err != nil {
		return nil, err
	}

	// Allow default OkCodes if none explicitly set
	okc := options.OkCodes
	if okc == nil {
		okc = defaultOkCodes(method)
	}

	// Validate the HTTP response status.
	var ok bool
	for _, code := range okc {
		if resp.StatusCode == code {
			ok = true
			break
		}
	}

	if !ok {
		body, _ := ioutil.ReadAll(resp.Body)
		resp.Body.Close()
		respErr := ErrUnexpectedResponseCode{
			URL:      url,
			Method:   method,
			Expected: options.OkCodes,
			Actual:   resp.StatusCode,
			Body:     body,
		}

		errType := options.ErrorContext
		switch resp.StatusCode {
		case http.StatusBadRequest:
			err = ErrDefault400{respErr}
			if error400er, ok := errType.(Err400er); ok {
				err = error400er.Error400(respErr)
			}
		case http.StatusUnauthorized:
			if client.ReauthFunc != nil {
				err = client.Reauthenticate(prereqtok)
				if err != nil {
					e := &ErrUnableToReauthenticate{}
					e.ErrOriginal = respErr
					return nil, e
				}
				if options.RawBody != nil {
					if seeker, ok := options.RawBody.(io.Seeker); ok {
						seeker.Seek(0, 0)
					}
				}
				resp, err = client.Request(method, url, options)
				if err != nil {
					switch err.(type) {
					case *ErrUnexpectedResponseCode:
						e := &ErrErrorAfterReauthentication{}
						e.ErrOriginal = err.(*ErrUnexpectedResponseCode)
						return nil, e
					default:
						e := &ErrErrorAfterReauthentication{}
						e.ErrOriginal = err
						return nil, e
					}
				}
				return resp, nil
			}
			err = ErrDefault401{respErr}
			if error401er, ok := errType.(Err401er); ok {
				err = error401er.Error401(respErr)
			}
		case http.StatusForbidden:
			err = ErrDefault403{respErr}
			if error403er, ok := errType.(Err403er); ok {
				err = error403er.Error403(respErr)
			}
		case http.StatusNotFound:
			err = ErrDefault404{respErr}
			if error404er, ok := errType.(Err404er); ok {
				err = error404er.Error404(respErr)
			}
		case http.StatusMethodNotAllowed:
			err = ErrDefault405{respErr}
			if error405er, ok := errType.(Err405er); ok {
				err = error405er.Error405(respErr)
			}
		case http.StatusRequestTimeout:
			err = ErrDefault408{respErr}
			if error408er, ok := errType.(Err408er); ok {
				err = error408er.Error408(respErr)
			}
		case http.StatusConflict:
			err = ErrDefault409{respErr}
			if error409er, ok := errType.(Err409er); ok {
				err = error409er.Error409(respErr)
			}
		case 429:
			err = ErrDefault429{respErr}
			if error429er, ok := errType.(Err429er); ok {
				err = error429er.Error429(respErr)
			}
		case http.StatusInternalServerError:
			err = ErrDefault500{respErr}
			if error500er, ok := errType.(Err500er); ok {
				err = error500er.Error500(respErr)
			}
		case http.StatusServiceUnavailable:
			err = ErrDefault503{respErr}
			if error503er, ok := errType.(Err503er); ok {
				err = error503er.Error503(respErr)
			}
		}

		if err == nil {
			err = respErr
		}

		return resp, err
	}

	// Parse the response body as JSON, if requested to do so.
	if options.JSONResponse != nil {
		defer resp.Body.Close()
		if err := json.NewDecoder(resp.Body).Decode(options.JSONResponse); err != nil {
			return nil, err
		}
	}

	return resp, nil
}

func defaultOkCodes(method string) []int {
	switch {
	case method == "GET":
		return []int{200}
	case method == "POST":
		return []int{201, 202}
	case method == "PUT":
		return []int{201, 202}
	case method == "PATCH":
		return []int{200, 202, 204}
	case method == "DELETE":
		return []int{202, 204}
	}

	return []int{}
}
