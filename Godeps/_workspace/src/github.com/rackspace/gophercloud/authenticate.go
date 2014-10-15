package gophercloud

import (
	"fmt"
	"github.com/racker/perigee"
)

// AuthOptions lets anyone calling Authenticate() supply the required access credentials.
// At present, only Identity V2 API support exists; therefore, only Username, Password,
// and optionally, TenantId are provided.  If future Identity API versions become available,
// alternative fields unique to those versions may appear here.
type AuthOptions struct {
	// Username and Password are required if using Identity V2 API.
	// Consult with your provider's control panel to discover your
	// account's username and password.
	Username, Password string

	// ApiKey used for providers that support Api Key authentication
	ApiKey string

	// The TenantId field is optional for the Identity V2 API.
	TenantId string

	// The TenantName can be specified instead of the TenantId
	TenantName string

	// AllowReauth should be set to true if you grant permission for Gophercloud to cache
	// your credentials in memory, and to allow Gophercloud to attempt to re-authenticate
	// automatically if/when your token expires.  If you set it to false, it will not cache
	// these settings, but re-authentication will not be possible.  This setting defaults
	// to false.
	AllowReauth bool
}

// AuthContainer provides a JSON encoding wrapper for passing credentials to the Identity
// service.  You will not work with this structure directly.
type AuthContainer struct {
	Auth Auth `json:"auth"`
}

// Auth provides a JSON encoding wrapper for passing credentials to the Identity
// service.  You will not work with this structure directly.
type Auth struct {
	PasswordCredentials *PasswordCredentials `json:"passwordCredentials,omitempty"`
	ApiKeyCredentials   *ApiKeyCredentials   `json:"RAX-KSKEY:apiKeyCredentials,omitempty"`
	TenantId            string               `json:"tenantId,omitempty"`
	TenantName          string               `json:"tenantName,omitempty"`
}

// PasswordCredentials provides a JSON encoding wrapper for passing credentials to the Identity
// service.  You will not work with this structure directly.
type PasswordCredentials struct {
	Username string `json:"username"`
	Password string `json:"password"`
}

type ApiKeyCredentials struct {
	Username string `json:"username"`
	ApiKey   string `json:"apiKey"`
}

// Access encapsulates the API token and its relevant fields, as well as the
// services catalog that Identity API returns once authenticated.
type Access struct {
	Token          Token
	ServiceCatalog []CatalogEntry
	User           User
	provider       Provider    `json:"-"`
	options        AuthOptions `json:"-"`
	context        *Context    `json:"-"`
}

// Token encapsulates an authentication token and when it expires.  It also includes
// tenant information if available.
type Token struct {
	Id, Expires string
	Tenant      Tenant
}

// Tenant encapsulates tenant authentication information.  If, after authentication,
// no tenant information is supplied, both Id and Name will be "".
type Tenant struct {
	Id, Name string
}

// User encapsulates the user credentials, and provides visibility in what
// the user can do through its role assignments.
type User struct {
	Id, Name          string
	XRaxDefaultRegion string `json:"RAX-AUTH:defaultRegion"`
	Roles             []Role
}

// Role encapsulates a permission that a user can rely on.
type Role struct {
	Description, Id, Name string
}

// CatalogEntry encapsulates a service catalog record.
type CatalogEntry struct {
	Name, Type string
	Endpoints  []EntryEndpoint
}

// EntryEndpoint encapsulates how to get to the API of some service.
type EntryEndpoint struct {
	Region, TenantId                    string
	PublicURL, InternalURL              string
	VersionId, VersionInfo, VersionList string
}

type AuthError struct {
	StatusCode int
}

func (ae *AuthError) Error() string {
	switch ae.StatusCode {
	case 401:
		return "Auth failed. Bad credentials."

	default:
		return fmt.Sprintf("Auth failed. Status code is: %s.", ae.StatusCode)
	}
}

//
func getAuthCredentials(options AuthOptions) Auth {
	if options.ApiKey == "" {
		return Auth{
			PasswordCredentials: &PasswordCredentials{
				Username: options.Username,
				Password: options.Password,
			},
			TenantId:   options.TenantId,
			TenantName: options.TenantName,
		}
	} else {
		return Auth{
			ApiKeyCredentials: &ApiKeyCredentials{
				Username: options.Username,
				ApiKey:   options.ApiKey,
			},
			TenantId:   options.TenantId,
			TenantName: options.TenantName,
		}
	}
}

// papersPlease contains the common logic between authentication and re-authentication.
// The name, obviously a joke on the process of authentication, was chosen because
// of how many other entities exist in the program containing the word Auth or Authorization.
// I didn't need another one.
func (c *Context) papersPlease(p Provider, options AuthOptions) (*Access, error) {
	var access *Access
	access = new(Access)

	if (options.Username == "") || (options.Password == "" && options.ApiKey == "") {
		return nil, ErrCredentials
	}

	resp, err := perigee.Request("POST", p.AuthEndpoint, perigee.Options{
		CustomClient: c.httpClient,
		ReqBody: &AuthContainer{
			Auth: getAuthCredentials(options),
		},
		Results: &struct {
			Access **Access `json:"access"`
		}{
			&access,
		},
	})

	if err == nil {
		switch resp.StatusCode {
		case 200:
			access.options = options
			access.provider = p
			access.context = c

		default:
			err = &AuthError {
				StatusCode: resp.StatusCode,
			}
		}
	}

	return access, err
}

// Authenticate() grants access to the OpenStack-compatible provider API.
//
// Providers are identified through a unique key string.
// See the RegisterProvider() method for more details.
//
// The supplied AuthOptions instance allows the client to specify only those credentials
// relevant for the authentication request.  At present, support exists for OpenStack
// Identity V2 API only; support for V3 will become available as soon as documentation for it
// becomes readily available.
//
// For Identity V2 API requirements, you must provide at least the Username and Password
// options.  The TenantId field is optional, and defaults to "".
func (c *Context) Authenticate(provider string, options AuthOptions) (*Access, error) {
	p, err := c.ProviderByName(provider)
	if err != nil {
		return nil, err
	}
	return c.papersPlease(p, options)
}

// Reauthenticate attempts to reauthenticate using the configured access credentials, if
// allowed.  This method takes no action unless your AuthOptions has the AllowReauth flag
// set to true.
func (a *Access) Reauthenticate() error {
	var other *Access
	var err error

	if a.options.AllowReauth {
		other, err = a.context.papersPlease(a.provider, a.options)
		if err == nil {
			*a = *other
		}
	}
	return err
}

// See AccessProvider interface definition for details.
func (a *Access) FirstEndpointUrlByCriteria(ac ApiCriteria) string {
	ep := FindFirstEndpointByCriteria(a.ServiceCatalog, ac)
	urls := []string{ep.PublicURL, ep.InternalURL}
	return urls[ac.UrlChoice]
}

// See AccessProvider interface definition for details.
func (a *Access) AuthToken() string {
	return a.Token.Id
}

// See AccessProvider interface definition for details.
func (a *Access) Revoke(tok string) error {
	url := a.provider.AuthEndpoint + "/" + tok
	err := perigee.Delete(url, perigee.Options{
		MoreHeaders: map[string]string{
			"X-Auth-Token": a.AuthToken(),
		},
		OkCodes: []int{204},
	})
	return err
}

// See ServiceCatalogerForIdentityV2 interface definition for details.
// Note that the raw slice is returend; be careful not to alter the fields of any members,
// for other components of Gophercloud may depend upon them.
// If this becomes a problem in the future,
// a future revision may return a deep-copy of the service catalog instead.
func (a *Access) V2ServiceCatalog() []CatalogEntry {
	return a.ServiceCatalog
}
