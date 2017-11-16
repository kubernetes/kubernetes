package api

// TokenAuth is used to perform token backend operations on Vault
type TokenAuth struct {
	c *Client
}

// Token is used to return the client for token-backend API calls
func (a *Auth) Token() *TokenAuth {
	return &TokenAuth{c: a.c}
}

func (c *TokenAuth) Create(opts *TokenCreateRequest) (*Secret, error) {
	r := c.c.NewRequest("POST", "/v1/auth/token/create")
	if err := r.SetJSONBody(opts); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) CreateOrphan(opts *TokenCreateRequest) (*Secret, error) {
	r := c.c.NewRequest("POST", "/v1/auth/token/create-orphan")
	if err := r.SetJSONBody(opts); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) CreateWithRole(opts *TokenCreateRequest, roleName string) (*Secret, error) {
	r := c.c.NewRequest("POST", "/v1/auth/token/create/"+roleName)
	if err := r.SetJSONBody(opts); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) Lookup(token string) (*Secret, error) {
	r := c.c.NewRequest("POST", "/v1/auth/token/lookup")
	if err := r.SetJSONBody(map[string]interface{}{
		"token": token,
	}); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) LookupAccessor(accessor string) (*Secret, error) {
	r := c.c.NewRequest("POST", "/v1/auth/token/lookup-accessor")
	if err := r.SetJSONBody(map[string]interface{}{
		"accessor": accessor,
	}); err != nil {
		return nil, err
	}
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) LookupSelf() (*Secret, error) {
	r := c.c.NewRequest("GET", "/v1/auth/token/lookup-self")

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) Renew(token string, increment int) (*Secret, error) {
	r := c.c.NewRequest("PUT", "/v1/auth/token/renew")
	if err := r.SetJSONBody(map[string]interface{}{
		"token":     token,
		"increment": increment,
	}); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

func (c *TokenAuth) RenewSelf(increment int) (*Secret, error) {
	r := c.c.NewRequest("PUT", "/v1/auth/token/renew-self")

	body := map[string]interface{}{"increment": increment}
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

// RenewTokenAsSelf behaves like renew-self, but authenticates using a provided
// token instead of the token attached to the client.
func (c *TokenAuth) RenewTokenAsSelf(token string, increment int) (*Secret, error) {
	r := c.c.NewRequest("PUT", "/v1/auth/token/renew-self")
	r.ClientToken = token

	body := map[string]interface{}{"increment": increment}
	if err := r.SetJSONBody(body); err != nil {
		return nil, err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	return ParseSecret(resp.Body)
}

// RevokeAccessor revokes a token associated with the given accessor
// along with all the child tokens.
func (c *TokenAuth) RevokeAccessor(accessor string) error {
	r := c.c.NewRequest("POST", "/v1/auth/token/revoke-accessor")
	if err := r.SetJSONBody(map[string]interface{}{
		"accessor": accessor,
	}); err != nil {
		return err
	}
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// RevokeOrphan revokes a token without revoking the tree underneath it (so
// child tokens are orphaned rather than revoked)
func (c *TokenAuth) RevokeOrphan(token string) error {
	r := c.c.NewRequest("PUT", "/v1/auth/token/revoke-orphan")
	if err := r.SetJSONBody(map[string]interface{}{
		"token": token,
	}); err != nil {
		return err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// RevokeSelf revokes the token making the call. The `token` parameter is kept
// for backwards compatibility but is ignored; only the client's set token has
// an effect.
func (c *TokenAuth) RevokeSelf(token string) error {
	r := c.c.NewRequest("PUT", "/v1/auth/token/revoke-self")
	resp, err := c.c.RawRequest(r)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// RevokeTree is the "normal" revoke operation that revokes the given token and
// the entire tree underneath -- all of its child tokens, their child tokens,
// etc.
func (c *TokenAuth) RevokeTree(token string) error {
	r := c.c.NewRequest("PUT", "/v1/auth/token/revoke")
	if err := r.SetJSONBody(map[string]interface{}{
		"token": token,
	}); err != nil {
		return err
	}

	resp, err := c.c.RawRequest(r)
	if err != nil {
		return err
	}
	defer resp.Body.Close()

	return nil
}

// TokenCreateRequest is the options structure for creating a token.
type TokenCreateRequest struct {
	ID              string            `json:"id,omitempty"`
	Policies        []string          `json:"policies,omitempty"`
	Metadata        map[string]string `json:"meta,omitempty"`
	Lease           string            `json:"lease,omitempty"`
	TTL             string            `json:"ttl,omitempty"`
	ExplicitMaxTTL  string            `json:"explicit_max_ttl,omitempty"`
	Period          string            `json:"period,omitempty"`
	NoParent        bool              `json:"no_parent,omitempty"`
	NoDefaultPolicy bool              `json:"no_default_policy,omitempty"`
	DisplayName     string            `json:"display_name"`
	NumUses         int               `json:"num_uses"`
	Renewable       *bool             `json:"renewable,omitempty"`
}
