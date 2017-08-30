package api

// Auth is used to perform credential backend related operations.
type Auth struct {
	c *Client
}

// Auth is used to return the client for credential-backend API calls.
func (c *Client) Auth() *Auth {
	return &Auth{c: c}
}
