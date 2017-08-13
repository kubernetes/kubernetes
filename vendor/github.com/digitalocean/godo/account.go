package godo

import "github.com/digitalocean/godo/context"

// AccountService is an interface for interfacing with the Account
// endpoints of the DigitalOcean API
// See: https://developers.digitalocean.com/documentation/v2/#account
type AccountService interface {
	Get(context.Context) (*Account, *Response, error)
}

// AccountServiceOp handles communication with the Account related methods of
// the DigitalOcean API.
type AccountServiceOp struct {
	client *Client
}

var _ AccountService = &AccountServiceOp{}

// Account represents a DigitalOcean Account
type Account struct {
	DropletLimit    int    `json:"droplet_limit,omitempty"`
	FloatingIPLimit int    `json:"floating_ip_limit,omitempty"`
	Email           string `json:"email,omitempty"`
	UUID            string `json:"uuid,omitempty"`
	EmailVerified   bool   `json:"email_verified,omitempty"`
	Status          string `json:"status,omitempty"`
	StatusMessage   string `json:"status_message,omitempty"`
}

type accountRoot struct {
	Account *Account `json:"account"`
}

func (r Account) String() string {
	return Stringify(r)
}

// Get DigitalOcean account info
func (s *AccountServiceOp) Get(ctx context.Context) (*Account, *Response, error) {

	path := "v2/account"

	req, err := s.client.NewRequest(ctx, "GET", path, nil)
	if err != nil {
		return nil, nil, err
	}

	root := new(accountRoot)
	resp, err := s.client.Do(ctx, req, root)
	if err != nil {
		return nil, resp, err
	}

	return root.Account, resp, err
}
