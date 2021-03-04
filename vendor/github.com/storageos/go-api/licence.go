package storageos

import (
	"encoding/json"

	"github.com/storageos/go-api/types"
)

const (
	// licenceAPIPrefix is a partial path to the HTTP endpoint.
	licenceAPIPrefix = "licencing"
)

// Licence returns the current licence on the server.
func (c *Client) Licence() (*types.Licence, error) {
	resp, err := c.do("GET", licenceAPIPrefix, doOptions{})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	licence := &types.Licence{}
	if err := json.NewDecoder(resp.Body).Decode(&licence); err != nil {
		return nil, err
	}
	return licence, nil
}

// LicenceApply applies a licence on the server.
func (c *Client) LicenceApply(licenceKey string) error {
	_, err := c.do("POST", licenceAPIPrefix, doOptions{
		data: &types.LicenceKeyContainer{Key: licenceKey},
	})
	return err
}

// LicenceDelete removes the current licence.
func (c *Client) LicenceDelete() error {
	resp, err := c.do("DELETE", licenceAPIPrefix, doOptions{})
	if err != nil {
		return err
	}
	return resp.Body.Close()
}
