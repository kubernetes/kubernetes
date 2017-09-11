package storageos

import (
	"context"
	"encoding/json"
	"net/http"

	"github.com/storageos/go-api/types"
)

// ServerVersion returns the server's version and runtime info.
func (c *Client) ServerVersion(ctx context.Context) (*types.VersionInfo, error) {

	// Send as unversioned
	resp, err := c.do("GET", "version", doOptions{context: ctx, unversioned: true})
	if err != nil {
		return nil, err
	}
	if resp.StatusCode != http.StatusOK {
		return nil, newError(resp)
	}
	defer resp.Body.Close()
	var version types.VersionInfo
	if err := json.NewDecoder(resp.Body).Decode(&version); err != nil {
		return nil, err
	}
	return &version, nil
}
