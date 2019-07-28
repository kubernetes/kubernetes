package storageos

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/storageos/go-api/types"
)

var (
	// HealthAPIPrefix is a partial path to the HTTP endpoint.
	HealthAPIPrefix = "health"
)

// CPHealth returns the health of the control plane server at a given url.
func (c *Client) CPHealth(ctx context.Context, hostname string) (*types.CPHealthStatus, error) {

	url := fmt.Sprintf("http://%s:%s/v1/%s", hostname, DefaultPort, HealthAPIPrefix)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" && c.secret != "" {
		req.SetBasicAuth(c.username, c.secret)
	}

	c.configLock.RLock()
	resp, err := c.httpClient.Do(req.WithContext(ctx))
	c.configLock.RUnlock()
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var status *types.CPHealthStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}

	return status, nil
}

// DPHealth returns the health of the data plane server at a given url.
func (c *Client) DPHealth(ctx context.Context, hostname string) (*types.DPHealthStatus, error) {

	url := fmt.Sprintf("http://%s:%s/v1/%s", hostname, DataplaneHealthPort, HealthAPIPrefix)
	req, err := http.NewRequest("GET", url, nil)
	if err != nil {
		return nil, err
	}

	req.Header.Set("User-Agent", c.userAgent)
	if c.username != "" && c.secret != "" {
		req.SetBasicAuth(c.username, c.secret)
	}

	c.configLock.RLock()
	resp, err := c.httpClient.Do(req.WithContext(ctx))
	c.configLock.RUnlock()
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	var status *types.DPHealthStatus
	if err := json.NewDecoder(resp.Body).Decode(&status); err != nil {
		return nil, err
	}

	return status, nil
}
