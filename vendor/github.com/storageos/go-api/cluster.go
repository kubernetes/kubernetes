package storageos

import (
	"encoding/json"

	"github.com/storageos/go-api/types"
)

var (
	// ClusterMaintenanceAPIPrefix is a path to the HTTP endpoint for managing
	// the cluster maintenance mode.
	ClusterMaintenanceAPIPrefix = "cluster/maintenance"
)

// Maintenance returns the maintenance status of the cluster
func (c *Client) Maintenance() (*types.Maintenance, error) {
	resp, err := c.do("GET", ClusterMaintenanceAPIPrefix, doOptions{})
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	res := &types.Maintenance{}
	if err := json.NewDecoder(resp.Body).Decode(res); err != nil {
		return nil, err
	}
	return res, nil
}

// EnableMaintenance enables maintenance mode in the cluster
func (c *Client) EnableMaintenance() error {
	resp, err := c.do("POST", ClusterMaintenanceAPIPrefix, doOptions{})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}

// DisableMaintenance disables maintenance mode in the cluster
func (c *Client) DisableMaintenance() error {
	resp, err := c.do("DELETE", ClusterMaintenanceAPIPrefix, doOptions{})
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return nil
}
