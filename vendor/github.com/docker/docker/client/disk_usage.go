package client // import "github.com/docker/docker/client"

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/docker/docker/api/types"
)

// DiskUsage requests the current data usage from the daemon
func (cli *Client) DiskUsage(ctx context.Context) (types.DiskUsage, error) {
	var du types.DiskUsage

	serverResp, err := cli.get(ctx, "/system/df", nil, nil)
	if err != nil {
		return du, err
	}
	defer ensureReaderClosed(serverResp)

	if err := json.NewDecoder(serverResp.body).Decode(&du); err != nil {
		return du, fmt.Errorf("Error retrieving disk usage: %v", err)
	}

	return du, nil
}
