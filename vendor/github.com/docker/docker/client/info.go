package client

import (
	"encoding/json"
	"fmt"
	"net/url"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// Info returns information about the docker server.
func (cli *Client) Info(ctx context.Context) (types.Info, error) {
	var info types.Info
	serverResp, err := cli.get(ctx, "/info", url.Values{}, nil)
	if err != nil {
		return info, err
	}
	defer ensureReaderClosed(serverResp)

	if err := json.NewDecoder(serverResp.body).Decode(&info); err != nil {
		return info, fmt.Errorf("Error reading remote info: %v", err)
	}

	return info, nil
}
