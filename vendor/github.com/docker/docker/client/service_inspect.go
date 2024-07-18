package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/url"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/swarm"
)

// ServiceInspectWithRaw returns the service information and the raw data.
func (cli *Client) ServiceInspectWithRaw(ctx context.Context, serviceID string, opts types.ServiceInspectOptions) (swarm.Service, []byte, error) {
	if serviceID == "" {
		return swarm.Service{}, nil, objectNotFoundError{object: "service", id: serviceID}
	}
	query := url.Values{}
	query.Set("insertDefaults", fmt.Sprintf("%v", opts.InsertDefaults))
	serverResp, err := cli.get(ctx, "/services/"+serviceID, query, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return swarm.Service{}, nil, wrapResponseError(err, serverResp, "service", serviceID)
	}

	body, err := io.ReadAll(serverResp.body)
	if err != nil {
		return swarm.Service{}, nil, err
	}

	var response swarm.Service
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
