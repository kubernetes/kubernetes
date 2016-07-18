package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"

	"github.com/docker/engine-api/types/swarm"
	"golang.org/x/net/context"
)

// ServiceInspectWithRaw returns the service information and the raw data.
func (cli *Client) ServiceInspectWithRaw(ctx context.Context, serviceID string) (swarm.Service, []byte, error) {
	serverResp, err := cli.get(ctx, "/services/"+serviceID, nil, nil)
	if err != nil {
		if serverResp.statusCode == http.StatusNotFound {
			return swarm.Service{}, nil, serviceNotFoundError{serviceID}
		}
		return swarm.Service{}, nil, err
	}
	defer ensureReaderClosed(serverResp)

	body, err := ioutil.ReadAll(serverResp.body)
	if err != nil {
		return swarm.Service{}, nil, err
	}

	var response swarm.Service
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
