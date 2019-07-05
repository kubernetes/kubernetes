package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"

	"github.com/docker/docker/api/types/swarm"
)

// ConfigInspectWithRaw returns the config information with raw data
func (cli *Client) ConfigInspectWithRaw(ctx context.Context, id string) (swarm.Config, []byte, error) {
	if id == "" {
		return swarm.Config{}, nil, objectNotFoundError{object: "config", id: id}
	}
	if err := cli.NewVersionError("1.30", "config inspect"); err != nil {
		return swarm.Config{}, nil, err
	}
	resp, err := cli.get(ctx, "/configs/"+id, nil, nil)
	defer ensureReaderClosed(resp)
	if err != nil {
		return swarm.Config{}, nil, wrapResponseError(err, resp, "config", id)
	}

	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return swarm.Config{}, nil, err
	}

	var config swarm.Config
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&config)

	return config, body, err
}
