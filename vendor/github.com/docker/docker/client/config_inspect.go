package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"

	"github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

// ConfigInspectWithRaw returns the config information with raw data
func (cli *Client) ConfigInspectWithRaw(ctx context.Context, id string) (swarm.Config, []byte, error) {
	if err := cli.NewVersionError("1.30", "config inspect"); err != nil {
		return swarm.Config{}, nil, err
	}
	resp, err := cli.get(ctx, "/configs/"+id, nil, nil)
	if err != nil {
		if resp.statusCode == http.StatusNotFound {
			return swarm.Config{}, nil, configNotFoundError{id}
		}
		return swarm.Config{}, nil, err
	}
	defer ensureReaderClosed(resp)

	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return swarm.Config{}, nil, err
	}

	var config swarm.Config
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&config)

	return config, body, err
}
