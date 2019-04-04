package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"

	"github.com/docker/docker/api/types/swarm"
)

// SecretInspectWithRaw returns the secret information with raw data
func (cli *Client) SecretInspectWithRaw(ctx context.Context, id string) (swarm.Secret, []byte, error) {
	if err := cli.NewVersionError("1.25", "secret inspect"); err != nil {
		return swarm.Secret{}, nil, err
	}
	if id == "" {
		return swarm.Secret{}, nil, objectNotFoundError{object: "secret", id: id}
	}
	resp, err := cli.get(ctx, "/secrets/"+id, nil, nil)
	if err != nil {
		return swarm.Secret{}, nil, wrapResponseError(err, resp, "secret", id)
	}
	defer ensureReaderClosed(resp)

	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return swarm.Secret{}, nil, err
	}

	var secret swarm.Secret
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&secret)

	return secret, body, err
}
