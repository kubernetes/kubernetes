package client

import (
	"net/url"
	"strconv"

	"github.com/docker/docker/api/types/swarm"
	"golang.org/x/net/context"
)

// SecretUpdate updates a Secret. Currently, the only part of a secret spec
// which can be updated is Labels.
func (cli *Client) SecretUpdate(ctx context.Context, id string, version swarm.Version, secret swarm.SecretSpec) error {
	query := url.Values{}
	query.Set("version", strconv.FormatUint(version.Index, 10))
	resp, err := cli.post(ctx, "/secrets/"+id+"/update", query, secret, nil)
	ensureReaderClosed(resp)
	return err
}
