package client

import (
	"net/url"
	"strconv"

	"golang.org/x/net/context"
)

// ContainerRestart stops and starts a container again.
// It makes the daemon to wait for the container to be up again for
// a specific amount of time, given the timeout.
func (cli *Client) ContainerRestart(ctx context.Context, containerID string, timeout int) error {
	query := url.Values{}
	query.Set("t", strconv.Itoa(timeout))
	resp, err := cli.post(ctx, "/containers/"+containerID+"/restart", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
