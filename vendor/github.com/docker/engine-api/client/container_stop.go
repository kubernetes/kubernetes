package client

import (
	"net/url"
	"strconv"

	"golang.org/x/net/context"
)

// ContainerStop stops a container without terminating the process.
// The process is blocked until the container stops or the timeout expires.
func (cli *Client) ContainerStop(ctx context.Context, containerID string, timeout int) error {
	query := url.Values{}
	query.Set("t", strconv.Itoa(timeout))
	resp, err := cli.post(ctx, "/containers/"+containerID+"/stop", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
