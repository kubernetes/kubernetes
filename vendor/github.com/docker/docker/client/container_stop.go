package client

import (
	"net/url"
	"time"

	timetypes "github.com/docker/docker/api/types/time"
	"golang.org/x/net/context"
)

// ContainerStop stops a container without terminating the process.
// The process is blocked until the container stops or the timeout expires.
func (cli *Client) ContainerStop(ctx context.Context, containerID string, timeout *time.Duration) error {
	query := url.Values{}
	if timeout != nil {
		query.Set("t", timetypes.DurationToSecondsString(*timeout))
	}
	resp, err := cli.post(ctx, "/containers/"+containerID+"/stop", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
