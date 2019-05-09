package client // import "github.com/docker/docker/client"

import (
	"context"
	"io"
	"net/url"
)

// ContainerExport retrieves the raw contents of a container
// and returns them as an io.ReadCloser. It's up to the caller
// to close the stream.
func (cli *Client) ContainerExport(ctx context.Context, containerID string) (io.ReadCloser, error) {
	serverResp, err := cli.get(ctx, "/containers/"+containerID+"/export", url.Values{}, nil)
	if err != nil {
		return nil, err
	}

	return serverResp.body, nil
}
