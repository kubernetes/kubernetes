package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net/url"

	"github.com/docker/docker/api/types"
)

// ContainerInspect returns the container information.
func (cli *Client) ContainerInspect(ctx context.Context, containerID string) (types.ContainerJSON, error) {
	if containerID == "" {
		return types.ContainerJSON{}, objectNotFoundError{object: "container", id: containerID}
	}
	serverResp, err := cli.get(ctx, "/containers/"+containerID+"/json", nil, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return types.ContainerJSON{}, wrapResponseError(err, serverResp, "container", containerID)
	}

	var response types.ContainerJSON
	err = json.NewDecoder(serverResp.body).Decode(&response)
	return response, err
}

// ContainerInspectWithRaw returns the container information and its raw representation.
func (cli *Client) ContainerInspectWithRaw(ctx context.Context, containerID string, getSize bool) (types.ContainerJSON, []byte, error) {
	if containerID == "" {
		return types.ContainerJSON{}, nil, objectNotFoundError{object: "container", id: containerID}
	}
	query := url.Values{}
	if getSize {
		query.Set("size", "1")
	}
	serverResp, err := cli.get(ctx, "/containers/"+containerID+"/json", query, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return types.ContainerJSON{}, nil, wrapResponseError(err, serverResp, "container", containerID)
	}

	body, err := io.ReadAll(serverResp.body)
	if err != nil {
		return types.ContainerJSON{}, nil, err
	}

	var response types.ContainerJSON
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
