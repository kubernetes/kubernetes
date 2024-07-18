package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types"
)

// VolumeInspect returns the information about a specific volume in the docker host.
func (cli *Client) VolumeInspect(ctx context.Context, volumeID string) (types.Volume, error) {
	volume, _, err := cli.VolumeInspectWithRaw(ctx, volumeID)
	return volume, err
}

// VolumeInspectWithRaw returns the information about a specific volume in the docker host and its raw representation
func (cli *Client) VolumeInspectWithRaw(ctx context.Context, volumeID string) (types.Volume, []byte, error) {
	if volumeID == "" {
		return types.Volume{}, nil, objectNotFoundError{object: "volume", id: volumeID}
	}

	var volume types.Volume
	resp, err := cli.get(ctx, "/volumes/"+volumeID, nil, nil)
	defer ensureReaderClosed(resp)
	if err != nil {
		return volume, nil, wrapResponseError(err, resp, "volume", volumeID)
	}

	body, err := io.ReadAll(resp.body)
	if err != nil {
		return volume, nil, err
	}
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&volume)
	return volume, body, err
}
