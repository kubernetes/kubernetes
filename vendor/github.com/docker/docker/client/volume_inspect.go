package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"path"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// VolumeInspect returns the information about a specific volume in the docker host.
func (cli *Client) VolumeInspect(ctx context.Context, volumeID string) (types.Volume, error) {
	volume, _, err := cli.VolumeInspectWithRaw(ctx, volumeID)
	return volume, err
}

// VolumeInspectWithRaw returns the information about a specific volume in the docker host and its raw representation
func (cli *Client) VolumeInspectWithRaw(ctx context.Context, volumeID string) (types.Volume, []byte, error) {
	// The empty ID needs to be handled here because with an empty ID the
	// request url will not contain a trailing / which calls the volume list API
	// instead of volume inspect
	if volumeID == "" {
		return types.Volume{}, nil, objectNotFoundError{object: "volume", id: volumeID}
	}

	var volume types.Volume
	resp, err := cli.get(ctx, path.Join("/volumes", volumeID), nil, nil)
	if err != nil {
		return volume, nil, wrapResponseError(err, resp, "volume", volumeID)
	}
	defer ensureReaderClosed(resp)

	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return volume, nil, err
	}
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&volume)
	return volume, body, err
}
