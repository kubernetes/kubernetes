package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io/ioutil"

	"github.com/docker/docker/api/types"
)

// ImageInspectWithRaw returns the image information and its raw representation.
func (cli *Client) ImageInspectWithRaw(ctx context.Context, imageID string) (types.ImageInspect, []byte, error) {
	if imageID == "" {
		return types.ImageInspect{}, nil, objectNotFoundError{object: "image", id: imageID}
	}
	serverResp, err := cli.get(ctx, "/images/"+imageID+"/json", nil, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return types.ImageInspect{}, nil, wrapResponseError(err, serverResp, "image", imageID)
	}

	body, err := ioutil.ReadAll(serverResp.body)
	if err != nil {
		return types.ImageInspect{}, nil, err
	}

	var response types.ImageInspect
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
