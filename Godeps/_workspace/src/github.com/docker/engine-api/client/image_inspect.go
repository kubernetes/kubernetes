package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"
	"net/url"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// ImageInspectWithRaw returns the image information and its raw representation.
func (cli *Client) ImageInspectWithRaw(ctx context.Context, imageID string, getSize bool) (types.ImageInspect, []byte, error) {
	query := url.Values{}
	if getSize {
		query.Set("size", "1")
	}
	serverResp, err := cli.get(ctx, "/images/"+imageID+"/json", query, nil)
	if err != nil {
		if serverResp.statusCode == http.StatusNotFound {
			return types.ImageInspect{}, nil, imageNotFoundError{imageID}
		}
		return types.ImageInspect{}, nil, err
	}
	defer ensureReaderClosed(serverResp)

	body, err := ioutil.ReadAll(serverResp.body)
	if err != nil {
		return types.ImageInspect{}, nil, err
	}

	var response types.ImageInspect
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
