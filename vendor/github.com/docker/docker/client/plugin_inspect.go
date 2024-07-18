package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types"
)

// PluginInspectWithRaw inspects an existing plugin
func (cli *Client) PluginInspectWithRaw(ctx context.Context, name string) (*types.Plugin, []byte, error) {
	if name == "" {
		return nil, nil, objectNotFoundError{object: "plugin", id: name}
	}
	resp, err := cli.get(ctx, "/plugins/"+name+"/json", nil, nil)
	defer ensureReaderClosed(resp)
	if err != nil {
		return nil, nil, wrapResponseError(err, resp, "plugin", name)
	}

	body, err := io.ReadAll(resp.body)
	if err != nil {
		return nil, nil, err
	}
	var p types.Plugin
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&p)
	return &p, body, err
}
