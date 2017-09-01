package client

import (
	"bytes"
	"encoding/json"
	"io/ioutil"
	"net/http"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// PluginInspectWithRaw inspects an existing plugin
func (cli *Client) PluginInspectWithRaw(ctx context.Context, name string) (*types.Plugin, []byte, error) {
	resp, err := cli.get(ctx, "/plugins/"+name+"/json", nil, nil)
	if err != nil {
		if resp.statusCode == http.StatusNotFound {
			return nil, nil, pluginNotFoundError{name}
		}
		return nil, nil, err
	}

	defer ensureReaderClosed(resp)
	body, err := ioutil.ReadAll(resp.body)
	if err != nil {
		return nil, nil, err
	}
	var p types.Plugin
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&p)
	return &p, body, err
}
