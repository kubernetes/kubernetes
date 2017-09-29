package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/docker/api/types/image"
	"golang.org/x/net/context"
)

// ImageHistory returns the changes in an image in history format.
func (cli *Client) ImageHistory(ctx context.Context, imageID string) ([]image.HistoryResponseItem, error) {
	var history []image.HistoryResponseItem
	serverResp, err := cli.get(ctx, "/images/"+imageID+"/history", url.Values{}, nil)
	if err != nil {
		return history, err
	}

	err = json.NewDecoder(serverResp.body).Decode(&history)
	ensureReaderClosed(serverResp)
	return history, err
}
