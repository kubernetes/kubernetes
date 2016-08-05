package client

import "golang.org/x/net/context"

// VolumeRemove removes a volume from the docker host.
func (cli *Client) VolumeRemove(ctx context.Context, volumeID string) error {
	resp, err := cli.delete(ctx, "/volumes/"+volumeID, nil, nil)
	ensureReaderClosed(resp)
	return err
}
