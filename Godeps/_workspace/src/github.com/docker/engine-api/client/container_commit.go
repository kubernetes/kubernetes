package client

import (
	"encoding/json"
	"net/url"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// ContainerCommit applies changes into a container and creates a new tagged image.
func (cli *Client) ContainerCommit(ctx context.Context, options types.ContainerCommitOptions) (types.ContainerCommitResponse, error) {
	query := url.Values{}
	query.Set("container", options.ContainerID)
	query.Set("repo", options.RepositoryName)
	query.Set("tag", options.Tag)
	query.Set("comment", options.Comment)
	query.Set("author", options.Author)
	for _, change := range options.Changes {
		query.Add("changes", change)
	}
	if options.Pause != true {
		query.Set("pause", "0")
	}

	var response types.ContainerCommitResponse
	resp, err := cli.post(ctx, "/commit", query, options.Config, nil)
	if err != nil {
		return response, err
	}

	err = json.NewDecoder(resp.body).Decode(&response)
	ensureReaderClosed(resp)
	return response, err
}
