package client

import (
	"encoding/json"
	"errors"
	"net/url"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// ContainerCommit applies changes into a container and creates a new tagged image.
func (cli *Client) ContainerCommit(ctx context.Context, container string, options types.ContainerCommitOptions) (types.IDResponse, error) {
	var repository, tag string
	if options.Reference != "" {
		ref, err := reference.ParseNormalizedNamed(options.Reference)
		if err != nil {
			return types.IDResponse{}, err
		}

		if _, isCanonical := ref.(reference.Canonical); isCanonical {
			return types.IDResponse{}, errors.New("refusing to create a tag with a digest reference")
		}
		ref = reference.TagNameOnly(ref)

		if tagged, ok := ref.(reference.Tagged); ok {
			tag = tagged.Tag()
		}
		repository = reference.FamiliarName(ref)
	}

	query := url.Values{}
	query.Set("container", container)
	query.Set("repo", repository)
	query.Set("tag", tag)
	query.Set("comment", options.Comment)
	query.Set("author", options.Author)
	for _, change := range options.Changes {
		query.Add("changes", change)
	}
	if options.Pause != true {
		query.Set("pause", "0")
	}

	var response types.IDResponse
	resp, err := cli.post(ctx, "/commit", query, options.Config, nil)
	if err != nil {
		return response, err
	}

	err = json.NewDecoder(resp.body).Decode(&response)
	ensureReaderClosed(resp)
	return response, err
}
