package client // import "github.com/docker/docker/client"

import (
	"context"
	"net/url"

	"github.com/docker/distribution/reference"
	"github.com/pkg/errors"
)

// ImageTag tags an image in the docker host
func (cli *Client) ImageTag(ctx context.Context, source, target string) error {
	if _, err := reference.ParseAnyReference(source); err != nil {
		return errors.Wrapf(err, "Error parsing reference: %q is not a valid repository/tag", source)
	}

	ref, err := reference.ParseNormalizedNamed(target)
	if err != nil {
		return errors.Wrapf(err, "Error parsing reference: %q is not a valid repository/tag", target)
	}

	if _, isCanonical := ref.(reference.Canonical); isCanonical {
		return errors.New("refusing to create a tag with a digest reference")
	}

	ref = reference.TagNameOnly(ref)

	query := url.Values{}
	query.Set("repo", reference.FamiliarName(ref))
	if tagged, ok := ref.(reference.Tagged); ok {
		query.Set("tag", tagged.Tag())
	}

	resp, err := cli.post(ctx, "/images/"+source+"/tag", query, nil, nil)
	ensureReaderClosed(resp)
	return err
}
