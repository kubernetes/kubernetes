package client

import (
	"errors"
	"io"
	"net/http"
	"net/url"

	"golang.org/x/net/context"

	distreference "github.com/docker/distribution/reference"
	"github.com/docker/engine-api/types"
	"github.com/docker/engine-api/types/reference"
)

// ImagePush requests the docker host to push an image to a remote registry.
// It executes the privileged function if the operation is unauthorized
// and it tries one more time.
// It's up to the caller to handle the io.ReadCloser and close it properly.
func (cli *Client) ImagePush(ctx context.Context, ref string, options types.ImagePushOptions) (io.ReadCloser, error) {
	distributionRef, err := distreference.ParseNamed(ref)
	if err != nil {
		return nil, err
	}

	if _, isCanonical := distributionRef.(distreference.Canonical); isCanonical {
		return nil, errors.New("cannot push a digest reference")
	}

	tag := reference.GetTagFromNamedRef(distributionRef)

	query := url.Values{}
	query.Set("tag", tag)

	resp, err := cli.tryImagePush(ctx, distributionRef.Name(), query, options.RegistryAuth)
	if resp.statusCode == http.StatusUnauthorized {
		newAuthHeader, privilegeErr := options.PrivilegeFunc()
		if privilegeErr != nil {
			return nil, privilegeErr
		}
		resp, err = cli.tryImagePush(ctx, distributionRef.Name(), query, newAuthHeader)
	}
	if err != nil {
		return nil, err
	}
	return resp.body, nil
}

func (cli *Client) tryImagePush(ctx context.Context, imageID string, query url.Values, registryAuth string) (*serverResponse, error) {
	headers := map[string][]string{"X-Registry-Auth": {registryAuth}}
	return cli.post(ctx, "/images/"+imageID+"/push", query, nil, headers)
}
