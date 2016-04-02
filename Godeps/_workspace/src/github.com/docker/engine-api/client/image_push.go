package client

import (
	"io"
	"net/http"
	"net/url"

	"golang.org/x/net/context"

	"github.com/docker/engine-api/types"
)

// ImagePush request the docker host to push an image to a remote registry.
// It executes the privileged function if the operation is unauthorized
// and it tries one more time.
// It's up to the caller to handle the io.ReadCloser and close it properly.
func (cli *Client) ImagePush(ctx context.Context, options types.ImagePushOptions, privilegeFunc RequestPrivilegeFunc) (io.ReadCloser, error) {
	query := url.Values{}
	query.Set("tag", options.Tag)

	resp, err := cli.tryImagePush(ctx, options.ImageID, query, options.RegistryAuth)
	if resp.statusCode == http.StatusUnauthorized {
		newAuthHeader, privilegeErr := privilegeFunc()
		if privilegeErr != nil {
			return nil, privilegeErr
		}
		resp, err = cli.tryImagePush(ctx, options.ImageID, query, newAuthHeader)
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
