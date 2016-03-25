package client

import (
	"encoding/json"
	"net/http"
	"net/url"

	"github.com/docker/engine-api/types"
	"golang.org/x/net/context"
)

// RegistryLogin authenticates the docker server with a given docker registry.
// It returns UnauthorizerError when the authentication fails.
func (cli *Client) RegistryLogin(ctx context.Context, auth types.AuthConfig) (types.AuthResponse, error) {
	resp, err := cli.post(ctx, "/auth", url.Values{}, auth, nil)

	if resp != nil && resp.statusCode == http.StatusUnauthorized {
		return types.AuthResponse{}, unauthorizedError{err}
	}
	if err != nil {
		return types.AuthResponse{}, err
	}

	var response types.AuthResponse
	err = json.NewDecoder(resp.body).Decode(&response)
	ensureReaderClosed(resp)
	return response, err
}
