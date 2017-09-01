package client

import (
	"fmt"

	"github.com/docker/docker/api/types"
	"golang.org/x/net/context"
)

// Ping pings the server and return the value of the "Docker-Experimental" & "API-Version" headers
func (cli *Client) Ping(ctx context.Context) (types.Ping, error) {
	var ping types.Ping
	req, err := cli.buildRequest("GET", fmt.Sprintf("%s/_ping", cli.basePath), nil, nil)
	if err != nil {
		return ping, err
	}
	serverResp, err := cli.doRequest(ctx, req)
	if err != nil {
		return ping, err
	}
	defer ensureReaderClosed(serverResp)

	ping.APIVersion = serverResp.header.Get("API-Version")

	if serverResp.header.Get("Docker-Experimental") == "true" {
		ping.Experimental = true
	}

	return ping, nil
}
