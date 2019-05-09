package client // import "github.com/docker/docker/client"

import (
	"context"
	"net/http"
	"path"

	"github.com/docker/docker/api/types"
	"github.com/docker/docker/errdefs"
)

// Ping pings the server and returns the value of the "Docker-Experimental",
// "Builder-Version", "OS-Type" & "API-Version" headers. It attempts to use
// a HEAD request on the endpoint, but falls back to GET if HEAD is not supported
// by the daemon.
func (cli *Client) Ping(ctx context.Context) (types.Ping, error) {
	var ping types.Ping

	// Using cli.buildRequest() + cli.doRequest() instead of cli.sendRequest()
	// because ping requests are used during  API version negotiation, so we want
	// to hit the non-versioned /_ping endpoint, not /v1.xx/_ping
	req, err := cli.buildRequest("HEAD", path.Join(cli.basePath, "/_ping"), nil, nil)
	if err != nil {
		return ping, err
	}
	serverResp, err := cli.doRequest(ctx, req)
	if err == nil {
		defer ensureReaderClosed(serverResp)
		switch serverResp.statusCode {
		case http.StatusOK, http.StatusInternalServerError:
			// Server handled the request, so parse the response
			return parsePingResponse(cli, serverResp)
		}
	}

	req, err = cli.buildRequest("GET", path.Join(cli.basePath, "/_ping"), nil, nil)
	if err != nil {
		return ping, err
	}
	serverResp, err = cli.doRequest(ctx, req)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return ping, err
	}
	return parsePingResponse(cli, serverResp)
}

func parsePingResponse(cli *Client, resp serverResponse) (types.Ping, error) {
	var ping types.Ping
	if resp.header == nil {
		err := cli.checkResponseErr(resp)
		return ping, errdefs.FromStatusCode(err, resp.statusCode)
	}
	ping.APIVersion = resp.header.Get("API-Version")
	ping.OSType = resp.header.Get("OSType")
	if resp.header.Get("Docker-Experimental") == "true" {
		ping.Experimental = true
	}
	if bv := resp.header.Get("Builder-Version"); bv != "" {
		ping.BuilderVersion = types.BuilderVersion(bv)
	}
	err := cli.checkResponseErr(resp)
	return ping, errdefs.FromStatusCode(err, resp.statusCode)
}
