package client

import (
	"encoding/json"
	"net/url"

	"golang.org/x/net/context"

	"github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/versions"
)

// ContainerWait waits until the specified container is in a certain state
// indicated by the given condition, either "not-running" (default),
// "next-exit", or "removed".
//
// If this client's API version is before 1.30, condition is ignored and
// ContainerWait will return immediately with the two channels, as the server
// will wait as if the condition were "not-running".
//
// If this client's API version is at least 1.30, ContainerWait blocks until
// the request has been acknowledged by the server (with a response header),
// then returns two channels on which the caller can wait for the exit status
// of the container or an error if there was a problem either beginning the
// wait request or in getting the response. This allows the caller to
// synchronize ContainerWait with other calls, such as specifying a
// "next-exit" condition before issuing a ContainerStart request.
func (cli *Client) ContainerWait(ctx context.Context, containerID string, condition container.WaitCondition) (<-chan container.ContainerWaitOKBody, <-chan error) {
	if versions.LessThan(cli.ClientVersion(), "1.30") {
		return cli.legacyContainerWait(ctx, containerID)
	}

	resultC := make(chan container.ContainerWaitOKBody)
	errC := make(chan error, 1)

	query := url.Values{}
	query.Set("condition", string(condition))

	resp, err := cli.post(ctx, "/containers/"+containerID+"/wait", query, nil, nil)
	if err != nil {
		defer ensureReaderClosed(resp)
		errC <- err
		return resultC, errC
	}

	go func() {
		defer ensureReaderClosed(resp)
		var res container.ContainerWaitOKBody
		if err := json.NewDecoder(resp.body).Decode(&res); err != nil {
			errC <- err
			return
		}

		resultC <- res
	}()

	return resultC, errC
}

// legacyContainerWait returns immediately and doesn't have an option to wait
// until the container is removed.
func (cli *Client) legacyContainerWait(ctx context.Context, containerID string) (<-chan container.ContainerWaitOKBody, <-chan error) {
	resultC := make(chan container.ContainerWaitOKBody)
	errC := make(chan error)

	go func() {
		resp, err := cli.post(ctx, "/containers/"+containerID+"/wait", nil, nil, nil)
		if err != nil {
			errC <- err
			return
		}
		defer ensureReaderClosed(resp)

		var res container.ContainerWaitOKBody
		if err := json.NewDecoder(resp.body).Decode(&res); err != nil {
			errC <- err
			return
		}

		resultC <- res
	}()

	return resultC, errC
}
