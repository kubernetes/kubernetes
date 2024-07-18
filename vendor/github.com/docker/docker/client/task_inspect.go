package client // import "github.com/docker/docker/client"

import (
	"bytes"
	"context"
	"encoding/json"
	"io"

	"github.com/docker/docker/api/types/swarm"
)

// TaskInspectWithRaw returns the task information and its raw representation..
func (cli *Client) TaskInspectWithRaw(ctx context.Context, taskID string) (swarm.Task, []byte, error) {
	if taskID == "" {
		return swarm.Task{}, nil, objectNotFoundError{object: "task", id: taskID}
	}
	serverResp, err := cli.get(ctx, "/tasks/"+taskID, nil, nil)
	defer ensureReaderClosed(serverResp)
	if err != nil {
		return swarm.Task{}, nil, wrapResponseError(err, serverResp, "task", taskID)
	}

	body, err := io.ReadAll(serverResp.body)
	if err != nil {
		return swarm.Task{}, nil, err
	}

	var response swarm.Task
	rdr := bytes.NewReader(body)
	err = json.NewDecoder(rdr).Decode(&response)
	return response, body, err
}
