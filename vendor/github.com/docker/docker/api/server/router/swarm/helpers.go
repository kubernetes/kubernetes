package swarm

import (
	"fmt"
	"net/http"

	"github.com/docker/docker/api/server/httputils"
	basictypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	"golang.org/x/net/context"
)

// swarmLogs takes an http response, request, and selector, and writes the logs
// specified by the selector to the response
func (sr *swarmRouter) swarmLogs(ctx context.Context, w http.ResponseWriter, r *http.Request, selector *backend.LogSelector) error {
	// Args are validated before the stream starts because when it starts we're
	// sending HTTP 200 by writing an empty chunk of data to tell the client that
	// daemon is going to stream. By sending this initial HTTP 200 we can't report
	// any error after the stream starts (i.e. container not found, wrong parameters)
	// with the appropriate status code.
	stdout, stderr := httputils.BoolValue(r, "stdout"), httputils.BoolValue(r, "stderr")
	if !(stdout || stderr) {
		return fmt.Errorf("Bad parameters: you must choose at least one stream")
	}

	// there is probably a neater way to manufacture the ContainerLogsOptions
	// struct, probably in the caller, to eliminate the dependency on net/http
	logsConfig := &basictypes.ContainerLogsOptions{
		Follow:     httputils.BoolValue(r, "follow"),
		Timestamps: httputils.BoolValue(r, "timestamps"),
		Since:      r.Form.Get("since"),
		Tail:       r.Form.Get("tail"),
		ShowStdout: stdout,
		ShowStderr: stderr,
		Details:    httputils.BoolValue(r, "details"),
	}

	tty := false
	// checking for whether logs are TTY involves iterating over every service
	// and task. idk if there is a better way
	for _, service := range selector.Services {
		s, err := sr.backend.GetService(service, false)
		if err != nil {
			// maybe should return some context with this error?
			return err
		}
		tty = (s.Spec.TaskTemplate.ContainerSpec != nil && s.Spec.TaskTemplate.ContainerSpec.TTY) || tty
	}
	for _, task := range selector.Tasks {
		t, err := sr.backend.GetTask(task)
		if err != nil {
			// as above
			return err
		}
		tty = t.Spec.ContainerSpec.TTY || tty
	}

	msgs, err := sr.backend.ServiceLogs(ctx, selector, logsConfig)
	if err != nil {
		return err
	}

	httputils.WriteLogStream(ctx, w, msgs, logsConfig, !tty)
	return nil
}
