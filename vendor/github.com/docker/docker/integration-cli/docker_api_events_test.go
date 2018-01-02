package main

import (
	"encoding/json"
	"io"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/integration-cli/checker"
	"github.com/docker/docker/integration-cli/request"
	"github.com/docker/docker/pkg/jsonmessage"
	"github.com/go-check/check"
)

func (s *DockerSuite) TestEventsAPIEmptyOutput(c *check.C) {
	type apiResp struct {
		resp *http.Response
		err  error
	}
	chResp := make(chan *apiResp)
	go func() {
		resp, body, err := request.Get("/events")
		body.Close()
		chResp <- &apiResp{resp, err}
	}()

	select {
	case r := <-chResp:
		c.Assert(r.err, checker.IsNil)
		c.Assert(r.resp.StatusCode, checker.Equals, http.StatusOK)
	case <-time.After(3 * time.Second):
		c.Fatal("timeout waiting for events api to respond, should have responded immediately")
	}
}

func (s *DockerSuite) TestEventsAPIBackwardsCompatible(c *check.C) {
	since := daemonTime(c).Unix()
	ts := strconv.FormatInt(since, 10)

	out := runSleepingContainer(c, "--name=foo", "-d")
	containerID := strings.TrimSpace(out)
	c.Assert(waitRun(containerID), checker.IsNil)

	q := url.Values{}
	q.Set("since", ts)

	_, body, err := request.Get("/events?" + q.Encode())
	c.Assert(err, checker.IsNil)
	defer body.Close()

	dec := json.NewDecoder(body)
	var containerCreateEvent *jsonmessage.JSONMessage
	for {
		var event jsonmessage.JSONMessage
		if err := dec.Decode(&event); err != nil {
			if err == io.EOF {
				break
			}
			c.Fatal(err)
		}
		if event.Status == "create" && event.ID == containerID {
			containerCreateEvent = &event
			break
		}
	}

	c.Assert(containerCreateEvent, checker.Not(checker.IsNil))
	c.Assert(containerCreateEvent.Status, checker.Equals, "create")
	c.Assert(containerCreateEvent.ID, checker.Equals, containerID)
	c.Assert(containerCreateEvent.From, checker.Equals, "busybox")
}
