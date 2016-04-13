package main

import (
	"net/http"
	"time"

	"github.com/go-check/check"
)

func (s *DockerSuite) TestEventsApiEmptyOutput(c *check.C) {
	type apiResp struct {
		resp *http.Response
		err  error
	}
	chResp := make(chan *apiResp)
	go func() {
		resp, body, err := sockRequestRaw("GET", "/events", nil, "")
		body.Close()
		chResp <- &apiResp{resp, err}
	}()

	select {
	case r := <-chResp:
		c.Assert(r.err, check.IsNil)
		c.Assert(r.resp.StatusCode, check.Equals, http.StatusOK)
	case <-time.After(3 * time.Second):
		c.Fatal("timeout waiting for events api to respond, should have responded immediately")
	}
}
