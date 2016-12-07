package api

import (
	"bytes"
	"strconv"
)

// Event can be used to query the Event endpoints
type Event struct {
	c *Client
}

// UserEvent represents an event that was fired by the user
type UserEvent struct {
	ID            string
	Name          string
	Payload       []byte
	NodeFilter    string
	ServiceFilter string
	TagFilter     string
	Version       int
	LTime         uint64
}

// Event returns a handle to the event endpoints
func (c *Client) Event() *Event {
	return &Event{c}
}

// Fire is used to fire a new user event. Only the Name, Payload and Filters
// are respected. This returns the ID or an associated error. Cross DC requests
// are supported.
func (e *Event) Fire(params *UserEvent, q *WriteOptions) (string, *WriteMeta, error) {
	r := e.c.newRequest("PUT", "/v1/event/fire/"+params.Name)
	r.setWriteOptions(q)
	if params.NodeFilter != "" {
		r.params.Set("node", params.NodeFilter)
	}
	if params.ServiceFilter != "" {
		r.params.Set("service", params.ServiceFilter)
	}
	if params.TagFilter != "" {
		r.params.Set("tag", params.TagFilter)
	}
	if params.Payload != nil {
		r.body = bytes.NewReader(params.Payload)
	}

	rtt, resp, err := requireOK(e.c.doRequest(r))
	if err != nil {
		return "", nil, err
	}
	defer resp.Body.Close()

	wm := &WriteMeta{RequestTime: rtt}
	var out UserEvent
	if err := decodeBody(resp, &out); err != nil {
		return "", nil, err
	}
	return out.ID, wm, nil
}

// List is used to get the most recent events an agent has received.
// This list can be optionally filtered by the name. This endpoint supports
// quasi-blocking queries. The index is not monotonic, nor does it provide provide
// LastContact or KnownLeader.
func (e *Event) List(name string, q *QueryOptions) ([]*UserEvent, *QueryMeta, error) {
	r := e.c.newRequest("GET", "/v1/event/list")
	r.setQueryOptions(q)
	if name != "" {
		r.params.Set("name", name)
	}
	rtt, resp, err := requireOK(e.c.doRequest(r))
	if err != nil {
		return nil, nil, err
	}
	defer resp.Body.Close()

	qm := &QueryMeta{}
	parseQueryMeta(resp, qm)
	qm.RequestTime = rtt

	var entries []*UserEvent
	if err := decodeBody(resp, &entries); err != nil {
		return nil, nil, err
	}
	return entries, qm, nil
}

// IDToIndex is a bit of a hack. This simulates the index generation to
// convert an event ID into a WaitIndex.
func (e *Event) IDToIndex(uuid string) uint64 {
	lower := uuid[0:8] + uuid[9:13] + uuid[14:18]
	upper := uuid[19:23] + uuid[24:36]
	lowVal, err := strconv.ParseUint(lower, 16, 64)
	if err != nil {
		panic("Failed to convert " + lower)
	}
	highVal, err := strconv.ParseUint(upper, 16, 64)
	if err != nil {
		panic("Failed to convert " + upper)
	}
	return lowVal ^ highVal
}
