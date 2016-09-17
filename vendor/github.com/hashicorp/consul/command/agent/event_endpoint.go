package agent

import (
	"bytes"
	"io"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/hashicorp/consul/consul/structs"
)

const (
	// maxQueryTime is used to bound the limit of a blocking query
	maxQueryTime = 600 * time.Second
)

// EventFire is used to fire a new event
func (s *HTTPServer) EventFire(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Mandate a PUT request
	if req.Method != "PUT" {
		resp.WriteHeader(405)
		return nil, nil
	}

	// Get the datacenter
	var dc string
	s.parseDC(req, &dc)

	event := &UserEvent{}
	event.Name = strings.TrimPrefix(req.URL.Path, "/v1/event/fire/")
	if event.Name == "" {
		resp.WriteHeader(400)
		resp.Write([]byte("Missing name"))
		return nil, nil
	}

	// Get the ACL token
	var token string
	s.parseToken(req, &token)

	// Get the filters
	if filt := req.URL.Query().Get("node"); filt != "" {
		event.NodeFilter = filt
	}
	if filt := req.URL.Query().Get("service"); filt != "" {
		event.ServiceFilter = filt
	}
	if filt := req.URL.Query().Get("tag"); filt != "" {
		event.TagFilter = filt
	}

	// Get the payload
	if req.ContentLength > 0 {
		var buf bytes.Buffer
		if _, err := io.Copy(&buf, req.Body); err != nil {
			return nil, err
		}
		event.Payload = buf.Bytes()
	}

	// Try to fire the event
	if err := s.agent.UserEvent(dc, token, event); err != nil {
		if strings.Contains(err.Error(), permissionDenied) {
			resp.WriteHeader(403)
			resp.Write([]byte(permissionDenied))
			return nil, nil
		}
		resp.WriteHeader(500)
		return nil, err
	}

	// Return the event
	return event, nil
}

// EventList is used to retrieve the recent list of events
func (s *HTTPServer) EventList(resp http.ResponseWriter, req *http.Request) (interface{}, error) {
	// Parse the query options, since we simulate a blocking query
	var b structs.QueryOptions
	if parseWait(resp, req, &b) {
		return nil, nil
	}

	// Look for a name filter
	var nameFilter string
	if filt := req.URL.Query().Get("name"); filt != "" {
		nameFilter = filt
	}

	// Lots of this logic is borrowed from consul/rpc.go:blockingRPC
	// However we cannot use that directly since this code has some
	// slight semantics differences...
	var timeout <-chan time.Time
	var notifyCh chan struct{}

	// Fast path non-blocking
	if b.MinQueryIndex == 0 {
		goto RUN_QUERY
	}

	// Restrict the max query time
	if b.MaxQueryTime > maxQueryTime {
		b.MaxQueryTime = maxQueryTime
	}

	// Ensure a time limit is set if we have an index
	if b.MinQueryIndex > 0 && b.MaxQueryTime == 0 {
		b.MaxQueryTime = maxQueryTime
	}

	// Setup a query timeout
	if b.MaxQueryTime > 0 {
		timeout = time.After(b.MaxQueryTime)
	}

	// Setup a notification channel for changes
SETUP_NOTIFY:
	if b.MinQueryIndex > 0 {
		notifyCh = make(chan struct{}, 1)
		s.agent.eventNotify.Wait(notifyCh)
	}

RUN_QUERY:
	// Get the recent events
	events := s.agent.UserEvents()

	// Filter the events if necessary
	if nameFilter != "" {
		for i := 0; i < len(events); i++ {
			if events[i].Name != nameFilter {
				events = append(events[:i], events[i+1:]...)
				i--
			}
		}
	}

	// Determine the index
	var index uint64
	if len(events) == 0 {
		// Return a non-zero index to prevent a hot query loop. This
		// can be caused by a watch for example when there is no matching
		// events.
		index = 1
	} else {
		last := events[len(events)-1]
		index = uuidToUint64(last.ID)
	}
	setIndex(resp, index)

	// Check for exact match on the query value. Because
	// the index value is not monotonic, we just ensure it is
	// not an exact match.
	if index > 0 && index == b.MinQueryIndex {
		select {
		case <-notifyCh:
			goto SETUP_NOTIFY
		case <-timeout:
		}
	}
	return events, nil
}

// uuidToUint64 is a bit of a hack to generate a 64bit Consul index.
// In effect, we take our random UUID, convert it to a 128 bit number,
// then XOR the high-order and low-order 64bit's together to get the
// output. This lets us generate an index which can be used to simulate
// the blocking behavior of other catalog endpoints.
func uuidToUint64(uuid string) uint64 {
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
