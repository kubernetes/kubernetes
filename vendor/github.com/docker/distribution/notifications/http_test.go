package notifications

import (
	"crypto/tls"
	"encoding/json"
	"fmt"
	"mime"
	"net"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strconv"
	"strings"
	"testing"

	"github.com/docker/distribution/manifest/schema1"
)

// TestHTTPSink mocks out an http endpoint and notifies it under a couple of
// conditions, ensuring correct behavior.
func TestHTTPSink(t *testing.T) {
	serverHandler := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		if r.Method != "POST" {
			w.WriteHeader(http.StatusMethodNotAllowed)
			t.Fatalf("unexpected request method: %v", r.Method)
			return
		}

		// Extract the content type and make sure it matches
		contentType := r.Header.Get("Content-Type")
		mediaType, _, err := mime.ParseMediaType(contentType)
		if err != nil {
			w.WriteHeader(http.StatusBadRequest)
			t.Fatalf("error parsing media type: %v, contenttype=%q", err, contentType)
			return
		}

		if mediaType != EventsMediaType {
			w.WriteHeader(http.StatusUnsupportedMediaType)
			t.Fatalf("incorrect media type: %q != %q", mediaType, EventsMediaType)
			return
		}

		var envelope Envelope
		dec := json.NewDecoder(r.Body)
		if err := dec.Decode(&envelope); err != nil {
			w.WriteHeader(http.StatusBadRequest)
			t.Fatalf("error decoding request body: %v", err)
			return
		}

		// Let caller choose the status
		status, err := strconv.Atoi(r.FormValue("status"))
		if err != nil {
			t.Logf("error parsing status: %v", err)

			// May just be empty, set status to 200
			status = http.StatusOK
		}

		w.WriteHeader(status)
	})
	server := httptest.NewTLSServer(serverHandler)

	metrics := newSafeMetrics()
	sink := newHTTPSink(server.URL, 0, nil, nil,
		&endpointMetricsHTTPStatusListener{safeMetrics: metrics})

	// first make sure that the default transport gives x509 untrusted cert error
	events := []Event{}
	err := sink.Write(events...)
	if !strings.Contains(err.Error(), "x509") {
		t.Fatal("TLS server with default transport should give unknown CA error")
	}
	if err := sink.Close(); err != nil {
		t.Fatalf("unexpected error closing http sink: %v", err)
	}

	// make sure that passing in the transport no longer gives this error
	tr := &http.Transport{
		TLSClientConfig: &tls.Config{InsecureSkipVerify: true},
	}
	sink = newHTTPSink(server.URL, 0, nil, tr,
		&endpointMetricsHTTPStatusListener{safeMetrics: metrics})
	err = sink.Write(events...)
	if err != nil {
		t.Fatalf("unexpected error writing events: %v", err)
	}

	// reset server to standard http server and sink to a basic sink
	server = httptest.NewServer(serverHandler)
	sink = newHTTPSink(server.URL, 0, nil, nil,
		&endpointMetricsHTTPStatusListener{safeMetrics: metrics})
	var expectedMetrics EndpointMetrics
	expectedMetrics.Statuses = make(map[string]int)

	closeL, err := net.Listen("tcp", "localhost:0")
	if err != nil {
		t.Fatalf("unexpected error creating listener: %v", err)
	}
	defer closeL.Close()
	go func() {
		for {
			c, err := closeL.Accept()
			if err != nil {
				return
			}
			c.Close()
		}
	}()

	for _, tc := range []struct {
		events     []Event // events to send
		url        string
		failure    bool // true if there should be a failure.
		statusCode int  // if not set, no status code should be incremented.
	}{
		{
			statusCode: http.StatusOK,
			events: []Event{
				createTestEvent("push", "library/test", schema1.MediaTypeSignedManifest)},
		},
		{
			statusCode: http.StatusOK,
			events: []Event{
				createTestEvent("push", "library/test", schema1.MediaTypeSignedManifest),
				createTestEvent("push", "library/test", layerMediaType),
				createTestEvent("push", "library/test", layerMediaType),
			},
		},
		{
			statusCode: http.StatusTemporaryRedirect,
		},
		{
			statusCode: http.StatusBadRequest,
			failure:    true,
		},
		{
			// Case where connection is immediately closed
			url:     closeL.Addr().String(),
			failure: true,
		},
	} {

		if tc.failure {
			expectedMetrics.Failures += len(tc.events)
		} else {
			expectedMetrics.Successes += len(tc.events)
		}

		if tc.statusCode > 0 {
			expectedMetrics.Statuses[fmt.Sprintf("%d %s", tc.statusCode, http.StatusText(tc.statusCode))] += len(tc.events)
		}

		url := tc.url
		if url == "" {
			url = server.URL + "/"
		}
		// setup endpoint to respond with expected status code.
		url += fmt.Sprintf("?status=%v", tc.statusCode)
		sink.url = url

		t.Logf("testcase: %v, fail=%v", url, tc.failure)
		// Try a simple event emission.
		err := sink.Write(tc.events...)

		if !tc.failure {
			if err != nil {
				t.Fatalf("unexpected error send event: %v", err)
			}
		} else {
			if err == nil {
				t.Fatalf("the endpoint should have rejected the request")
			}
		}

		if !reflect.DeepEqual(metrics.EndpointMetrics, expectedMetrics) {
			t.Fatalf("metrics not as expected: %#v != %#v", metrics.EndpointMetrics, expectedMetrics)
		}
	}

	if err := sink.Close(); err != nil {
		t.Fatalf("unexpected error closing http sink: %v", err)
	}

	// double close returns error
	if err := sink.Close(); err == nil {
		t.Fatalf("second close should have returned error: %v", err)
	}

}

func createTestEvent(action, repo, typ string) Event {
	event := createEvent(action)

	event.Target.MediaType = typ
	event.Target.Repository = repo

	return *event
}
