// Copyright 2014 go-dockerclient authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package docker

import (
	"bufio"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

func TestEventListeners(t *testing.T) {
	response := `{"status":"create","id":"dfdf82bd3881","from":"base:latest","time":1374067924}
{"status":"start","id":"dfdf82bd3881","from":"base:latest","time":1374067924}
{"status":"stop","id":"dfdf82bd3881","from":"base:latest","time":1374067966}
{"status":"destroy","id":"dfdf82bd3881","from":"base:latest","time":1374067970}
`

	var req http.Request
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		rsc := bufio.NewScanner(strings.NewReader(response))
		for rsc.Scan() {
			w.Write([]byte(rsc.Text()))
			w.(http.Flusher).Flush()
			time.Sleep(10 * time.Millisecond)
		}
		req = *r
	}))
	defer server.Close()

	client, err := NewClient(server.URL)
	if err != nil {
		t.Errorf("Failed to create client: %s", err)
	}
	client.SkipServerVersionCheck = true

	listener := make(chan *APIEvents, 10)
	defer func() { time.Sleep(10 * time.Millisecond); client.RemoveEventListener(listener) }()

	err = client.AddEventListener(listener)
	if err != nil {
		t.Errorf("Failed to add event listener: %s", err)
	}

	timeout := time.After(1 * time.Second)
	var count int

	for {
		select {
		case msg := <-listener:
			t.Logf("Recieved: %s", *msg)
			count++
			err = checkEvent(count, msg)
			if err != nil {
				t.Fatalf("Check event failed: %s", err)
			}
			if count == 4 {
				return
			}
		case <-timeout:
			t.Fatal("TestAddEventListener timed out waiting on events")
		}
	}
}

func checkEvent(index int, event *APIEvents) error {
	if event.ID != "dfdf82bd3881" {
		return fmt.Errorf("event ID did not match. Expected dfdf82bd3881 got %s", event.ID)
	}
	if event.From != "base:latest" {
		return fmt.Errorf("event from did not match. Expected base:latest got %s", event.From)
	}
	var status string
	switch index {
	case 1:
		status = "create"
	case 2:
		status = "start"
	case 3:
		status = "stop"
	case 4:
		status = "destroy"
	}
	if event.Status != status {
		return fmt.Errorf("event status did not match. Expected %s got %s", status, event.Status)
	}
	return nil
}
