// Copyright 2015 Light Code Labs, LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package telemetry implements the client for server-side telemetry
// of the network. Functions in this package are synchronous and blocking
// unless otherwise specified. For convenience, most functions here do
// not return errors, but errors are logged to the standard logger.
//
// To use this package, first call Init(). You can then call any of the
// collection/aggregation functions. Call StartEmitting() when you are
// ready to begin sending telemetry updates.
//
// When collecting metrics (functions like Set, AppendUnique, or Increment),
// it may be desirable and even recommended to invoke them in a new
// goroutine in case there is lock contention; they are thread-safe (unless
// noted), and you may not want them to block the main thread of execution.
// However, sometimes blocking may be necessary too; for example, adding
// startup metrics to the buffer before the call to StartEmitting().
//
// This package is designed to be as fast and space-efficient as reasonably
// possible, so that it does not disrupt the flow of execution.
package telemetry

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"log"
	"math/rand"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/google/uuid"
)

// logEmit calls emit and then logs the error, if any.
// See docs for emit.
func logEmit(final bool) {
	err := emit(final)
	if err != nil {
		log.Printf("[ERROR] Sending telemetry: %v", err)
	}
}

// emit sends an update to the telemetry server.
// Set final to true if this is the last call to emit.
// If final is true, no future updates will be scheduled.
// Otherwise, the next update will be scheduled.
func emit(final bool) error {
	if !enabled {
		return fmt.Errorf("telemetry not enabled")
	}

	// some metrics are updated/set at time of emission
	setEmitTimeMetrics()

	// ensure only one update happens at a time;
	// skip update if previous one still in progress
	updateMu.Lock()
	if updating {
		updateMu.Unlock()
		log.Println("[NOTICE] Skipping this telemetry update because previous one is still working")
		return nil
	}
	updating = true
	updateMu.Unlock()
	defer func() {
		updateMu.Lock()
		updating = false
		updateMu.Unlock()
	}()

	// terminate any pending update if this is the last one
	if final {
		stopUpdateTimer()
	}

	payloadBytes, err := makePayloadAndResetBuffer()
	if err != nil {
		return err
	}

	// this will hold the server's reply
	var reply Response

	// transmit the payload - use a loop to retry in case of failure
	for i := 0; i < 4; i++ {
		if i > 0 && err != nil {
			// don't hammer the server; first failure might have been
			// a fluke, but back off more after that
			log.Printf("[WARNING] Sending telemetry (attempt %d): %v - backing off and retrying", i, err)
			time.Sleep(time.Duration((i+1)*(i+1)*(i+1)) * time.Second)
		}

		// send it
		var resp *http.Response
		resp, err = httpClient.Post(endpoint+instanceUUID.String(), "application/json", bytes.NewReader(payloadBytes))
		if err != nil {
			continue
		}

		// check for any special-case response codes
		if resp.StatusCode == http.StatusGone {
			// the endpoint has been deprecated and is no longer servicing clients
			err = fmt.Errorf("telemetry server replied with HTTP %d; upgrade required", resp.StatusCode)
			if clen := resp.Header.Get("Content-Length"); clen != "0" && clen != "" {
				bodyBytes, readErr := ioutil.ReadAll(resp.Body)
				if readErr != nil {
					log.Printf("[ERROR] Reading response body from server: %v", readErr)
				}
				err = fmt.Errorf("%v - %s", err, bodyBytes)
			}
			resp.Body.Close()
			reply.Stop = true
			break
		}
		if resp.StatusCode == http.StatusUnavailableForLegalReasons {
			// the endpoint is unavailable, at least to this client, for legal reasons (!)
			err = fmt.Errorf("telemetry server replied with HTTP %d %s: please consult the project website and developers for guidance", resp.StatusCode, resp.Status)
			if clen := resp.Header.Get("Content-Length"); clen != "0" && clen != "" {
				bodyBytes, readErr := ioutil.ReadAll(resp.Body)
				if readErr != nil {
					log.Printf("[ERROR] Reading response body from server: %v", readErr)
				}
				err = fmt.Errorf("%v - %s", err, bodyBytes)
			}
			resp.Body.Close()
			reply.Stop = true
			break
		}

		// okay, ensure we can interpret the response
		if ct := resp.Header.Get("Content-Type"); (resp.StatusCode < 300 || resp.StatusCode >= 400) &&
			!strings.Contains(ct, "json") {
			err = fmt.Errorf("telemetry server replied with unknown content-type: '%s' and HTTP %s", ct, resp.Status)
			resp.Body.Close()
			continue
		}

		// read the response body
		err = json.NewDecoder(resp.Body).Decode(&reply)
		resp.Body.Close() // close response body as soon as we're done with it
		if err != nil {
			continue
		}

		// update the list of enabled/disabled keys, if any
		for _, key := range reply.EnableKeys {
			disabledMetricsMu.Lock()
			// only re-enable this metric if it is temporarily disabled
			if temp, ok := disabledMetrics[key]; ok && temp {
				delete(disabledMetrics, key)
			}
			disabledMetricsMu.Unlock()
		}
		for _, key := range reply.DisableKeys {
			disabledMetricsMu.Lock()
			disabledMetrics[key] = true // all remotely-disabled keys are "temporarily" disabled
			disabledMetricsMu.Unlock()
		}

		// make sure we didn't send the update too soon; if so,
		// just wait and try again -- this is a special case of
		// error that we handle differently, as you can see
		if resp.StatusCode == http.StatusTooManyRequests {
			if reply.NextUpdate <= 0 {
				raStr := resp.Header.Get("Retry-After")
				if ra, err := strconv.Atoi(raStr); err == nil {
					reply.NextUpdate = time.Duration(ra) * time.Second
				}
			}
			if !final {
				log.Printf("[NOTICE] Sending telemetry: we were too early; waiting %s before trying again", reply.NextUpdate)
				time.Sleep(reply.NextUpdate)
				continue
			}
		} else if resp.StatusCode >= 400 {
			err = fmt.Errorf("telemetry server returned status code %d", resp.StatusCode)
			continue
		}

		break
	}
	if err == nil && !final {
		// (remember, if there was an error, we return it
		// below, so it WILL get logged if it's supposed to)
		log.Println("[INFO] Sending telemetry: success")
	}

	// even if there was an error after all retries, we should
	// schedule the next update using our default update
	// interval because the server might be healthy later

	// ensure we won't slam the telemetry server; add a little variance
	if reply.NextUpdate < 1*time.Second {
		reply.NextUpdate = defaultUpdateInterval + time.Duration(rand.Int63n(int64(1*time.Minute)))
	}

	// schedule the next update (if this wasn't the last one and
	// if the remote server didn't tell us to stop sending)
	if !final && !reply.Stop {
		updateTimerMu.Lock()
		updateTimer = time.AfterFunc(reply.NextUpdate, func() {
			logEmit(false)
		})
		updateTimerMu.Unlock()
	}

	return err
}

func stopUpdateTimer() {
	updateTimerMu.Lock()
	updateTimer.Stop()
	updateTimer = nil
	updateTimerMu.Unlock()
}

// setEmitTimeMetrics sets some metrics that should
// be recorded just before emitting.
func setEmitTimeMetrics() {
	Set("goroutines", runtime.NumGoroutine())

	var mem runtime.MemStats
	runtime.ReadMemStats(&mem)
	SetNested("memory", "heap_alloc", mem.HeapAlloc)
	SetNested("memory", "sys", mem.Sys)
}

// makePayloadAndResetBuffer prepares a payload
// by emptying the collection buffer. It returns
// the bytes of the payload to send to the server.
// Since the buffer is reset by this, if the
// resulting byte slice is lost, the payload is
// gone with it.
func makePayloadAndResetBuffer() ([]byte, error) {
	bufCopy := resetBuffer()

	// encode payload in preparation for transmission
	payload := Payload{
		InstanceID: instanceUUID.String(),
		Timestamp:  time.Now().UTC(),
		Data:       bufCopy,
	}
	return json.Marshal(payload)
}

// resetBuffer makes a local pointer to the buffer,
// then resets the buffer by assigning to be a newly-
// made value to clear it out, then sets the buffer
// item count to 0. It returns the copied pointer to
// the original map so the old buffer value can be
// used locally.
func resetBuffer() map[string]interface{} {
	bufferMu.Lock()
	bufCopy := buffer
	buffer = make(map[string]interface{})
	bufferItemCount = 0
	bufferMu.Unlock()
	return bufCopy
}

// Response contains the body of a response from the
// telemetry server.
type Response struct {
	// NextUpdate is how long to wait before the next update.
	NextUpdate time.Duration `json:"next_update"`

	// Stop instructs the telemetry server to stop sending
	// telemetry. This would only be done under extenuating
	// circumstances, but we are prepared for it nonetheless.
	Stop bool `json:"stop,omitempty"`

	// Error will be populated with an error message, if any.
	// This field should be empty if the status code is < 400.
	Error string `json:"error,omitempty"`

	// DisableKeys will contain a list of keys/metrics that
	// should NOT be sent until further notice. The client
	// must NOT store these items in its buffer or send them
	// to the telemetry server while they are disabled. If
	// this list and EnableKeys have the same value (which is
	// not supposed to happen), this field should dominate.
	DisableKeys []string `json:"disable_keys,omitempty"`

	// EnableKeys will contain a list of keys/metrics that
	// MAY be sent until further notice.
	EnableKeys []string `json:"enable_keys,omitempty"`
}

// Payload is the data that gets sent to the telemetry server.
type Payload struct {
	// The universally unique ID of the instance
	InstanceID string `json:"instance_id"`

	// The UTC timestamp of the transmission
	Timestamp time.Time `json:"timestamp"`

	// The timestamp before which the next update is expected
	// (NOT populated by client - the server fills this in
	// before it stores the data)
	ExpectNext time.Time `json:"expect_next,omitempty"`

	// The metrics
	Data map[string]interface{} `json:"data,omitempty"`
}

// Int returns the value of the data keyed by key
// if it is an integer; otherwise it returns 0.
func (p Payload) Int(key string) int {
	val, _ := p.Data[key]
	switch p.Data[key].(type) {
	case int:
		return val.(int)
	case float64: // after JSON-decoding, int becomes float64...
		return int(val.(float64))
	}
	return 0
}

// countingSet implements a set that counts how many
// times a key is inserted. It marshals to JSON in a
// way such that keys are converted to values next
// to their associated counts.
type countingSet map[interface{}]int

// MarshalJSON implements the json.Marshaler interface.
// It converts the set to an array so that the values
// are JSON object values instead of keys, since keys
// are difficult to query in databases.
func (s countingSet) MarshalJSON() ([]byte, error) {
	type Item struct {
		Value interface{} `json:"value"`
		Count int         `json:"count"`
	}
	var list []Item

	for k, v := range s {
		list = append(list, Item{Value: k, Count: v})
	}

	return json.Marshal(list)
}

var (
	// httpClient should be used for HTTP requests. It
	// is configured with a timeout for reliability.
	httpClient = http.Client{
		Transport: &http.Transport{
			TLSHandshakeTimeout: 30 * time.Second,
			DisableKeepAlives:   true,
		},
		Timeout: 1 * time.Minute,
	}

	// buffer holds the data that we are building up to send.
	buffer          = make(map[string]interface{})
	bufferItemCount = 0
	bufferMu        sync.RWMutex // protects both the buffer and its count

	// updating is used to ensure only one
	// update happens at a time.
	updating bool
	updateMu sync.Mutex

	// updateTimer fires off the next update.
	// If no update is scheduled, this is nil.
	updateTimer   *time.Timer
	updateTimerMu sync.Mutex

	// disabledMetrics is a set of metric keys
	// that should NOT be saved to the buffer
	// or sent to the telemetry server. The value
	// indicates whether the entry is temporary.
	// If the value is true, it may be removed if
	// the metric is re-enabled remotely later. If
	// the value is false, it is permanent
	// (presumably because the user explicitly
	// disabled it) and can only be re-enabled
	// with user consent.
	disabledMetrics   = make(map[string]bool)
	disabledMetricsMu sync.RWMutex

	// instanceUUID is the ID of the current instance.
	// This MUST be set to emit telemetry.
	// This MUST NOT be openly exposed to clients, for privacy.
	instanceUUID uuid.UUID

	// enabled indicates whether the package has
	// been initialized and can be actively used.
	enabled bool

	// maxBufferItems is the maximum number of items we'll allow
	// in the buffer before we start dropping new ones, in a
	// rough (simple) attempt to keep memory use under control.
	maxBufferItems = 100000
)

const (
	// endpoint is the base URL to remote telemetry server;
	// the instance ID will be appended to it.
	endpoint = "https://telemetry.caddyserver.com/v1/update/"

	// defaultUpdateInterval is how long to wait before emitting
	// more telemetry data if all retires fail. This value is
	// only used if the client receives a nonsensical value, or
	// doesn't send one at all, or if a connection can't be made,
	// likely indicating a problem with the server. Thus, this
	// value should be a long duration to help alleviate extra
	// load on the server.
	defaultUpdateInterval = 1 * time.Hour
)
