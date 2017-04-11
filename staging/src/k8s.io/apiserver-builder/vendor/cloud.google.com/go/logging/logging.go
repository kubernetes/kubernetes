// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// Package logging contains a Google Cloud Logging client.
//
// This package is experimental and subject to API changes.
package logging // import "cloud.google.com/go/logging"

import (
	"errors"
	"io"
	"log"
	"sync"
	"time"

	"golang.org/x/net/context"
	api "google.golang.org/api/logging/v1beta3"
	"google.golang.org/api/option"
	"google.golang.org/api/transport"
)

// Scope is the OAuth2 scope necessary to use Google Cloud Logging.
const Scope = api.LoggingWriteScope

// Level is the log level.
type Level int

const (
	// Default means no assigned severity level.
	Default Level = iota
	Debug
	Info
	Warning
	Error
	Critical
	Alert
	Emergency
	nLevel
)

var levelName = [nLevel]string{
	Default:   "",
	Debug:     "DEBUG",
	Info:      "INFO",
	Warning:   "WARNING",
	Error:     "ERROR",
	Critical:  "CRITICAL",
	Alert:     "ALERT",
	Emergency: "EMERGENCY",
}

func (v Level) String() string {
	return levelName[v]
}

// Client is a Google Cloud Logging client.
// It must be constructed via NewClient.
type Client struct {
	svc     *api.Service
	logs    *api.ProjectsLogsEntriesService
	projID  string
	logName string
	writer  [nLevel]io.Writer
	logger  [nLevel]*log.Logger

	mu          sync.Mutex
	queued      []*api.LogEntry
	curFlush    *flushCall  // currently in-flight flush
	flushTimer  *time.Timer // nil before first use
	timerActive bool        // whether flushTimer is armed
	inFlight    int         // number of log entries sent to API service but not yet ACKed

	// For testing:
	timeNow func() time.Time // optional

	// ServiceName may be "appengine.googleapis.com",
	// "compute.googleapis.com" or "custom.googleapis.com".
	//
	// The default is "custom.googleapis.com".
	//
	// The service name is only used by the API server to
	// determine which of the labels are used to index the logs.
	ServiceName string

	// CommonLabels are metadata labels that apply to all log
	// entries in this request, so that you don't have to repeat
	// them in each log entry's metadata.labels field. If any of
	// the log entries contains a (key, value) with the same key
	// that is in CommonLabels, then the entry's (key, value)
	// overrides the one in CommonLabels.
	CommonLabels map[string]string

	// BufferLimit is the maximum number of items to keep in memory
	// before flushing. Zero means automatic. A value of 1 means to
	// flush after each log entry.
	// The default is currently 10,000.
	BufferLimit int

	// FlushAfter optionally specifies a threshold count at which buffered
	// log entries are flushed, even if the BufferInterval has not yet
	// been reached.
	// The default is currently 10.
	FlushAfter int

	// BufferInterval is the maximum amount of time that an item
	// should remain buffered in memory before being flushed to
	// the logging service.
	// The default is currently 1 second.
	BufferInterval time.Duration

	// Overflow is a function which runs when the Log function
	// overflows its configured buffer limit. If nil, the log
	// entry is dropped. The return value from Overflow is
	// returned by Log.
	Overflow func(*Client, Entry) error
}

func (c *Client) flushAfter() int {
	if v := c.FlushAfter; v > 0 {
		return v
	}
	return 10
}

func (c *Client) bufferInterval() time.Duration {
	if v := c.BufferInterval; v > 0 {
		return v
	}
	return time.Second
}

func (c *Client) bufferLimit() int {
	if v := c.BufferLimit; v > 0 {
		return v
	}
	return 10000
}

func (c *Client) serviceName() string {
	if v := c.ServiceName; v != "" {
		return v
	}
	return "custom.googleapis.com"
}

func (c *Client) now() time.Time {
	if now := c.timeNow; now != nil {
		return now()
	}
	return time.Now()
}

// Writer returns an io.Writer for the provided log level.
//
// Each Write call on the returned Writer generates a log entry.
//
// This Writer accessor does not allocate, so callers do not need to
// cache.
func (c *Client) Writer(v Level) io.Writer { return c.writer[v] }

// Logger returns a *log.Logger for the provided log level.
//
// A Logger for each Level is pre-allocated by NewClient with an empty
// prefix and no flags.  This Logger accessor does not allocate.
// Callers wishing to use alternate flags (such as log.Lshortfile) may
// mutate the returned Logger with SetFlags. Such mutations affect all
// callers in the program.
func (c *Client) Logger(v Level) *log.Logger { return c.logger[v] }

type levelWriter struct {
	level Level
	c     *Client
}

func (w levelWriter) Write(p []byte) (n int, err error) {
	return len(p), w.c.Log(Entry{
		Level:   w.level,
		Payload: string(p),
	})
}

// Entry is a log entry.
type Entry struct {
	// Time is the time of the entry. If the zero value, the current time is used.
	Time time.Time

	// Level is log entry's severity level.
	// The zero value means no assigned severity level.
	Level Level

	// Payload must be either a string, []byte, or something that
	// marshals via the encoding/json package to a JSON object
	// (and not any other type of JSON value).
	Payload interface{}

	// Labels optionally specifies key/value labels for the log entry.
	// Depending on the Client's ServiceName, these are indexed differently
	// by the Cloud Logging Service.
	// See https://cloud.google.com/logging/docs/logs_index
	// The Client.Log method takes ownership of this map.
	Labels map[string]string

	// TODO: de-duping id
}

func (c *Client) apiEntry(e Entry) (*api.LogEntry, error) {
	t := e.Time
	if t.IsZero() {
		t = c.now()
	}

	ent := &api.LogEntry{
		Metadata: &api.LogEntryMetadata{
			Timestamp:   t.UTC().Format(time.RFC3339Nano),
			ServiceName: c.serviceName(),
			Severity:    e.Level.String(),
			Labels:      e.Labels,
		},
	}
	switch p := e.Payload.(type) {
	case string:
		ent.TextPayload = p
	case []byte:
		ent.TextPayload = string(p)
	default:
		ent.StructPayload = api.LogEntryStructPayload(p)
	}
	return ent, nil
}

// LogSync logs e synchronously without any buffering.
// This is mostly intended for debugging or critical errors.
func (c *Client) LogSync(e Entry) error {
	ent, err := c.apiEntry(e)
	if err != nil {
		return err
	}
	_, err = c.logs.Write(c.projID, c.logName, &api.WriteLogEntriesRequest{
		CommonLabels: c.CommonLabels,
		Entries:      []*api.LogEntry{ent},
	}).Do()
	return err
}

var ErrOverflow = errors.New("logging: log entry overflowed buffer limits")

// Log queues an entry to be sent to the logging service, subject to the
// Client's parameters. By default, the log will be flushed within
// one second.
// Log only returns an error if the entry is invalid or the queue is at
// capacity. If the queue is at capacity and the entry can't be added,
// Log returns either ErrOverflow when c.Overflow is nil, or the
// value returned by c.Overflow.
func (c *Client) Log(e Entry) error {
	ent, err := c.apiEntry(e)
	if err != nil {
		return err
	}

	c.mu.Lock()
	buffered := len(c.queued) + c.inFlight

	if buffered >= c.bufferLimit() {
		c.mu.Unlock()
		if fn := c.Overflow; fn != nil {
			return fn(c, e)
		}
		return ErrOverflow
	}
	defer c.mu.Unlock()

	c.queued = append(c.queued, ent)
	if len(c.queued) >= c.flushAfter() {
		c.scheduleFlushLocked(0)
		return nil
	}
	c.scheduleFlushLocked(c.bufferInterval())
	return nil
}

// c.mu must be held.
//
// d will be one of two values: either c.BufferInterval (or its
// default value) or 0.
func (c *Client) scheduleFlushLocked(d time.Duration) {
	if c.inFlight > 0 {
		// For now to keep things simple, only allow one HTTP
		// request in flight at a time.
		return
	}
	switch {
	case c.flushTimer == nil:
		// First flush.
		c.timerActive = true
		c.flushTimer = time.AfterFunc(d, c.timeoutFlush)
	case c.timerActive && d == 0:
		// Make it happen sooner.  For example, this is the
		// case of transitioning from a 1 second flush after
		// the 1st item to an immediate flush after the 10th
		// item.
		c.flushTimer.Reset(0)
	case !c.timerActive:
		c.timerActive = true
		c.flushTimer.Reset(d)
	default:
		// else timer was already active, also at d > 0,
		// so we don't touch it and let it fire as previously
		// scheduled.
	}
}

// timeoutFlush runs in its own goroutine (from time.AfterFunc) and
// flushes c.queued.
func (c *Client) timeoutFlush() {
	c.mu.Lock()
	c.timerActive = false
	c.mu.Unlock()
	if err := c.Flush(); err != nil {
		// schedule another try
		// TODO: smarter back-off?
		c.mu.Lock()
		c.scheduleFlushLocked(5 * time.Second)
		c.mu.Unlock()
	}
}

// Ping reports whether the client's connection to Google Cloud Logging and the
// authentication configuration are valid. To accomplish this, Ping writes a
// log entry "ping" to a log named "ping".
func (c *Client) Ping() error {
	ent := &api.LogEntry{
		Metadata: &api.LogEntryMetadata{
			// Identical timestamps required for deduping in addition to identical insert IDs.
			Timestamp:   time.Unix(0, 0).UTC().Format(time.RFC3339Nano),
			ServiceName: c.serviceName(),
		},
		InsertId:    "ping", // dedup, so there is only ever one entry
		TextPayload: "ping",
	}
	_, err := c.logs.Write(c.projID, "ping", &api.WriteLogEntriesRequest{
		Entries: []*api.LogEntry{ent},
	}).Do()
	return err
}

// Flush flushes any buffered log entries.
func (c *Client) Flush() error {
	var numFlush int
	c.mu.Lock()
	for {
		// We're already flushing (or we just started flushing
		// ourselves), so wait for it to finish.
		if f := c.curFlush; f != nil {
			wasEmpty := len(c.queued) == 0
			c.mu.Unlock()
			<-f.donec // wait for it
			numFlush++
			// Terminate whenever there's an error, we've
			// already flushed twice (one that was already
			// in-flight when flush was called, and then
			// one we instigated), or the queue was empty
			// when we released the locked (meaning this
			// in-flight flush removes everything present
			// when Flush was called, and we don't need to
			// kick off a new flush for things arriving
			// afterward)
			if f.err != nil || numFlush == 2 || wasEmpty {
				return f.err
			}
			// Otherwise, re-obtain the lock and loop,
			// starting over with seeing if a flush is in
			// progress, which might've been started by a
			// different goroutine before aquiring this
			// lock again.
			c.mu.Lock()
			continue
		}

		// Terminal case:
		if len(c.queued) == 0 {
			c.mu.Unlock()
			return nil
		}

		c.startFlushLocked()
	}
}

// requires c.mu be held.
func (c *Client) startFlushLocked() {
	if c.curFlush != nil {
		panic("internal error: flush already in flight")
	}
	if len(c.queued) == 0 {
		panic("internal error: no items queued")
	}
	logEntries := c.queued
	c.inFlight = len(logEntries)
	c.queued = nil

	flush := &flushCall{
		donec: make(chan struct{}),
	}
	c.curFlush = flush
	go func() {
		defer close(flush.donec)
		_, err := c.logs.Write(c.projID, c.logName, &api.WriteLogEntriesRequest{
			CommonLabels: c.CommonLabels,
			Entries:      logEntries,
		}).Do()
		flush.err = err
		c.mu.Lock()
		defer c.mu.Unlock()
		c.inFlight = 0
		c.curFlush = nil
		if err != nil {
			c.queued = append(c.queued, logEntries...)
		} else if len(c.queued) > 0 {
			c.scheduleFlushLocked(c.bufferInterval())
		}
	}()

}

const prodAddr = "https://logging.googleapis.com/"

const userAgent = "gcloud-golang-logging/20150922"

// NewClient returns a new log client, logging to the named log in the
// provided project.
//
// The exported fields on the returned client may be modified before
// the client is used for logging. Once log entries are in flight,
// the fields must not be modified.
func NewClient(ctx context.Context, projectID, logName string, opts ...option.ClientOption) (*Client, error) {
	httpClient, endpoint, err := transport.NewHTTPClient(ctx, append([]option.ClientOption{
		option.WithEndpoint(prodAddr),
		option.WithScopes(Scope),
		option.WithUserAgent(userAgent),
	}, opts...)...)
	if err != nil {
		return nil, err
	}
	svc, err := api.New(httpClient)
	if err != nil {
		return nil, err
	}
	svc.BasePath = endpoint
	c := &Client{
		svc:     svc,
		logs:    api.NewProjectsLogsEntriesService(svc),
		logName: logName,
		projID:  projectID,
	}
	for i := range c.writer {
		level := Level(i)
		c.writer[level] = levelWriter{level, c}
		c.logger[level] = log.New(c.writer[level], "", 0)
	}
	return c, nil
}

// flushCall is an in-flight or completed flush.
type flushCall struct {
	donec chan struct{} // closed when response is in
	err   error         // error is valid after wg is Done
}
