// Copyright 2016 Google LLC
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

// Package errorreporting is a Google Stackdriver Error Reporting library.
//
// Any provided stacktraces must match the format produced by https://golang.org/pkg/runtime/#Stack
// or as per https://cloud.google.com/error-reporting/reference/rest/v1beta1/projects.events/report#ReportedErrorEvent
// for language specific stacktrace formats.
//
// This package is still experimental and subject to change.
//
// See https://cloud.google.com/error-reporting/ for more information.
package errorreporting // import "cloud.google.com/go/errorreporting"

import (
	"bytes"
	"context"
	"fmt"
	"log"
	"net/http"
	"runtime"
	"time"

	vkit "cloud.google.com/go/errorreporting/apiv1beta1"
	"cloud.google.com/go/internal/version"
	"github.com/golang/protobuf/ptypes"
	gax "github.com/googleapis/gax-go/v2"
	"google.golang.org/api/option"
	"google.golang.org/api/support/bundler"
	pb "google.golang.org/genproto/googleapis/devtools/clouderrorreporting/v1beta1"
)

// Config is additional configuration for Client.
type Config struct {
	// ServiceName identifies the running program and is included in the error reports.
	// Optional.
	ServiceName string

	// ServiceVersion identifies the version of the running program and is
	// included in the error reports.
	// Optional.
	ServiceVersion string

	// OnError is the function to call if any background
	// tasks errored. By default, errors are logged.
	OnError func(err error)
}

// Entry holds information about the reported error.
type Entry struct {
	Error error
	Req   *http.Request // if error is associated with a request.
	User  string        // an identifier for the user affected by the error

	// Stack specifies the stacktrace and call sequence correlated with
	// the error. Stack's content must match the format specified by
	// https://cloud.google.com/error-reporting/reference/rest/v1beta1/projects.events/report#ReportedErrorEvent.message
	// or at least for Go programs, it must match the format produced
	// by https://golang.org/pkg/runtime/debug/#Stack.
	//
	// If Stack is blank, the result of runtime.Stack will be used instead.
	Stack []byte
}

// Client represents a Google Cloud Error Reporting client.
type Client struct {
	projectName    string
	apiClient      client
	serviceContext *pb.ServiceContext
	bundler        *bundler.Bundler

	onErrorFn func(err error)
}

var newClient = func(ctx context.Context, opts ...option.ClientOption) (client, error) {
	client, err := vkit.NewReportErrorsClient(ctx, opts...)
	if err != nil {
		return nil, err
	}
	client.SetGoogleClientInfo("gccl", version.Repo)
	return client, nil
}

// NewClient returns a new error reporting client. Generally you will want
// to create a client on program initialization and use it through the lifetime
// of the process.
func NewClient(ctx context.Context, projectID string, cfg Config, opts ...option.ClientOption) (*Client, error) {
	if cfg.ServiceName == "" {
		cfg.ServiceName = "goapp"
	}
	c, err := newClient(ctx, opts...)
	if err != nil {
		return nil, fmt.Errorf("creating client: %v", err)
	}

	client := &Client{
		apiClient:   c,
		projectName: "projects/" + projectID,
		serviceContext: &pb.ServiceContext{
			Service: cfg.ServiceName,
			Version: cfg.ServiceVersion,
		},
		onErrorFn: cfg.OnError,
	}
	bundler := bundler.NewBundler((*pb.ReportErrorEventRequest)(nil), func(bundle interface{}) {
		reqs := bundle.([]*pb.ReportErrorEventRequest)
		for _, req := range reqs {
			_, err = client.apiClient.ReportErrorEvent(ctx, req)
			if err != nil {
				client.onError(err)
			}
		}
	})
	// TODO(jbd): Optimize bundler limits.
	bundler.DelayThreshold = 2 * time.Second
	bundler.BundleCountThreshold = 100
	bundler.BundleByteThreshold = 1000
	bundler.BundleByteLimit = 1000
	bundler.BufferedByteLimit = 10000
	client.bundler = bundler
	return client, nil
}

func (c *Client) onError(err error) {
	if c.onErrorFn != nil {
		c.onErrorFn(err)
		return
	}
	log.Println(err)
}

// Close calls Flush, then closes any resources held by the client.
// Close should be called when the client is no longer needed.
func (c *Client) Close() error {
	c.Flush()
	return c.apiClient.Close()
}

// Report writes an error report. It doesn't block. Errors in
// writing the error report can be handled via Config.OnError.
func (c *Client) Report(e Entry) {
	c.bundler.Add(c.newRequest(e), 1)
}

// ReportSync writes an error report. It blocks until the entry is written.
func (c *Client) ReportSync(ctx context.Context, e Entry) error {
	_, err := c.apiClient.ReportErrorEvent(ctx, c.newRequest(e))
	return err
}

// Flush blocks until all currently buffered error reports are sent.
//
// If any errors occurred since the last call to Flush, or the
// creation of the client if this is the first call, then Flush reports the
// error via the Config.OnError handler.
func (c *Client) Flush() {
	c.bundler.Flush()
}

func (c *Client) newRequest(e Entry) *pb.ReportErrorEventRequest {
	var stack string
	if e.Stack != nil {
		stack = string(e.Stack)
	} else {
		// limit the stack trace to 16k.
		var buf [16 * 1024]byte
		stack = chopStack(buf[0:runtime.Stack(buf[:], false)])
	}
	message := e.Error.Error() + "\n" + stack

	var errorContext *pb.ErrorContext
	if r := e.Req; r != nil {
		errorContext = &pb.ErrorContext{
			HttpRequest: &pb.HttpRequestContext{
				Method:    r.Method,
				Url:       r.Host + r.RequestURI,
				UserAgent: r.UserAgent(),
				Referrer:  r.Referer(),
				RemoteIp:  r.RemoteAddr,
			},
		}
	}
	if e.User != "" {
		if errorContext == nil {
			errorContext = &pb.ErrorContext{}
		}
		errorContext.User = e.User
	}
	return &pb.ReportErrorEventRequest{
		ProjectName: c.projectName,
		Event: &pb.ReportedErrorEvent{
			EventTime:      ptypes.TimestampNow(),
			ServiceContext: c.serviceContext,
			Message:        message,
			Context:        errorContext,
		},
	}
}

// chopStack trims a stack trace so that the function which panics or calls
// Report is first.
func chopStack(s []byte) string {
	f := []byte("cloud.google.com/go/errorreporting.(*Client).Report")

	lfFirst := bytes.IndexByte(s, '\n')
	if lfFirst == -1 {
		return string(s)
	}
	stack := s[lfFirst:]
	panicLine := bytes.Index(stack, f)
	if panicLine == -1 {
		return string(s)
	}
	stack = stack[panicLine+1:]
	for i := 0; i < 2; i++ {
		nextLine := bytes.IndexByte(stack, '\n')
		if nextLine == -1 {
			return string(s)
		}
		stack = stack[nextLine+1:]
	}
	return string(s[:lfFirst+1]) + string(stack)
}

type client interface {
	ReportErrorEvent(ctx context.Context, req *pb.ReportErrorEventRequest, opts ...gax.CallOption) (*pb.ReportErrorEventResponse, error)
	Close() error
}
