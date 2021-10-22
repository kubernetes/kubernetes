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

// Package controller is a library for interacting with the Google Cloud Debugger's Debuglet Controller service.
package controller

import (
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"sync"

	"golang.org/x/oauth2"
	cd "google.golang.org/api/clouddebugger/v2"
	"google.golang.org/api/googleapi"
	"google.golang.org/api/option"
	htransport "google.golang.org/api/transport/http"
)

const (
	// agentVersionString identifies the agent to the service.
	agentVersionString = "google.com/go-gcp/v0.2"
	// initWaitToken is the wait token sent in the first Update request to a server.
	initWaitToken = "init"
)

var (
	// ErrListUnchanged is returned by List if the server time limit is reached
	// before the list of breakpoints changes.
	ErrListUnchanged = errors.New("breakpoint list unchanged")
	// ErrDebuggeeDisabled is returned by List or Update if the server has disabled
	// this Debuggee.  The caller can retry later.
	ErrDebuggeeDisabled = errors.New("debuglet disabled by server")
)

// Controller manages a connection to the Debuglet Controller service.
type Controller struct {
	s serviceInterface
	// waitToken is sent with List requests so the server knows which set of
	// breakpoints this client has already seen. Each successful List request
	// returns a new waitToken to send in the next request.
	waitToken string
	// verbose determines whether to do some logging
	verbose bool
	// options, uniquifier and description are used in register.
	options     Options
	uniquifier  string
	description string
	// labels are included when registering the debuggee. They should contain
	// the module name, version and minorversion, and are used by the debug UI
	// to label the correct version active for debugging.
	labels map[string]string
	// mu protects debuggeeID
	mu sync.Mutex
	// debuggeeID is returned from the server on registration, and is passed back
	// to the server in List and Update requests.
	debuggeeID string
}

// Options controls how the Debuglet Controller client identifies itself to the server.
// See https://cloud.google.com/storage/docs/projects and
// https://cloud.google.com/tools/cloud-debugger/setting-up-on-compute-engine
// for further documentation of these parameters.
type Options struct {
	ProjectNumber  string              // GCP Project Number.
	ProjectID      string              // GCP Project ID.
	AppModule      string              // Module name for the debugged program.
	AppVersion     string              // Version number for this module.
	SourceContexts []*cd.SourceContext // Description of source.
	Verbose        bool
	TokenSource    oauth2.TokenSource // Source of Credentials used for Stackdriver Debugger.
}

type serviceInterface interface {
	Register(ctx context.Context, req *cd.RegisterDebuggeeRequest) (*cd.RegisterDebuggeeResponse, error)
	Update(ctx context.Context, debuggeeID, breakpointID string, req *cd.UpdateActiveBreakpointRequest) (*cd.UpdateActiveBreakpointResponse, error)
	List(ctx context.Context, debuggeeID, waitToken string) (*cd.ListActiveBreakpointsResponse, error)
}

var newService = func(ctx context.Context, tokenSource oauth2.TokenSource) (serviceInterface, error) {
	httpClient, endpoint, err := htransport.NewClient(ctx, option.WithTokenSource(tokenSource))
	if err != nil {
		return nil, err
	}
	s, err := cd.New(httpClient)
	if err != nil {
		return nil, err
	}
	if endpoint != "" {
		s.BasePath = endpoint
	}
	return &service{s: s}, nil
}

type service struct {
	s *cd.Service
}

func (s service) Register(ctx context.Context, req *cd.RegisterDebuggeeRequest) (*cd.RegisterDebuggeeResponse, error) {
	call := cd.NewControllerDebuggeesService(s.s).Register(req)
	return call.Context(ctx).Do()
}

func (s service) Update(ctx context.Context, debuggeeID, breakpointID string, req *cd.UpdateActiveBreakpointRequest) (*cd.UpdateActiveBreakpointResponse, error) {
	call := cd.NewControllerDebuggeesBreakpointsService(s.s).Update(debuggeeID, breakpointID, req)
	return call.Context(ctx).Do()
}

func (s service) List(ctx context.Context, debuggeeID, waitToken string) (*cd.ListActiveBreakpointsResponse, error) {
	call := cd.NewControllerDebuggeesBreakpointsService(s.s).List(debuggeeID)
	call.WaitToken(waitToken)
	return call.Context(ctx).Do()
}

// NewController connects to the Debuglet Controller server using the given options,
// and returns a Controller for that connection.
// Google Application Default Credentials are used to connect to the Debuglet Controller;
// see https://developers.google.com/identity/protocols/application-default-credentials
func NewController(ctx context.Context, o Options) (*Controller, error) {
	// We build a JSON encoding of o.SourceContexts so we can hash it.
	scJSON, err := json.Marshal(o.SourceContexts)
	if err != nil {
		scJSON = nil
		o.SourceContexts = nil
	}
	const minorversion = "107157" // any arbitrary numeric string

	// Compute a uniquifier string by hashing the project number, app module name,
	// app module version, debuglet version, and source context.
	// The choice of hash function is arbitrary.
	h := sha256.Sum256([]byte(fmt.Sprintf("%d %s %d %s %d %s %d %s %d %s %d %s",
		len(o.ProjectNumber), o.ProjectNumber,
		len(o.AppModule), o.AppModule,
		len(o.AppVersion), o.AppVersion,
		len(agentVersionString), agentVersionString,
		len(scJSON), scJSON,
		len(minorversion), minorversion)))
	uniquifier := fmt.Sprintf("%X", h[0:16]) // 32 hex characters

	description := o.ProjectID
	if o.AppModule != "" {
		description += "-" + o.AppModule
	}
	if o.AppVersion != "" {
		description += "-" + o.AppVersion
	}

	s, err := newService(ctx, o.TokenSource)
	if err != nil {
		return nil, err
	}

	// Construct client.
	c := &Controller{
		s:           s,
		waitToken:   initWaitToken,
		verbose:     o.Verbose,
		options:     o,
		uniquifier:  uniquifier,
		description: description,
		labels: map[string]string{
			"module":       o.AppModule,
			"version":      o.AppVersion,
			"minorversion": minorversion,
		},
	}

	return c, nil
}

func (c *Controller) getDebuggeeID(ctx context.Context) (string, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.debuggeeID != "" {
		return c.debuggeeID, nil
	}
	// The debuglet hasn't been registered yet, or it is disabled and we should try registering again.
	if err := c.register(ctx); err != nil {
		return "", err
	}
	return c.debuggeeID, nil
}

// List retrieves the current list of breakpoints from the server.
// If the set of breakpoints on the server is the same as the one returned in
// the previous call to List, the server can delay responding until it changes,
// and return an error instead if no change occurs before a time limit the
// server sets.  List can't be called concurrently with itself.
func (c *Controller) List(ctx context.Context) (*cd.ListActiveBreakpointsResponse, error) {
	id, err := c.getDebuggeeID(ctx)
	if err != nil {
		return nil, err
	}
	resp, err := c.s.List(ctx, id, c.waitToken)
	if err != nil {
		if isAbortedError(err) {
			return nil, ErrListUnchanged
		}
		// For other errors, the protocol requires that we attempt to re-register.
		c.mu.Lock()
		defer c.mu.Unlock()
		if regError := c.register(ctx); regError != nil {
			return nil, regError
		}
		return nil, err
	}
	if resp == nil {
		return nil, errors.New("no response")
	}
	if c.verbose {
		log.Printf("List response: %v", resp)
	}
	c.waitToken = resp.NextWaitToken
	return resp, nil
}

// isAbortedError tests if err is a *googleapi.Error, that it contains one error
// in Errors, and that that error's Reason is "aborted".
func isAbortedError(err error) bool {
	e, _ := err.(*googleapi.Error)
	if e == nil {
		return false
	}
	if len(e.Errors) != 1 {
		return false
	}
	return e.Errors[0].Reason == "aborted"
}

// Update reports information to the server about a breakpoint that was hit.
// Update can be called concurrently with List and Update.
func (c *Controller) Update(ctx context.Context, breakpointID string, bp *cd.Breakpoint) error {
	req := &cd.UpdateActiveBreakpointRequest{Breakpoint: bp}
	if c.verbose {
		log.Printf("sending update for %s: %v", breakpointID, req)
	}
	id, err := c.getDebuggeeID(ctx)
	if err != nil {
		return err
	}
	_, err = c.s.Update(ctx, id, breakpointID, req)
	return err
}

// register calls the Debuglet Controller Register method, and sets c.debuggeeID.
// c.mu should be locked while calling this function.  List and Update can't
// make progress until it returns.
func (c *Controller) register(ctx context.Context) error {
	req := cd.RegisterDebuggeeRequest{
		Debuggee: &cd.Debuggee{
			AgentVersion:   agentVersionString,
			Description:    c.description,
			Project:        c.options.ProjectNumber,
			SourceContexts: c.options.SourceContexts,
			Uniquifier:     c.uniquifier,
			Labels:         c.labels,
		},
	}
	resp, err := c.s.Register(ctx, &req)
	if err != nil {
		return err
	}
	if resp == nil {
		return errors.New("register: no response")
	}
	if resp.Debuggee.IsDisabled {
		// Setting c.debuggeeID to empty makes sure future List and Update calls
		// will call register first.
		c.debuggeeID = ""
	} else {
		c.debuggeeID = resp.Debuggee.Id
	}
	if c.debuggeeID == "" {
		return ErrDebuggeeDisabled
	}
	return nil
}
