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

package controller

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"strconv"
	"testing"

	"golang.org/x/oauth2"
	cd "google.golang.org/api/clouddebugger/v2"
	"google.golang.org/api/googleapi"
)

const (
	testDebuggeeID   = "d12345"
	testBreakpointID = "bp12345"
)

var (
	// The sequence of wait tokens in List requests and responses.
	expectedWaitToken = []string{"init", "token1", "token2", "token1", "token1"}
	// The set of breakpoints returned from each List call.
	expectedBreakpoints = [][]*cd.Breakpoint{
		nil,
		{
			&cd.Breakpoint{
				Id:           testBreakpointID,
				IsFinalState: false,
				Location:     &cd.SourceLocation{Line: 42, Path: "foo.go"},
			},
		},
		nil,
	}
	abortedError error = &googleapi.Error{
		Code:    409,
		Message: "Conflict",
		Body: `{
		 "error": {
		  "errors": [
		   {
		    "domain": "global",
		    "reason": "aborted",
		    "message": "Conflict"
		   }
		  ],
		  "code": 409,
		  "message": "Conflict"
		 }
		}`,
		Errors: []googleapi.ErrorItem{
			{Reason: "aborted", Message: "Conflict"},
		},
	}
	backendError error = &googleapi.Error{
		Code:    503,
		Message: "Backend Error",
		Body: `{
		 "error": {
		  "errors": [
		   {
		    "domain": "global",
		    "reason": "backendError",
		    "message": "Backend Error"
		   }
		  ],
		  "code": 503,
		  "message": "Backend Error"
		 }
		}`,
		Errors: []googleapi.ErrorItem{
			{Reason: "backendError", Message: "Backend Error"},
		},
	}
)

type mockService struct {
	t                 *testing.T
	listCallsSeen     int
	registerCallsSeen int
}

func (s *mockService) Register(ctx context.Context, req *cd.RegisterDebuggeeRequest) (*cd.RegisterDebuggeeResponse, error) {
	s.registerCallsSeen++
	if req.Debuggee == nil {
		s.t.Errorf("missing debuggee")
		return nil, nil
	}
	if req.Debuggee.AgentVersion == "" {
		s.t.Errorf("missing agent version")
	}
	if req.Debuggee.Description == "" {
		s.t.Errorf("missing debuglet description")
	}
	if req.Debuggee.Project == "" {
		s.t.Errorf("missing project id")
	}
	if req.Debuggee.Uniquifier == "" {
		s.t.Errorf("missing uniquifier")
	}
	return &cd.RegisterDebuggeeResponse{
		Debuggee: &cd.Debuggee{Id: testDebuggeeID},
	}, nil
}

func (s *mockService) Update(ctx context.Context, id, breakpointID string, req *cd.UpdateActiveBreakpointRequest) (*cd.UpdateActiveBreakpointResponse, error) {
	if id != testDebuggeeID {
		s.t.Errorf("got debuggee ID %s want %s", id, testDebuggeeID)
	}
	if breakpointID != testBreakpointID {
		s.t.Errorf("got breakpoint ID %s want %s", breakpointID, testBreakpointID)
	}
	if !req.Breakpoint.IsFinalState {
		s.t.Errorf("got IsFinalState = false, want true")
	}
	return nil, nil
}

func (s *mockService) List(ctx context.Context, id, waitToken string) (*cd.ListActiveBreakpointsResponse, error) {
	if id != testDebuggeeID {
		s.t.Errorf("got debuggee ID %s want %s", id, testDebuggeeID)
	}
	if waitToken != expectedWaitToken[s.listCallsSeen] {
		s.t.Errorf("got wait token %s want %s", waitToken, expectedWaitToken[s.listCallsSeen])
	}
	s.listCallsSeen++
	if s.listCallsSeen == 4 {
		return nil, backendError
	}
	if s.listCallsSeen == 5 {
		return nil, abortedError
	}
	resp := &cd.ListActiveBreakpointsResponse{
		Breakpoints:   expectedBreakpoints[s.listCallsSeen-1],
		NextWaitToken: expectedWaitToken[s.listCallsSeen],
	}
	return resp, nil
}

func TestDebugletControllerClientLibrary(t *testing.T) {
	var (
		m    *mockService
		c    *Controller
		list *cd.ListActiveBreakpointsResponse
		err  error
	)
	m = &mockService{t: t}
	newService = func(context.Context, oauth2.TokenSource) (serviceInterface, error) { return m, nil }
	opts := Options{
		ProjectNumber: "5",
		ProjectID:     "p1",
		AppModule:     "mod1",
		AppVersion:    "v1",
	}
	ctx := context.Background()
	if c, err = NewController(ctx, opts); err != nil {
		t.Fatal("Initializing Controller client:", err)
	}
	if err := validateLabels(c, opts); err != nil {
		t.Fatalf("Invalid labels:\n%v", err)
	}
	if list, err = c.List(ctx); err != nil {
		t.Fatal("List:", err)
	}
	if m.registerCallsSeen != 1 {
		t.Errorf("saw %d Register calls, want 1", m.registerCallsSeen)
	}
	if list, err = c.List(ctx); err != nil {
		t.Fatal("List:", err)
	}
	if len(list.Breakpoints) != 1 {
		t.Fatalf("got %d breakpoints, want 1", len(list.Breakpoints))
	}
	if err = c.Update(ctx, list.Breakpoints[0].Id, &cd.Breakpoint{Id: testBreakpointID, IsFinalState: true}); err != nil {
		t.Fatal("Update:", err)
	}
	if list, err = c.List(ctx); err != nil {
		t.Fatal("List:", err)
	}
	if m.registerCallsSeen != 1 {
		t.Errorf("saw %d Register calls, want 1", m.registerCallsSeen)
	}
	// The next List call produces an error that should cause a Register call.
	if list, err = c.List(ctx); err == nil {
		t.Fatal("List should have returned an error")
	}
	if m.registerCallsSeen != 2 {
		t.Errorf("saw %d Register calls, want 2", m.registerCallsSeen)
	}
	// The next List call produces an error that should not cause a Register call.
	if list, err = c.List(ctx); err == nil {
		t.Fatal("List should have returned an error")
	}
	if m.registerCallsSeen != 2 {
		t.Errorf("saw %d Register calls, want 2", m.registerCallsSeen)
	}
	if m.listCallsSeen != 5 {
		t.Errorf("saw %d list calls, want 5", m.listCallsSeen)
	}
}

func validateLabels(c *Controller, o Options) error {
	errMsg := new(bytes.Buffer)
	if m, ok := c.labels["module"]; ok {
		if m != o.AppModule {
			errMsg.WriteString(fmt.Sprintf("label module: want %s, got %s\n", o.AppModule, m))
		}
	} else {
		errMsg.WriteString("Missing \"module\" label\n")
	}
	if v, ok := c.labels["version"]; ok {
		if v != o.AppVersion {
			errMsg.WriteString(fmt.Sprintf("label version: want %s, got %s\n", o.AppVersion, v))
		}
	} else {
		errMsg.WriteString("Missing \"version\" label\n")
	}
	if mv, ok := c.labels["minorversion"]; ok {
		if _, err := strconv.Atoi(mv); err != nil {
			errMsg.WriteString(fmt.Sprintln("label minorversion: not a numeric string:", mv))
		}
	} else {
		errMsg.WriteString("Missing \"minorversion\" label\n")
	}
	if errMsg.Len() != 0 {
		return errors.New(errMsg.String())
	}
	return nil
}

func TestIsAbortedError(t *testing.T) {
	if !isAbortedError(abortedError) {
		t.Errorf("isAborted(%+v): got false, want true", abortedError)
	}
	if isAbortedError(backendError) {
		t.Errorf("isAborted(%+v): got true, want false", backendError)
	}
}
