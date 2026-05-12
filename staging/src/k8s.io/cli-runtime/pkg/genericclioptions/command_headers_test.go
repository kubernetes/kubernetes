/*
Copyright 2021 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package genericclioptions

import (
	"net/http"
	"testing"

	"github.com/spf13/cobra"
)

var kubectlCmd = &cobra.Command{Use: "kubectl"}
var applyCmd = &cobra.Command{Use: "apply"}
var createCmd = &cobra.Command{Use: "create"}
var secretCmd = &cobra.Command{Use: "secret"}
var genericCmd = &cobra.Command{Use: "generic"}
var authCmd = &cobra.Command{Use: "auth"}
var reconcileCmd = &cobra.Command{Use: "reconcile"}

func TestParseCommandHeaders(t *testing.T) {
	tests := map[string]struct {
		// Ordering is important; each subsequent command is added as a subcommand
		// of the previous command.
		commands []*cobra.Command
		// Headers which should be present; but other headers may exist
		expectedHeaders map[string]string
	}{
		"Single kubectl command example": {
			commands: []*cobra.Command{kubectlCmd},
			expectedHeaders: map[string]string{
				kubectlCommandHeader: "kubectl",
			},
		},
		"Simple kubectl apply example": {
			commands: []*cobra.Command{kubectlCmd, applyCmd},
			expectedHeaders: map[string]string{
				kubectlCommandHeader: "kubectl apply",
			},
		},
		"Kubectl auth reconcile example": {
			commands: []*cobra.Command{kubectlCmd, authCmd, reconcileCmd},
			expectedHeaders: map[string]string{
				kubectlCommandHeader: "kubectl auth reconcile",
			},
		},
		"Long kubectl create secret generic example": {
			commands: []*cobra.Command{kubectlCmd, createCmd, secretCmd, genericCmd},
			expectedHeaders: map[string]string{
				kubectlCommandHeader: "kubectl create secret generic",
			},
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			rootCmd := buildCommandChain(tc.commands)
			ch := &CommandHeaderRoundTripper{}
			ch.ParseCommandHeaders(rootCmd, []string{})
			// Unique session ID header should always be present.
			if _, found := ch.Headers[kubectlSessionHeader]; !found {
				t.Errorf("expected kubectl session header (%s) is missing", kubectlSessionHeader)
			}
			// All expected headers must be present; but there may be extras.
			for key, expectedValue := range tc.expectedHeaders {
				actualValue, found := ch.Headers[key]
				if found {
					if expectedValue != actualValue {
						t.Errorf("expected header value (%s), got (%s)", expectedValue, actualValue)
					}
				} else {
					t.Errorf("expected header (%s) not found", key)
				}
			}
		})
	}
}

// Builds a hierarchy of commands in order from the passed slice of commands,
// by adding each subsequent command as a child of the previous command,
// returning the last leaf command.
func buildCommandChain(commands []*cobra.Command) *cobra.Command {
	var currCmd *cobra.Command
	if len(commands) > 0 {
		currCmd = commands[0]
	}
	for i := 1; i < len(commands); i++ {
		cmd := commands[i]
		currCmd.AddCommand(cmd)
		currCmd = cmd
	}
	return currCmd
}

// Tests that the CancelRequest function is propogated to the wrapped Delegate
// RoundTripper; but only if the Delegate implements the CancelRequest function.
func TestCancelRequest(t *testing.T) {
	tests := map[string]struct {
		delegate  http.RoundTripper
		cancelled bool
	}{
		"CancelRequest propagated to delegate": {
			delegate:  &cancellableRoundTripper{},
			cancelled: true,
		},
		"CancelRequest not propagated to delegate": {
			delegate:  &nonCancellableRoundTripper{},
			cancelled: false,
		},
	}

	for name, tc := range tests {
		t.Run(name, func(t *testing.T) {
			rt := &CommandHeaderRoundTripper{
				Delegate: tc.delegate,
			}
			req := http.Request{}
			rt.CancelRequest(&req)
			if tc.cancelled != req.Close {
				t.Errorf("expected RoundTripper cancel (%v), got (%v)", tc.cancelled, req.Close)
			}
		})
	}
}

// Test RoundTripper with CancelRequest function.
type cancellableRoundTripper struct{}

func (rtc *cancellableRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, nil
}

func (rtc *cancellableRoundTripper) CancelRequest(req *http.Request) {
	req.Close = true
}

// Test RoundTripper without CancelRequest function.
type nonCancellableRoundTripper struct{}

func (rtc *nonCancellableRoundTripper) RoundTrip(req *http.Request) (*http.Response, error) {
	return nil, nil
}
