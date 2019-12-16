/*
Copyright 2019 The Kubernetes Authors.

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

package webhook

import (
	"context"
	"fmt"
	"testing"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/authentication/user"
	"k8s.io/apiserver/pkg/authorization/authorizer"
)

const invalidIamUser = "NotAllowedbyIAM"

func TestHasUserAssertion(t *testing.T) {
	tests := []struct {
		name     string
		attr     authorizer.AttributesRecord
		expected bool
	}{
		{name: "empty attributes", attr: authorizer.AttributesRecord{}, expected: false},
		{name: "simple user", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Name: "Jane",
		}}, expected: false},
		{name: "wrong key", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Extra: map[string][]string{"wrong-key": {"some-user"}},
		}}, expected: false},
		{name: "valid user assertion key", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Extra: map[string][]string{userAssertionKey: {"some-user"}},
		}}, expected: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision := hasUserAssertion(test.attr)
			if test.expected != decision {
				t.Errorf("expected %v, got %v", test.expected, decision)
			}
		})
	}
}

type DummyCongruentAuthorizer struct{}

func (d *DummyCongruentAuthorizer) Authorize(ctx context.Context, attr authorizer.Attributes) (decision authorizer.Decision, reason string, err error) {
	if hasUserAssertion(attr) && attr.GetUser().GetExtra()[userAssertionKey][0] == invalidIamUser {
		return authorizer.DecisionNoOpinion, "", nil
	}
	return authorizer.DecisionAllow, "", nil
}

func (d *DummyCongruentAuthorizer) RulesFor(user user.Info, namespace string) ([]authorizer.ResourceRuleInfo, []authorizer.NonResourceRuleInfo, bool, error) {
	return nil, nil, false, nil
}

func TestGkeApiserverAuthorize(t *testing.T) {
	tests := []struct {
		name     string
		attr     authorizer.AttributesRecord
		expected authorizer.Decision
	}{
		{name: "empty gke: true, attributes", attr: authorizer.AttributesRecord{}, expected: authorizer.DecisionNoOpinion},
		{name: "simple user", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Name: "Jane",
		}}, expected: authorizer.DecisionNoOpinion},
		{name: "wrong key", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Extra: map[string][]string{"wrong-key": {"some-user"}},
		}}, expected: authorizer.DecisionNoOpinion},
		{name: "valid user assertion key", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Extra: map[string][]string{userAssertionKey: {"some-user"}},
		}}, expected: authorizer.DecisionAllow},
		{name: "invalid IAM user", attr: authorizer.AttributesRecord{User: &user.DefaultInfo{
			Extra: map[string][]string{userAssertionKey: {invalidIamUser}},
		}}, expected: authorizer.DecisionNoOpinion},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			authorizer := GkeApiserverWebhookAuthorizer{&DummyCongruentAuthorizer{}}
			decision, _, err := authorizer.Authorize(context.Background(), test.attr)
			if err != nil {
				t.Errorf("error occured: %v", err)
			}
			if test.expected != decision {
				t.Errorf("expected %v, got %v", test.expected, decision)
			}
		})
	}
}

func TestIsUsingGkeHostedMaster(t *testing.T) {
	tests := []struct {
		name     string
		file     string
		expected bool
	}{
		{name: "gke", file: "gcp_authz.config", expected: true},
		{name: "not gke", file: "other_authz.config", expected: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision, err := isUsingGkeHostedMaster(fmt.Sprintf("testdata/%s", test.file))
			if err != nil {
				t.Errorf("error occured: %v", err)
				return
			}
			if test.expected != decision {
				t.Errorf("expected %v, got %v", test.expected, decision)
			}
		})
	}
}

func TestNewGkeApiserverWebhookAuthorizer(t *testing.T) {
	tests := []struct {
		name           string
		file           string
		expected_gke   bool
		expected_error bool
	}{
		{name: "gke", file: "gcp_authz.config", expected_gke: true},
		{name: "not gke", file: "other_authz.config", expected_gke: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			authorizer, err := NewGkeApiserverWebhookAuthorizer(fmt.Sprintf("testdata/%s", test.file), "v1", time.Second, time.Second, wait.Backoff{}, nil)
			if err != nil {
				t.Errorf("error occured: %v", err)
				return
			}
			_, is_gke := authorizer.(*GkeApiserverWebhookAuthorizer)
			if test.expected_gke != is_gke {
				t.Errorf("expected %v, got %v", test.expected_gke, is_gke)
			}
		})
	}
}

func TestIsGkeHostedMasterUrl(t *testing.T) {
	tests := []struct {
		name     string
		host     string
		expected bool
	}{
		{name: "gke prod", host: "container.googleapis.com", expected: true},
		{name: "gke staging", host: "staging-container.sandbox.googleapis.com", expected: true},
		{name: "gke staging2", host: "staging2-container.sandbox.googleapis.com", expected: true},
		{name: "gke test", host: "test-container.sandbox.googleapis.com", expected: true},
		{name: "gke sandbox", host: "some-gke-sandbox-test-container.sandbox.googleapis.com", expected: true},
		{name: "not gke", host: "anthos.googleapis.com", expected: false},
		{name: "other url", host: "some-other-domain.com", expected: false},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			decision, err := isGkeHostedMasterUrl(fmt.Sprintf("https://%s/", test.host))
			if err != nil {
				t.Errorf("error occured: %v", err)
			}
			if test.expected != decision {
				t.Errorf("expected %v, got %v", test.expected, decision)
			}
		})
	}
}
