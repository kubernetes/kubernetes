/*
Copyright The Kubernetes Authors.

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

package admission

import (
	"context"
	"errors"
	"testing"

	"k8s.io/component-helpers/auth/admission/internal/verify"
)

type readinessSpy struct {
	healthErr error
}

func (*readinessSpy) AuthenticateToken(context.Context, string) ([]string, error) {
	return []string{"apps"}, nil
}

func (*readinessSpy) BindAudience(string) error { return nil }
func (s *readinessSpy) HealthCheck() error      { return s.healthErr }

func TestHandlerHealthCheckDelegates(t *testing.T) {
	spy := &readinessSpy{healthErr: errors.New("issuer signing keys unavailable")}
	verifier, err := verify.NewVerifier(spy)
	if err != nil {
		t.Fatalf("construct core verifier: %v", err)
	}
	handler := &Handler{verifier: &Verifier{verifier: verifier}}
	if err := handler.HealthCheck(); !errors.Is(err, spy.healthErr) {
		t.Fatalf("HealthCheck = %v, want issuer readiness error", err)
	}
	spy.healthErr = nil
	if err := handler.HealthCheck(); err != nil {
		t.Fatalf("HealthCheck after issuer recovery = %v, want ready", err)
	}
}
