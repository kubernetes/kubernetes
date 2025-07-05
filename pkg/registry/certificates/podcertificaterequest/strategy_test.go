/*
Copyright 2025 The Kubernetes Authors.

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

package podcertificaterequest

import (
	"context"
	"testing"

	"github.com/google/go-cmp/cmp"
	"k8s.io/kubernetes/pkg/apis/certificates"
)

func TestWarningsOnCreate(t *testing.T) {
	strategy := NewStrategy()

	var wantWarnings []string
	gotWarnings := strategy.WarningsOnCreate(context.Background(), &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowCreateOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowCreateOnUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	strategy := NewStrategy()
	var wantWarnings []string
	gotWarnings := strategy.WarningsOnUpdate(context.Background(), &certificates.PodCertificateRequest{}, &certificates.PodCertificateRequest{})
	if diff := cmp.Diff(gotWarnings, wantWarnings); diff != "" {
		t.Errorf("Got wrong warnings; diff (-got +want):\n%s", diff)
	}
}

func TestAllowUnconditionalUpdate(t *testing.T) {
	strategy := NewStrategy()
	if strategy.AllowUnconditionalUpdate() != false {
		t.Errorf("Got true, want false")
	}
}
