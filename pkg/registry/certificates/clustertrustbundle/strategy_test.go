/*
Copyright 2022 The Kubernetes Authors.

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

package clustertrustbundle

import (
	"context"
	"testing"

	"k8s.io/kubernetes/pkg/apis/certificates"
)

func TestWarningsOnCreate(t *testing.T) {
	if warnings := Strategy.WarningsOnCreate(context.Background(), &certificates.ClusterTrustBundle{}); warnings != nil {
		t.Errorf("Got %v, want nil", warnings)
	}
}

func TestAllowCreateOnUpdate(t *testing.T) {
	if Strategy.AllowCreateOnUpdate() != false {
		t.Errorf("Got true, want false")
	}
}

func TestWarningsOnUpdate(t *testing.T) {
	if warnings := Strategy.WarningsOnUpdate(context.Background(), &certificates.ClusterTrustBundle{}, &certificates.ClusterTrustBundle{}); warnings != nil {
		t.Errorf("Got %v, want nil", warnings)
	}
}

func TestAllowUnconditionalUpdate(t *testing.T) {
	if Strategy.AllowUnconditionalUpdate() != false {
		t.Errorf("Got true, want false")
	}
}
