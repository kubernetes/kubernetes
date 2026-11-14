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

package storage

import (
	"context"
	"encoding/json"
	"testing"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/kubernetes/fake"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	"k8s.io/kubernetes/pkg/features"
)

func TestRestorePolicyRejectsReplacedCheckpoint(t *testing.T) {
	featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLevelCheckpointRestore, true)
	pod, checkpoint := preparedRestore(t)
	checkpoint.UID = "replacement"
	// Decode the reference admitted for the original object. Older code discards
	// the UID and incorrectly accepts the same-name replacement.
	if err := json.Unmarshal([]byte(`{"name":"cp-1","uid":"original"}`), pod.Spec.RestoreFrom); err != nil {
		t.Fatal(err)
	}
	policy := &restorePolicy{checkpoints: fake.NewClientset(checkpoint).NodeV1alpha1(), authorizer: authorizer.AuthorizerFunc(func(context.Context, authorizer.Attributes) (authorizer.Decision, string, error) {
		return authorizer.DecisionAllow, "", nil
	})}
	if err := policy.validateCreate(restoreContext(), pod); !apierrors.IsForbidden(err) {
		t.Fatalf("replacement checkpoint must be rejected, got %v", err)
	}
}
