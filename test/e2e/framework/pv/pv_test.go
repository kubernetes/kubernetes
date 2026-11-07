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

package pv

import (
	"context"
	"errors"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/client-go/kubernetes/fake"
	k8stesting "k8s.io/client-go/testing"
	"k8s.io/kubernetes/test/e2e/framework"
)

func TestCreatePV(t *testing.T) {
	quotaErr := errors.New("googleapi: Error 403: Quota exceeded for quota group 'CPUS'")
	otherErr := errors.New("admission webhook denied the request")

	tcs := []struct {
		desc      string
		createErr error
		wantErr   error
	}{
		{
			desc: "create succeeds",
		}, {
			// The quota error is retried until PVCreate expires. The caller
			// should see why creation kept failing, not just the timeout.
			desc:      "quota error until timeout reports the API error",
			createErr: quotaErr,
			wantErr:   quotaErr,
		}, {
			desc:      "other API error fails immediately",
			createErr: otherErr,
			wantErr:   otherErr,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.desc, func(t *testing.T) {
			client := fake.NewClientset()
			if tc.createErr != nil {
				client.PrependReactor("create", "persistentvolumes", func(action k8stesting.Action) (bool, runtime.Object, error) {
					return true, nil, tc.createErr
				})
			}
			timeouts := framework.NewTimeoutContext()
			timeouts.PVCreate = 100 * time.Millisecond

			pv := &v1.PersistentVolume{ObjectMeta: metav1.ObjectMeta{Name: "test-pv"}}
			got, err := createPV(context.Background(), client, timeouts, pv)
			if tc.wantErr == nil {
				if err != nil {
					t.Fatalf("unexpected error: %v", err)
				}
				if got == nil || got.Name != pv.Name {
					t.Fatalf("expected PV %q, got %v", pv.Name, got)
				}
				return
			}
			if !errors.Is(err, tc.wantErr) {
				t.Errorf("expected error wrapping %q, got: %v", tc.wantErr, err)
			}
		})
	}
}
