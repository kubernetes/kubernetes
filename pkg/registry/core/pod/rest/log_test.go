/*
Copyright 2014 The Kubernetes Authors.

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

package rest

import (
	"fmt"
	"strings"
	"testing"

	"k8s.io/apimachinery/pkg/api/errors"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistry "k8s.io/apiserver/pkg/registry/generic/registry"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	"k8s.io/utils/ptr"
)

func TestPodLogValidates(t *testing.T) {
	config, server := registrytest.NewEtcdStorage(t, "")
	defer server.Terminate(t)
	s, destroyFunc, err := generic.NewRawStorage(config, nil, nil, "")
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer destroyFunc()
	store := &genericregistry.Store{
		Storage: genericregistry.DryRunnableStorage{Storage: s},
	}
	logRest := &LogREST{Store: store, KubeletConn: nil}

	// This test will panic if you don't have a validation failure.
	negativeOne := int64(-1)
	testCases := []struct {
		name               string
		podOptions         api.PodLogOptions
		podQueryLogOptions bool
		invalidStreamMatch string
	}{
		{
			name: "SinceSeconds",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
			},
		},
		{
			name: "TailLines",
			podOptions: api.PodLogOptions{
				TailLines: &negativeOne,
			},
		},
		{
			name: "StreamWithGateOff",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
			},
			podQueryLogOptions: false,
		},
		{
			name: "StreamWithGateOnDefault",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
			},
			podQueryLogOptions: true,
		},
		{
			name: "StreamWithGateOnAll",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
				Stream:       ptr.To(api.LogStreamAll),
			},
			podQueryLogOptions: true,
		},
		{
			name: "StreamWithGateOnStdErr",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
				Stream:       ptr.To(api.LogStreamStderr),
			},
			podQueryLogOptions: true,
		},
		{
			name: "StreamWithGateOnStdOut",
			podOptions: api.PodLogOptions{
				SinceSeconds: &negativeOne,
				Stream:       ptr.To(api.LogStreamStdout),
			},
			podQueryLogOptions: true,
		},
		{
			name: "StreamWithGateOnAndBadValue",
			podOptions: api.PodLogOptions{
				Stream: ptr.To("nostream"),
			},
			podQueryLogOptions: true,
			invalidStreamMatch: "nostream",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.PodLogsQuerySplitStreams, tc.podQueryLogOptions)
			_, err := logRest.Get(genericapirequest.NewDefaultContext(), "test", &tc.podOptions)
			if !errors.IsInvalid(err) {
				t.Fatalf("Unexpected error: %v", err)
			}
			if tc.invalidStreamMatch != "" {
				if !strings.Contains(err.Error(), "nostream") {
					t.Error(fmt.Printf("Expected %s got %s", err.Error(), "nostream"))
				}
			}
		})
	}
}
