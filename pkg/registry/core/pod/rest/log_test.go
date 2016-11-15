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
	"testing"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/registry/generic"
	"k8s.io/kubernetes/pkg/registry/generic/registry"
	"k8s.io/kubernetes/pkg/registry/registrytest"
)

func TestPodLogValidates(t *testing.T) {
	config, server := registrytest.NewEtcdStorage(t, "")
	defer server.Terminate(t)
	s, destroyFunc := generic.NewRawStorage(config)
	defer destroyFunc()
	store := &registry.Store{
		Storage: s,
	}
	logRest := &LogREST{Store: store, KubeletConn: nil}

	negativeOne := int64(-1)
	testCases := []*api.PodLogOptions{
		{SinceSeconds: &negativeOne},
		{TailLines: &negativeOne},
	}

	for _, tc := range testCases {
		_, err := logRest.Get(api.NewDefaultContext(), "test", tc)
		if !errors.IsInvalid(err) {
			t.Fatalf("unexpected error: %v", err)
		}
	}
}
