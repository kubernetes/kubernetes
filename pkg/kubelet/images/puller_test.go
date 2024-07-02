/*
Copyright 2024 The Kubernetes Authors.

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

package images

import (
	"context"
	"sync/atomic"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	kubecontainer "k8s.io/kubernetes/pkg/kubelet/container"
	ctest "k8s.io/kubernetes/pkg/kubelet/container/testing"
)

func TestPullKey(t *testing.T) {
	t.Parallel()
	for _, tc := range []struct {
		name     string
		spec     kubecontainer.ImageSpec
		expected string
	}{
		{
			name: "image",
			spec: kubecontainer.ImageSpec{
				Image: "image",
			},
			expected: "image",
		},
		{
			name: "image with runtime handler",
			spec: kubecontainer.ImageSpec{
				Image:          "image",
				RuntimeHandler: "runtimeHandler",
			},
			expected: "image-runtimeHandler",
		},
		{
			name: "image with runtime handler and annotations",
			spec: kubecontainer.ImageSpec{
				Image:          "image",
				RuntimeHandler: "runtimeHandler",
				Annotations: []kubecontainer.Annotation{
					{Name: "name", Value: "value"},
					{Name: "foo", Value: "bar"},
				},
			},
			expected: "image-runtimeHandler-foo-bar-name-value",
		},
	} {
		expected := tc.expected
		spec := tc.spec

		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			res := pullKey(spec)
			require.Equal(t, expected, res)
		})
	}
}

func TestPullImageParallel(t *testing.T) {
	const (
		parallelPulls = 5
		delay         = 10 * time.Millisecond
	)
	var (
		maxParallelPulls int32 = parallelPulls
		results          atomic.Uint32
	)

	fake := &ctest.FakeRuntime{T: t, DelayImagePulls: parallelPulls * delay}
	p := newParallelImagePuller(fake, &maxParallelPulls)

	for range parallelPulls {
		resCh := make(chan pullResult)
		p.pullImage(context.TODO(), kubecontainer.ImageSpec{Image: "image"}, nil, resCh, nil, v1.PullIfNotPresent)
		go func() {
			<-resCh
			results.Add(1)
		}()
		time.Sleep(delay)
	}
	time.Sleep(delay)

	fake.AssertCallCounts("PullImage", 1)
	require.EqualValues(t, results.Load(), parallelPulls)
}
