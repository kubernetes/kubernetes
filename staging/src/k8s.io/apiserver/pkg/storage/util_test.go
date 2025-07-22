/*
Copyright 2015 The Kubernetes Authors.

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

package storage_test

import (
	"math/rand"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	"k8s.io/apiserver/pkg/apis/example"
	examplev1 "k8s.io/apiserver/pkg/apis/example/v1"
	example2v1 "k8s.io/apiserver/pkg/apis/example2/v1"
	"k8s.io/apiserver/pkg/storage"
)

var (
	scheme = runtime.NewScheme()
)

func init() {
	metav1.AddToGroupVersion(scheme, metav1.SchemeGroupVersion)
	utilruntime.Must(example.AddToScheme(scheme))
	utilruntime.Must(examplev1.AddToScheme(scheme))
	utilruntime.Must(example2v1.AddToScheme(scheme))
}

func TestHighWaterMark(t *testing.T) {
	var h storage.HighWaterMark

	for i := int64(10); i < 20; i++ {
		if !h.Update(i) {
			t.Errorf("unexpected false for %v", i)
		}
		if h.Update(i - 1) {
			t.Errorf("unexpected true for %v", i-1)
		}
	}

	m := int64(0)
	wg := sync.WaitGroup{}
	for i := 0; i < 300; i++ {
		wg.Add(1)
		v := rand.Int63()
		go func(v int64) {
			defer wg.Done()
			h.Update(v)
		}(v)
		if v > m {
			m = v
		}
	}
	wg.Wait()
	if m != int64(h) {
		t.Errorf("unexpected value, wanted %v, got %v", m, int64(h))
	}
}

func TestHasInitialEventsEndBookmarkAnnotation(t *testing.T) {
	createPod := func(name string) *example.Pod {
		return &example.Pod{ObjectMeta: metav1.ObjectMeta{Name: name}}
	}
	createAnnotatedPod := func(name, value string) *example.Pod {
		p := createPod(name)
		p.Annotations = map[string]string{}
		p.Annotations[metav1.InitialEventsAnnotationKey] = value
		return p
	}
	scenarios := []struct {
		name             string
		object           runtime.Object
		expectAnnotation bool
	}{
		{
			name:             "a standard obj with the initial-events-end annotation set to true",
			object:           createAnnotatedPod("p1", "true"),
			expectAnnotation: true,
		},
		{
			name:   "a standard obj with the initial-events-end annotation set to false",
			object: createAnnotatedPod("p1", "false"),
		},
		{
			name:   "a standard obj without the annotation",
			object: createPod("p1"),
		},
	}

	for _, scenario := range scenarios {
		t.Run(scenario.name, func(t *testing.T) {
			hasAnnotation, err := storage.HasInitialEventsEndBookmarkAnnotation(scenario.object)
			require.NoError(t, err)
			require.Equal(t, scenario.expectAnnotation, hasAnnotation)
		})
	}
}
