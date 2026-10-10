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

package cache

import (
	"context"
	"fmt"
	"testing"
	"testing/synctest"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	clientfeatures "k8s.io/client-go/features"
	clientfeaturestesting "k8s.io/client-go/features/testing"
)

type initializationTestProvider struct {
	noopInformerMetricsProvider
	observations []float64
	created      int
}

func (p *initializationTestProvider) NewInitializationDurationMetric(InformerNameAndResource) HistogramMetric {
	p.created++
	return p
}
func (p *initializationTestProvider) Observe(seconds float64) {
	p.observations = append(p.observations, seconds)
}

func TestInitializationMetricsProvider(t *testing.T) {
	name, err := NewInformerName(t.Name())
	if err != nil {
		t.Fatal(err)
	}
	defer name.Release()
	gvr := schema.GroupVersionResource{Version: "v1", Resource: "pods"}
	id := name.WithResource(gvr)
	p := &initializationTestProvider{}
	newInitializationMetrics(InformerNameAndResource{}, p).initializationDuration.Observe(1)
	newInitializationMetrics(name.WithResource(gvr), p).initializationDuration.Observe(1)
	if p.created != 0 {
		t.Fatalf("invalid identities created %d metrics", p.created)
	}
	newInitializationMetrics(id, noopInformerMetricsProvider{}).initializationDuration.Observe(1)
	newInitializationMetrics(id, p).initializationDuration.Observe(2)
	if p.created != 1 || len(p.observations) != 1 || p.observations[0] != 2 {
		t.Fatalf("unexpected provider state: %+v", p)
	}
	name.Release()
	newInitializationMetrics(id, p).initializationDuration.Observe(1)
	if p.created != 1 {
		t.Fatal("released identity created a metric")
	}
}

func TestSharedInformerInitializationDuration(t *testing.T) {
	for _, mode := range []string{"list", "empty", "watchlist", "fallback", "cancel"} {
		t.Run(mode, func(t *testing.T) {
			clientfeaturestesting.SetFeatureDuringTest(t, clientfeatures.WatchListClient, mode != "list" && mode != "empty")
			synctest.Test(t, func(t *testing.T) {
				name, err := NewInformerName(t.Name())
				if err != nil {
					t.Fatal(err)
				}
				defer name.Release()
				provider := &initializationTestProvider{}
				ctx, cancel := context.WithCancel(context.Background())
				defer cancel()
				lw := &ListWatch{
					ListWithContextFunc: func(ctx context.Context, _ metav1.ListOptions) (runtime.Object, error) {
						if ctx.Err() != nil {
							return nil, ctx.Err()
						}
						time.Sleep(2 * time.Second)
						if mode == "empty" {
							return &v1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "1"}}, nil
						}
						return &v1.PodList{ListMeta: metav1.ListMeta{ResourceVersion: "1"}, Items: []v1.Pod{{ObjectMeta: metav1.ObjectMeta{Name: "pod", ResourceVersion: "1"}}}}, nil
					},
					WatchFuncWithContext: func(ctx context.Context, opts metav1.ListOptions) (watch.Interface, error) {
						if opts.SendInitialEvents != nil && *opts.SendInitialEvents {
							if mode == "cancel" {
								<-ctx.Done()
								return nil, ctx.Err()
							}
							time.Sleep(3 * time.Second)
							if mode == "fallback" {
								return nil, fmt.Errorf("streaming lists unavailable")
							}
							w := watch.NewRaceFreeFake()
							w.Add(&v1.Pod{ObjectMeta: metav1.ObjectMeta{Name: "pod", ResourceVersion: "1"}})
							w.Action(watch.Bookmark, &v1.Pod{ObjectMeta: metav1.ObjectMeta{ResourceVersion: "1", Annotations: map[string]string{metav1.InitialEventsAnnotationKey: "true"}}})
							return w, nil
						}
						return watch.NewRaceFreeFake(), nil
					},
				}
				informer := NewSharedIndexInformerWithOptions(ToListWatcherWithWatchListSemantics(lw, nil), &v1.Pod{}, SharedIndexInformerOptions{
					Identifier:              name.WithResource(schema.GroupVersionResource{Version: "v1", Resource: "pods"}),
					InformerMetricsProvider: provider,
				})
				// A blocked asynchronous handler must not delay cache initialization.
				if _, err := informer.AddEventHandler(ResourceEventHandlerFuncs{
					AddFunc: func(interface{}) { <-ctx.Done() },
				}); err != nil {
					t.Fatal(err)
				}
				if err := informer.SetTransform(func(obj interface{}) (interface{}, error) {
					time.Sleep(time.Second)
					return obj, nil
				}); err != nil {
					t.Fatal(err)
				}
				done := make(chan struct{})
				go func() { defer close(done); informer.RunWithContext(ctx) }()
				if mode == "cancel" {
					synctest.Wait()
					cancel()
				} else {
					<-informer.HasSyncedChecker().Done()
					if len(provider.observations) != 1 {
						t.Fatalf("got observations %v", provider.observations)
					}
					minimum := 3.0
					if mode == "empty" {
						minimum = 2
					}
					if mode == "watchlist" {
						minimum = 4
					}
					if mode == "fallback" {
						minimum = 6
					}
					if provider.observations[0] < minimum {
						t.Fatalf("duration %v excludes initialization work; want >= %v", provider.observations[0], minimum)
					}
					informer.RunWithContext(ctx)
					if len(provider.observations) != 1 {
						t.Fatal("repeated Run recorded another observation")
					}
					cancel()
				}
				<-done
				if mode == "cancel" && len(provider.observations) != 0 {
					t.Fatalf("canceled sync recorded %v", provider.observations)
				}
			})
		})
	}
}
