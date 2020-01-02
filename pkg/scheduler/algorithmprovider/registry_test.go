/*
Copyright 2017 The Kubernetes Authors.

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

package algorithmprovider

import (
	"testing"
	"time"

	"github.com/google/go-cmp/cmp"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
	"k8s.io/kubernetes/pkg/scheduler/framework/plugins"
	framework "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1"
	nodeinfosnapshot "k8s.io/kubernetes/pkg/scheduler/nodeinfo/snapshot"
	"k8s.io/kubernetes/pkg/scheduler/volumebinder"
)

func TestCompatibility(t *testing.T) {
	testcases := []struct {
		name        string
		provider    string
		wantPlugins map[string][]config.Plugin
	}{
		{
			name:     "DefaultProvider",
			provider: config.SchedulerDefaultProviderName,
		},
		{
			name:     "ClusterAutoscalerProvider",
			provider: ClusterAutoscalerProvider,
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			client := fake.NewSimpleClientset()
			informerFactory := informers.NewSharedInformerFactory(client, 0)

			sched, err := scheduler.New(
				client,
				informerFactory,
				informerFactory.Core().V1().Pods(),
				nil,
				make(chan struct{}),
				scheduler.WithAlgorithmSource(config.SchedulerAlgorithmSource{
					Provider: &tc.provider,
				}))
			if err != nil {
				t.Fatalf("Error constructing: %v", err)
			}
			gotPlugins := sched.Framework.ListPlugins()

			volumeBinder := volumebinder.NewVolumeBinder(
				client,
				informerFactory.Core().V1().Nodes(),
				informerFactory.Storage().V1().CSINodes(),
				informerFactory.Core().V1().PersistentVolumeClaims(),
				informerFactory.Core().V1().PersistentVolumes(),
				informerFactory.Storage().V1().StorageClasses(),
				time.Second,
			)
			providerRegistry := NewRegistry(1)
			config := providerRegistry[tc.provider]
			fwk, err := framework.NewFramework(
				plugins.NewInTreeRegistry(&plugins.RegistryArgs{
					VolumeBinder: volumeBinder,
				}),
				config.FrameworkPlugins,
				config.FrameworkPluginConfig,
				framework.WithClientSet(client),
				framework.WithInformerFactory(informerFactory),
				framework.WithSnapshotSharedLister(nodeinfosnapshot.NewEmptySnapshot()),
			)
			if err != nil {
				t.Fatalf("error initializing the scheduling framework: %v", err)
			}
			wantPlugins := fwk.ListPlugins()

			if diff := cmp.Diff(wantPlugins, gotPlugins); diff != "" {
				t.Errorf("unexpected plugins diff (-want, +got): %s", diff)
			}
		})
	}
}
