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

package benchmark

import (
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/pkg/scheduler/factory"
	"k8s.io/kubernetes/test/integration/util"
)

// mustSetupScheduler starts the following components:
// - k8s api server (a.k.a. master)
// - scheduler
// It returns scheduler config factory and destroyFunc which should be used to
// remove resources after finished.
// Notes on rate limiter:
//   - client rate limit is set to 5000.
func mustSetupScheduler() (*factory.Config, util.ShutdownFunc, clientset.Interface) {
	apiURL, apiShutdown := util.StartApiserver()
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           5000.0,
		Burst:         5000,
	})
	schedulerConfig, schedulerShutdown := util.StartScheduler(clientSet)

	shutdownFunc := func() {
		schedulerShutdown()
		apiShutdown()
	}
	return schedulerConfig, shutdownFunc, clientSet
}

func getScheduledPods(clientset clientset.Interface) ([]*v1.Pod, error) {
	podList, err := clientset.CoreV1().Pods("").List(metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	allPods := podList.Items
	scheduled := make([]*v1.Pod, 0, len(allPods))
	for i := range allPods {
		pod := allPods[i]
		if len(pod.Spec.NodeName) > 0 {
			scheduled = append(scheduled, &pod)
		}
	}
	return scheduled, nil
}
