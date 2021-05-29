/*
Copyright 2018 The Kubernetes Authors.

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

package ipamperf

import (
	"context"
	"time"

	"k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/klog/v2"
)

const (
	maxCreateRetries = 10
	retryDelay       = 10 * time.Second
)

var (
	baseNodeTemplate = &v1.Node{
		ObjectMeta: metav1.ObjectMeta{
			GenerateName: "sample-node-",
		},
		Status: v1.NodeStatus{
			Capacity: v1.ResourceList{
				v1.ResourcePods:   *resource.NewQuantity(110, resource.DecimalSI),
				v1.ResourceCPU:    resource.MustParse("4"),
				v1.ResourceMemory: resource.MustParse("32Gi"),
			},
			Phase: v1.NodeRunning,
			Conditions: []v1.NodeCondition{
				{Type: v1.NodeReady, Status: v1.ConditionTrue},
			},
		},
	}
)

func deleteNodes(apiURL string, config *Config) {
	klog.Info("Deleting nodes")
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           float32(config.CreateQPS),
		Burst:         config.CreateQPS,
	})
	noGrace := int64(0)
	if err := clientSet.CoreV1().Nodes().DeleteCollection(context.TODO(), metav1.DeleteOptions{GracePeriodSeconds: &noGrace}, metav1.ListOptions{}); err != nil {
		klog.Errorf("Error deleting node: %v", err)
	}
}

func createNodes(apiURL string, config *Config) error {
	clientSet := clientset.NewForConfigOrDie(&restclient.Config{
		Host:          apiURL,
		ContentConfig: restclient.ContentConfig{GroupVersion: &schema.GroupVersion{Group: "", Version: "v1"}},
		QPS:           float32(config.CreateQPS),
		Burst:         config.CreateQPS,
	})
	klog.Infof("Creating %d nodes", config.NumNodes)
	for i := 0; i < config.NumNodes; i++ {
		var err error
		for j := 0; j < maxCreateRetries; j++ {
			if _, err = clientSet.CoreV1().Nodes().Create(context.TODO(), baseNodeTemplate, metav1.CreateOptions{}); err != nil && apierrors.IsServerTimeout(err) {
				klog.Infof("Server timeout creating nodes, retrying after %v", retryDelay)
				time.Sleep(retryDelay)
				continue
			}
			break
		}
		if err != nil {
			klog.Errorf("Error creating nodes: %v", err)
			return err
		}
	}
	klog.Infof("%d nodes created", config.NumNodes)
	return nil
}
