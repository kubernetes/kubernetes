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

// ### ATTENTION ###
//
// ReplicationManager is now just a wrapper around ReplicaSetController,
// with a conversion layer that effectively treats ReplicationController
// as if it were an older API version of ReplicaSet.
//
// However, RC and RS still have separate storage and separate instantiations
// of the ReplicaSetController object.

package replication

import (
	"context"

	v1 "k8s.io/api/core/v1"
	coreinformers "k8s.io/client-go/informers/core/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/tools/record"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/controller"
	"k8s.io/kubernetes/pkg/controller/replicaset"
)

const (
	BurstReplicas = replicaset.BurstReplicas
)

// ReplicationManager is responsible for synchronizing ReplicationController objects stored
// in the system with actual running pods.
// It is actually just a wrapper around ReplicaSetController.
type ReplicationManager struct {
	replicaset.ReplicaSetController
}

// NewReplicationManager configures a replication manager with the specified event recorder
func NewReplicationManager(ctx context.Context, podInformer coreinformers.PodInformer, rcInformer coreinformers.ReplicationControllerInformer, kubeClient clientset.Interface, burstReplicas int) *ReplicationManager {
	logger := klog.FromContext(ctx)
	eventBroadcaster := record.NewBroadcaster(record.WithContext(ctx))
	return &ReplicationManager{
		*replicaset.NewBaseController(logger, informerAdapter{rcInformer}, podInformer, clientsetAdapter{kubeClient}, burstReplicas,
			v1.SchemeGroupVersion.WithKind("ReplicationController"),
			"replication_controller",
			"replicationmanager",
			podControlAdapter{controller.RealPodControl{
				KubeClient: kubeClient,
				Recorder:   eventBroadcaster.NewRecorder(scheme.Scheme, v1.EventSource{Component: "replication-controller"}),
			}},
			eventBroadcaster,
		),
	}
}
