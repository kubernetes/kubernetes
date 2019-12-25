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

package defaults

import (
	"k8s.io/kubernetes/pkg/scheduler"
	"k8s.io/kubernetes/pkg/scheduler/algorithm/priorities"
)

func init() {
	// Register functions that extract metadata used by priorities computations.
	scheduler.RegisterPriorityMetadataProducerFactory(
		func(args scheduler.AlgorithmFactoryArgs) priorities.MetadataProducer {
			serviceLister := args.InformerFactory.Core().V1().Services().Lister()
			controllerLister := args.InformerFactory.Core().V1().ReplicationControllers().Lister()
			replicaSetLister := args.InformerFactory.Apps().V1().ReplicaSets().Lister()
			statefulSetLister := args.InformerFactory.Apps().V1().StatefulSets().Lister()
			return priorities.NewMetadataFactory(serviceLister, controllerLister, replicaSetLister, statefulSetLister, args.HardPodAffinitySymmetricWeight)
		})
}
