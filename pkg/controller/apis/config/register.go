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

package config

import (
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	csrsigningconfig "k8s.io/kubernetes/pkg/controller/certificates/signer/config"
	daemonconfig "k8s.io/kubernetes/pkg/controller/daemon/config"
	deploymentconfig "k8s.io/kubernetes/pkg/controller/deployment/config"
	endpointconfig "k8s.io/kubernetes/pkg/controller/endpoint/config"
	garbagecollectorconfig "k8s.io/kubernetes/pkg/controller/garbagecollector/config"
	jobconfig "k8s.io/kubernetes/pkg/controller/job/config"
	namespaceconfig "k8s.io/kubernetes/pkg/controller/namespace/config"
	nodeipamconfig "k8s.io/kubernetes/pkg/controller/nodeipam/config"
	nodelifecycleconfig "k8s.io/kubernetes/pkg/controller/nodelifecycle/config"
	podautosclerconfig "k8s.io/kubernetes/pkg/controller/podautoscaler/config"
	podgcconfig "k8s.io/kubernetes/pkg/controller/podgc/config"
	replicasetconfig "k8s.io/kubernetes/pkg/controller/replicaset/config"
	replicationconfig "k8s.io/kubernetes/pkg/controller/replication/config"
	resourcequotaconfig "k8s.io/kubernetes/pkg/controller/resourcequota/config"
	serviceconfig "k8s.io/kubernetes/pkg/controller/service/config"
	serviceaccountconfig "k8s.io/kubernetes/pkg/controller/serviceaccount/config"
	ttlafterfinishedconfig "k8s.io/kubernetes/pkg/controller/ttlafterfinished/config"
	attachdetachconfig "k8s.io/kubernetes/pkg/controller/volume/attachdetach/config"
	persistentvolumeconfig "k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config"
)

// GroupName is the group name used in this package
const GroupName = "kubecontrollermanager.config.k8s.io"

// SchemeGroupVersion is group version used to register these objects
var SchemeGroupVersion = schema.GroupVersion{Group: GroupName, Version: runtime.APIVersionInternal}

var (
	// SchemeBuilder is the scheme builder with scheme init functions to run for this API package
	SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)
	// AddToScheme is a global function that registers this API group & version to a scheme
	AddToScheme = SchemeBuilder.AddToScheme
)

// addKnownTypes registers known types to the given scheme
func addKnownTypes(scheme *runtime.Scheme) error {
	scheme.AddKnownTypes(SchemeGroupVersion,
		&KubeControllerManagerConfiguration{},
		&attachdetachconfig.AttachDetachControllerConfiguration{},
		&csrsigningconfig.CSRSigningControllerConfiguration{},
		&daemonconfig.DaemonSetControllerConfiguration{},
		&deploymentconfig.DeploymentControllerConfiguration{},
		&endpointconfig.EndpointControllerConfiguration{},
		&garbagecollectorconfig.GarbageCollectorControllerConfiguration{},
		&podautosclerconfig.HPAControllerConfiguration{},
		&jobconfig.JobControllerConfiguration{},
		&namespaceconfig.NamespaceControllerConfiguration{},
		&nodeipamconfig.NodeIPAMControllerConfiguration{},
		&nodelifecycleconfig.NodeLifecycleControllerConfiguration{},
		&persistentvolumeconfig.PersistentVolumeBinderControllerConfiguration{},
		&podgcconfig.PodGCControllerConfiguration{},
		&replicasetconfig.ReplicaSetControllerConfiguration{},
		&replicationconfig.ReplicationControllerConfiguration{},
		&resourcequotaconfig.ResourceQuotaControllerConfiguration{},
		&serviceaccountconfig.SAControllerConfiguration{},
		&serviceconfig.ServiceControllerConfiguration{},
		&ttlafterfinishedconfig.TTLAfterFinishedControllerConfiguration{},
	)
	return nil
}
