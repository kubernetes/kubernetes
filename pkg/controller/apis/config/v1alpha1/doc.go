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

// +k8s:deepcopy-gen=package
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/apis/config
// +k8s:conversion-gen=k8s.io/component-base/config/v1alpha1
// +k8s:conversion-gen=k8s.io/cloud-provider/config/v1alpha1
// +k8s:conversion-gen=k8s.io/cloud-provider/controllers/service/config/v1alpha1
// +k8s:conversion-gen=k8s.io/controller-manager/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/certificates/signer/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/daemon/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/deployment/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/endpoint/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/endpointslice/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/garbagecollector/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/endpointslice/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/endpointslicemirroring/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/job/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/namespace/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/nodeipam/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/nodelifecycle/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/podautoscaler/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/podgc/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/replicaset/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/replication/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/resourcequota/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/serviceaccount/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/statefulset/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/ttlafterfinished/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/volume/attachdetach/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/volume/ephemeral/config/v1alpha1
// +k8s:conversion-gen=k8s.io/kubernetes/pkg/controller/volume/persistentvolume/config/v1alpha1
// +k8s:conversion-gen-external-types=k8s.io/kube-controller-manager/config/v1alpha1
// +k8s:defaulter-gen=TypeMeta
// +k8s:defaulter-gen-input=k8s.io/kube-controller-manager/config/v1alpha1
// +groupName=kubecontrollermanager.config.k8s.io

package v1alpha1
