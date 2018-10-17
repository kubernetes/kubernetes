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

package noderestriction

import (
	"fmt"
	"reflect"
	"strings"
	"testing"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/authentication/user"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	corev1lister "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	csiv1alpha1 "k8s.io/csi-api/pkg/apis/csi/v1alpha1"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	"k8s.io/kubernetes/pkg/apis/coordination"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
	"k8s.io/utils/pointer"
)

var (
	trEnabledFeature           = utilfeature.NewFeatureGate()
	trDisabledFeature          = utilfeature.NewFeatureGate()
	leaseEnabledFeature        = utilfeature.NewFeatureGate()
	leaseDisabledFeature       = utilfeature.NewFeatureGate()
	csiNodeInfoEnabledFeature  = utilfeature.NewFeatureGate()
	csiNodeInfoDisabledFeature = utilfeature.NewFeatureGate()
)

func init() {
	if err := trEnabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.TokenRequest: {Default: true}}); err != nil {
		panic(err)
	}
	if err := trDisabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.TokenRequest: {Default: false}}); err != nil {
		panic(err)
	}
	if err := leaseEnabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.NodeLease: {Default: true}}); err != nil {
		panic(err)
	}
	if err := leaseDisabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.NodeLease: {Default: false}}); err != nil {
		panic(err)
	}
	if err := csiNodeInfoEnabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.KubeletPluginsWatcher: {Default: true}}); err != nil {
		panic(err)
	}
	if err := csiNodeInfoEnabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.CSINodeInfo: {Default: true}}); err != nil {
		panic(err)
	}
	if err := csiNodeInfoDisabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.KubeletPluginsWatcher: {Default: false}}); err != nil {
		panic(err)
	}
	if err := csiNodeInfoDisabledFeature.Add(map[utilfeature.Feature]utilfeature.FeatureSpec{features.CSINodeInfo: {Default: false}}); err != nil {
		panic(err)
	}
}

func makeTestPod(namespace, name, node string, mirror bool) (*api.Pod, *corev1.Pod) {
	corePod := &api.Pod{}
	corePod.Namespace = namespace
	corePod.UID = types.UID("pod-uid")
	corePod.Name = name
	corePod.Spec.NodeName = node
	v1Pod := &corev1.Pod{}
	v1Pod.Namespace = namespace
	v1Pod.UID = types.UID("pod-uid")
	v1Pod.Name = name
	v1Pod.Spec.NodeName = node
	if mirror {
		corePod.Annotations = map[string]string{api.MirrorPodAnnotationKey: "true"}
		v1Pod.Annotations = map[string]string{api.MirrorPodAnnotationKey: "true"}
	}
	return corePod, v1Pod
}

func makeTestPodEviction(name string) *policy.Eviction {
	eviction := &policy.Eviction{}
	eviction.Name = name
	eviction.Namespace = "ns"
	return eviction
}

func makeTokenRequest(podname string, poduid types.UID) *authenticationapi.TokenRequest {
	tr := &authenticationapi.TokenRequest{
		Spec: authenticationapi.TokenRequestSpec{
			Audiences: []string{"foo"},
		},
	}
	if podname != "" {
		tr.Spec.BoundObjectRef = &authenticationapi.BoundObjectReference{
			Kind:       "Pod",
			APIVersion: "v1",
			Name:       podname,
			UID:        poduid,
		}
	}
	return tr
}

func setAllLabels(node *api.Node, value string) *api.Node {
	node = setAllowedCreateLabels(node, value)
	node = setAllowedUpdateLabels(node, value)
	node = setForbiddenCreateLabels(node, value)
	node = setForbiddenUpdateLabels(node, value)
	return node
}

func setAllowedCreateLabels(node *api.Node, value string) *api.Node {
	node = setAllowedUpdateLabels(node, value)
	// also allow other kubernetes labels on create until 1.17 (TODO: remove this in 1.17)
	node.Labels["other.kubernetes.io/foo"] = value
	node.Labels["other.k8s.io/foo"] = value
	return node
}

func setAllowedUpdateLabels(node *api.Node, value string) *api.Node {
	node = node.DeepCopy()
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	if value == "" {
		value = "value"
	}
	// non-kube labels
	node.Labels["foo"] = value
	node.Labels["example.com/foo"] = value

	// kubelet labels
	node.Labels["kubernetes.io/hostname"] = value
	node.Labels["failure-domain.beta.kubernetes.io/zone"] = value
	node.Labels["failure-domain.beta.kubernetes.io/region"] = value
	node.Labels["beta.kubernetes.io/instance-type"] = value
	node.Labels["beta.kubernetes.io/os"] = value
	node.Labels["beta.kubernetes.io/arch"] = value
	node.Labels["failure-domain.kubernetes.io/zone"] = value
	node.Labels["failure-domain.kubernetes.io/region"] = value
	node.Labels["kubernetes.io/instance-type"] = value
	node.Labels["kubernetes.io/os"] = value
	node.Labels["kubernetes.io/arch"] = value

	// kubelet label prefixes
	node.Labels["kubelet.kubernetes.io/foo"] = value
	node.Labels["foo.kubelet.kubernetes.io/foo"] = value
	node.Labels["node.kubernetes.io/foo"] = value
	node.Labels["foo.node.kubernetes.io/foo"] = value

	// test all explicitly allowed labels and prefixes
	for _, key := range kubeletapis.KubeletLabels() {
		node.Labels[key] = value
	}
	for _, namespace := range kubeletapis.KubeletLabelNamespaces() {
		node.Labels[namespace+"/foo"] = value
		node.Labels["foo."+namespace+"/foo"] = value
	}

	return node
}

func setForbiddenCreateLabels(node *api.Node, value string) *api.Node {
	node = node.DeepCopy()
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	if value == "" {
		value = "value"
	}
	// node restriction labels are forbidden
	node.Labels["node-restriction.kubernetes.io/foo"] = value
	node.Labels["foo.node-restriction.kubernetes.io/foo"] = value
	// TODO: in 1.17, forbid arbitrary kubernetes labels on create
	// node.Labels["other.kubernetes.io/foo"] = value
	// node.Labels["other.k8s.io/foo"] = value
	return node
}

func setForbiddenUpdateLabels(node *api.Node, value string) *api.Node {
	node = node.DeepCopy()
	if node.Labels == nil {
		node.Labels = map[string]string{}
	}
	if value == "" {
		value = "value"
	}
	// node restriction labels are forbidden
	node.Labels["node-restriction.kubernetes.io/foo"] = value
	node.Labels["foo.node-restriction.kubernetes.io/foo"] = value
	// arbitrary kubernetes labels are forbidden on update
	node.Labels["other.kubernetes.io/foo"] = value
	node.Labels["other.k8s.io/foo"] = value
	return node
}

func Test_nodePlugin_Admit(t *testing.T) {
	var (
		mynode = &user.DefaultInfo{Name: "system:node:mynode", Groups: []string{"system:nodes"}}
		bob    = &user.DefaultInfo{Name: "bob"}

		mynodeObjMeta    = metav1.ObjectMeta{Name: "mynode"}
		mynodeObj        = &api.Node{ObjectMeta: mynodeObjMeta}
		mynodeObjConfigA = &api.Node{ObjectMeta: mynodeObjMeta, Spec: api.NodeSpec{ConfigSource: &api.NodeConfigSource{
			ConfigMap: &api.ConfigMapNodeConfigSource{
				Name:             "foo",
				Namespace:        "bar",
				UID:              "fooUID",
				KubeletConfigKey: "kubelet",
			}}}}
		mynodeObjConfigB = &api.Node{ObjectMeta: mynodeObjMeta, Spec: api.NodeSpec{ConfigSource: &api.NodeConfigSource{
			ConfigMap: &api.ConfigMapNodeConfigSource{
				Name:             "qux",
				Namespace:        "bar",
				UID:              "quxUID",
				KubeletConfigKey: "kubelet",
			}}}}

		mynodeObjTaintA = &api.Node{ObjectMeta: mynodeObjMeta, Spec: api.NodeSpec{Taints: []api.Taint{{Key: "mykey", Value: "A"}}}}
		mynodeObjTaintB = &api.Node{ObjectMeta: mynodeObjMeta, Spec: api.NodeSpec{Taints: []api.Taint{{Key: "mykey", Value: "B"}}}}
		othernodeObj    = &api.Node{ObjectMeta: metav1.ObjectMeta{Name: "othernode"}}

		coremymirrorpod, v1mymirrorpod           = makeTestPod("ns", "mymirrorpod", "mynode", true)
		coreothermirrorpod, v1othermirrorpod     = makeTestPod("ns", "othermirrorpod", "othernode", true)
		coreunboundmirrorpod, v1unboundmirrorpod = makeTestPod("ns", "unboundmirrorpod", "", true)
		coremypod, v1mypod                       = makeTestPod("ns", "mypod", "mynode", false)
		coreotherpod, v1otherpod                 = makeTestPod("ns", "otherpod", "othernode", false)
		coreunboundpod, v1unboundpod             = makeTestPod("ns", "unboundpod", "", false)
		coreunnamedpod, _                        = makeTestPod("ns", "", "mynode", false)

		mymirrorpodEviction      = makeTestPodEviction("mymirrorpod")
		othermirrorpodEviction   = makeTestPodEviction("othermirrorpod")
		unboundmirrorpodEviction = makeTestPodEviction("unboundmirrorpod")
		mypodEviction            = makeTestPodEviction("mypod")
		otherpodEviction         = makeTestPodEviction("otherpod")
		unboundpodEviction       = makeTestPodEviction("unboundpod")
		unnamedEviction          = makeTestPodEviction("")

		configmapResource = api.Resource("configmap").WithVersion("v1")
		configmapKind     = api.Kind("ConfigMap").WithVersion("v1")

		podResource  = api.Resource("pods").WithVersion("v1")
		podKind      = api.Kind("Pod").WithVersion("v1")
		evictionKind = policy.Kind("Eviction").WithVersion("v1beta1")

		nodeResource = api.Resource("nodes").WithVersion("v1")
		nodeKind     = api.Kind("Node").WithVersion("v1")

		svcacctResource  = api.Resource("serviceaccounts").WithVersion("v1")
		tokenrequestKind = api.Kind("TokenRequest").WithVersion("v1")

		leaseResource = coordination.Resource("leases").WithVersion("v1beta1")
		leaseKind     = coordination.Kind("Lease").WithVersion("v1beta1")
		lease         = &coordination.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mynode",
				Namespace: api.NamespaceNodeLease,
			},
			Spec: coordination.LeaseSpec{
				HolderIdentity:       pointer.StringPtr("mynode"),
				LeaseDurationSeconds: pointer.Int32Ptr(40),
				RenewTime:            &metav1.MicroTime{Time: time.Now()},
			},
		}
		leaseWrongNS = &coordination.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "mynode",
				Namespace: "foo",
			},
			Spec: coordination.LeaseSpec{
				HolderIdentity:       pointer.StringPtr("mynode"),
				LeaseDurationSeconds: pointer.Int32Ptr(40),
				RenewTime:            &metav1.MicroTime{Time: time.Now()},
			},
		}
		leaseWrongName = &coordination.Lease{
			ObjectMeta: metav1.ObjectMeta{
				Name:      "foo",
				Namespace: api.NamespaceNodeLease,
			},
			Spec: coordination.LeaseSpec{
				HolderIdentity:       pointer.StringPtr("mynode"),
				LeaseDurationSeconds: pointer.Int32Ptr(40),
				RenewTime:            &metav1.MicroTime{Time: time.Now()},
			},
		}

		csiNodeInfoResource = csiv1alpha1.Resource("csinodeinfos").WithVersion("v1alpha1")
		csiNodeInfoKind     = schema.GroupVersionKind{Group: "csi.storage.k8s.io", Version: "v1alpha1", Kind: "CSINodeInfo"}
		nodeInfo            = &csiv1alpha1.CSINodeInfo{
			ObjectMeta: metav1.ObjectMeta{
				Name: "mynode",
			},
			Spec: csiv1alpha1.CSINodeInfoSpec{
				Drivers: []csiv1alpha1.CSIDriverInfoSpec{
					{
						Name:         "com.example.csi/mydriver",
						NodeID:       "com.example.csi/mynode",
						TopologyKeys: []string{"com.example.csi/zone"},
					},
				},
			},
			Status: csiv1alpha1.CSINodeInfoStatus{
				Drivers: []csiv1alpha1.CSIDriverInfoStatus{
					{
						Name:                  "com.example.csi/mydriver",
						Available:             true,
						VolumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
					},
				},
			},
		}
		nodeInfoWrongName = &csiv1alpha1.CSINodeInfo{
			ObjectMeta: metav1.ObjectMeta{
				Name: "foo",
			},
			Spec: csiv1alpha1.CSINodeInfoSpec{
				Drivers: []csiv1alpha1.CSIDriverInfoSpec{
					{
						Name:         "com.example.csi/mydriver",
						NodeID:       "com.example.csi/foo",
						TopologyKeys: []string{"com.example.csi/zone"},
					},
				},
			},
			Status: csiv1alpha1.CSINodeInfoStatus{
				Drivers: []csiv1alpha1.CSIDriverInfoStatus{
					{
						Name:                  "com.example.csi/mydriver",
						Available:             true,
						VolumePluginMechanism: csiv1alpha1.VolumePluginMechanismInTree,
					},
				},
			},
		}

		noExistingPodsIndex = cache.NewIndexer(cache.MetaNamespaceKeyFunc, nil)
		noExistingPods      = corev1lister.NewPodLister(noExistingPodsIndex)

		existingPodsIndex = cache.NewIndexer(cache.MetaNamespaceKeyFunc, nil)
		existingPods      = corev1lister.NewPodLister(existingPodsIndex)
	)

	existingPodsIndex.Add(v1mymirrorpod)
	existingPodsIndex.Add(v1othermirrorpod)
	existingPodsIndex.Add(v1unboundmirrorpod)
	existingPodsIndex.Add(v1mypod)
	existingPodsIndex.Add(v1otherpod)
	existingPodsIndex.Add(v1unboundpod)

	sapod, _ := makeTestPod("ns", "mysapod", "mynode", true)
	sapod.Spec.ServiceAccountName = "foo"

	secretpod, _ := makeTestPod("ns", "mysecretpod", "mynode", true)
	secretpod.Spec.Volumes = []api.Volume{{VolumeSource: api.VolumeSource{Secret: &api.SecretVolumeSource{SecretName: "foo"}}}}

	configmappod, _ := makeTestPod("ns", "myconfigmappod", "mynode", true)
	configmappod.Spec.Volumes = []api.Volume{{VolumeSource: api.VolumeSource{ConfigMap: &api.ConfigMapVolumeSource{LocalObjectReference: api.LocalObjectReference{Name: "foo"}}}}}

	pvcpod, _ := makeTestPod("ns", "mypvcpod", "mynode", true)
	pvcpod.Spec.Volumes = []api.Volume{{VolumeSource: api.VolumeSource{PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{ClaimName: "foo"}}}}

	tests := []struct {
		name       string
		podsGetter corev1lister.PodLister
		attributes admission.Attributes
		features   utilfeature.FeatureGate
		err        string
	}{
		// Mirror pods bound to us
		{
			name:       "allow creating a mirror pod bound to self",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coremymirrorpod, nil, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "forbid update of mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coremymirrorpod, coremymirrorpod, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow delete of mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "",
		},
		{
			name:       "forbid create of mirror pod status bound to self",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coremymirrorpod, nil, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow update of mirror pod status bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coremymirrorpod, coremymirrorpod, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "forbid delete of mirror pod status bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow create of eviction for mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mymirrorpodEviction, nil, evictionKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "forbid update of eviction for mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mymirrorpodEviction, nil, evictionKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mymirrorpodEviction, nil, evictionKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow create of unnamed eviction for mirror pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coremymirrorpod.Namespace, coremymirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "",
		},

		// Mirror pods bound to another node
		{
			name:       "forbid creating a mirror pod bound to another",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreothermirrorpod, nil, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of mirror pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreothermirrorpod, coreothermirrorpod, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of mirror pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid create of mirror pod status bound to another",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreothermirrorpod, nil, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid update of mirror pod status bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreothermirrorpod, coreothermirrorpod, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid delete of mirror pod status bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of eviction for mirror pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(othermirrorpodEviction, nil, evictionKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of eviction for mirror pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(othermirrorpodEviction, nil, evictionKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for mirror pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(othermirrorpodEviction, nil, evictionKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for mirror pod to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coreothermirrorpod.Namespace, coreothermirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},

		// Mirror pods not bound to any node
		{
			name:       "forbid creating a mirror pod unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundmirrorpod, nil, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundmirrorpod, coreunboundmirrorpod, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid create of mirror pod status unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundmirrorpod, nil, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid update of mirror pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundmirrorpod, coreunboundmirrorpod, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid delete of mirror pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of eviction for mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundmirrorpodEviction, nil, evictionKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of eviction for mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundmirrorpodEviction, nil, evictionKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundmirrorpodEviction, nil, evictionKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for mirror pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coreunboundmirrorpod.Namespace, coreunboundmirrorpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},

		// Normal pods bound to us
		{
			name:       "forbid creating a normal pod bound to self",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coremypod, nil, podKind, coremypod.Namespace, coremypod.Name, podResource, "", admission.Create, false, mynode),
			err:        "can only create mirror pods",
		},
		{
			name:       "forbid update of normal pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coremypod, coremypod, podKind, coremypod.Namespace, coremypod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow delete of normal pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coremypod.Namespace, coremypod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "",
		},
		{
			name:       "forbid create of normal pod status bound to self",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coremypod, nil, podKind, coremypod.Namespace, coremypod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow update of normal pod status bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coremypod, coremypod, podKind, coremypod.Namespace, coremypod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "forbid delete of normal pod status bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coremypod.Namespace, coremypod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid update of eviction for normal pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for normal pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "allow create of unnamed eviction for normal pod bound to self",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "",
		},

		// Normal pods bound to another
		{
			name:       "forbid creating a normal pod bound to another",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreotherpod, nil, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "can only create mirror pods",
		},
		{
			name:       "forbid update of normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreotherpod, coreotherpod, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid create of normal pod status bound to another",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreotherpod, nil, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid update of normal pod status bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreotherpod, coreotherpod, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid delete of normal pod status bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of eviction for normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(otherpodEviction, nil, evictionKind, otherpodEviction.Namespace, otherpodEviction.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of eviction for normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(otherpodEviction, nil, evictionKind, otherpodEviction.Namespace, otherpodEviction.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(otherpodEviction, nil, evictionKind, otherpodEviction.Namespace, otherpodEviction.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for normal pod bound to another",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coreotherpod.Namespace, coreotherpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},

		// Normal pods not bound to any node
		{
			name:       "forbid creating a normal pod unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "can only create mirror pods",
		},
		{
			name:       "forbid update of normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, coreunboundpod, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid create of normal pod status unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Create, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid update of normal pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, coreunboundpod, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Update, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid delete of normal pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of eviction for normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundpodEviction, nil, evictionKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},
		{
			name:       "forbid update of eviction for normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundpodEviction, nil, evictionKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unboundpodEviction, nil, evictionKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for normal unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "spec.nodeName set to itself",
		},

		// Missing pod
		{
			name:       "forbid delete of unknown pod",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Delete, false, mynode),
			err:        "not found",
		},
		{
			name:       "forbid create of eviction for unknown pod",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "not found",
		},
		{
			name:       "forbid update of eviction for unknown pod",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for unknown pod",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for unknown pod",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coremypod.Namespace, coremypod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "not found",
		},

		// Eviction for unnamed pod
		{
			name:       "allow create of eviction for unnamed pod",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coreunnamedpod.Namespace, coreunnamedpod.Name, podResource, "eviction", admission.Create, false, mynode),
			// use the submitted eviction resource name as the pod name
			err: "",
		},
		{
			name:       "forbid update of eviction for unnamed pod",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coreunnamedpod.Namespace, coreunnamedpod.Name, podResource, "eviction", admission.Update, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid delete of eviction for unnamed pod",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mypodEviction, nil, evictionKind, coreunnamedpod.Namespace, coreunnamedpod.Name, podResource, "eviction", admission.Delete, false, mynode),
			err:        "forbidden: unexpected operation",
		},
		{
			name:       "forbid create of unnamed eviction for unnamed pod",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(unnamedEviction, nil, evictionKind, coreunnamedpod.Namespace, coreunnamedpod.Name, podResource, "eviction", admission.Create, false, mynode),
			err:        "could not determine pod from request data",
		},

		// Resource pods
		{
			name:       "forbid create of pod referencing service account",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(sapod, nil, podKind, sapod.Namespace, sapod.Name, podResource, "", admission.Create, false, mynode),
			err:        "reference a service account",
		},
		{
			name:       "forbid create of pod referencing secret",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(secretpod, nil, podKind, secretpod.Namespace, secretpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "reference secrets",
		},
		{
			name:       "forbid create of pod referencing configmap",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(configmappod, nil, podKind, configmappod.Namespace, configmappod.Name, podResource, "", admission.Create, false, mynode),
			err:        "reference configmaps",
		},
		{
			name:       "forbid create of pod referencing persistentvolumeclaim",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(pvcpod, nil, podKind, pvcpod.Namespace, pvcpod.Name, podResource, "", admission.Create, false, mynode),
			err:        "reference persistentvolumeclaims",
		},

		// My node object
		{
			name:       "allow create of my node",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, nil, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "allow create of my node pulling name from object",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, nil, nodeKind, mynodeObj.Namespace, "", nodeResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "allow create of my node with taints",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mynodeObjTaintA, nil, nodeKind, mynodeObj.Namespace, "", nodeResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "allow create of my node with labels",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(setAllowedCreateLabels(mynodeObj, ""), nil, nodeKind, mynodeObj.Namespace, "", nodeResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "forbid create of my node with forbidden labels",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(setForbiddenCreateLabels(mynodeObj, ""), nil, nodeKind, mynodeObj.Namespace, "", nodeResource, "", admission.Create, false, mynode),
			err:        `cannot set labels: foo.node-restriction.kubernetes.io/foo, node-restriction.kubernetes.io/foo`,
		},
		{
			name:       "allow update of my node",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow delete of my node",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Delete, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node status",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "status", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "forbid create of my node with non-nil configSource",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(mynodeObjConfigA, nil, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Create, false, mynode),
			err:        "create with non-nil configSource",
		},
		{
			name:       "forbid update of my node: nil configSource to new non-nil configSource",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjConfigA, mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "update configSource to a new non-nil configSource",
		},
		{
			name:       "forbid update of my node: non-nil configSource to new non-nil configSource",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjConfigB, mynodeObjConfigA, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "update configSource to a new non-nil configSource",
		},
		{
			name:       "allow update of my node: non-nil configSource unchanged",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjConfigA, mynodeObjConfigA, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: non-nil configSource to nil configSource",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, mynodeObjConfigA, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: no change to taints",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjTaintA, mynodeObjTaintA, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: add allowed labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setAllowedUpdateLabels(mynodeObj, ""), mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: remove allowed labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, setAllowedUpdateLabels(mynodeObj, ""), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: modify allowed labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setAllowedUpdateLabels(mynodeObj, "b"), setAllowedUpdateLabels(mynodeObj, "a"), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: no change to labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setAllLabels(mynodeObj, ""), setAllLabels(mynodeObj, ""), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: add allowed labels while forbidden labels exist unmodified",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setAllLabels(mynodeObj, ""), setForbiddenUpdateLabels(mynodeObj, ""), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of my node: remove allowed labels while forbidden labels exist unmodified",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setForbiddenUpdateLabels(mynodeObj, ""), setAllLabels(mynodeObj, ""), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "forbid update of my node: add taints",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjTaintA, mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "cannot modify taints",
		},
		{
			name:       "forbid update of my node: remove taints",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, mynodeObjTaintA, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "cannot modify taints",
		},
		{
			name:       "forbid update of my node: change taints",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObjTaintA, mynodeObjTaintB, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "cannot modify taints",
		},
		{
			name:       "forbid update of my node: add labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setForbiddenUpdateLabels(mynodeObj, ""), mynodeObj, nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        `cannot modify labels: foo.node-restriction.kubernetes.io/foo, node-restriction.kubernetes.io/foo, other.k8s.io/foo, other.kubernetes.io/foo`,
		},
		{
			name:       "forbid update of my node: remove labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(mynodeObj, setForbiddenUpdateLabels(mynodeObj, ""), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        `cannot modify labels: foo.node-restriction.kubernetes.io/foo, node-restriction.kubernetes.io/foo, other.k8s.io/foo, other.kubernetes.io/foo`,
		},
		{
			name:       "forbid update of my node: change labels",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(setForbiddenUpdateLabels(mynodeObj, "new"), setForbiddenUpdateLabels(mynodeObj, "old"), nodeKind, mynodeObj.Namespace, mynodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        `cannot modify labels: foo.node-restriction.kubernetes.io/foo, node-restriction.kubernetes.io/foo, other.k8s.io/foo, other.kubernetes.io/foo`,
		},

		// Other node object
		{
			name:       "forbid create of other node",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(othernodeObj, nil, nodeKind, othernodeObj.Namespace, othernodeObj.Name, nodeResource, "", admission.Create, false, mynode),
			err:        "cannot modify node",
		},
		{
			name:       "forbid create of other node pulling name from object",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(othernodeObj, nil, nodeKind, othernodeObj.Namespace, "", nodeResource, "", admission.Create, false, mynode),
			err:        "cannot modify node",
		},
		{
			name:       "forbid update of other node",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(othernodeObj, othernodeObj, nodeKind, othernodeObj.Namespace, othernodeObj.Name, nodeResource, "", admission.Update, false, mynode),
			err:        "cannot modify node",
		},
		{
			name:       "forbid delete of other node",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, nodeKind, othernodeObj.Namespace, othernodeObj.Name, nodeResource, "", admission.Delete, false, mynode),
			err:        "cannot modify node",
		},
		{
			name:       "forbid update of other node status",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(othernodeObj, othernodeObj, nodeKind, othernodeObj.Namespace, othernodeObj.Name, nodeResource, "status", admission.Update, false, mynode),
			err:        "cannot modify node",
		},

		// Service accounts
		{
			name:       "forbid create of unbound token",
			podsGetter: noExistingPods,
			features:   trEnabledFeature,
			attributes: admission.NewAttributesRecord(makeTokenRequest("", ""), nil, tokenrequestKind, "ns", "mysa", svcacctResource, "token", admission.Create, false, mynode),
			err:        "not bound to a pod",
		},
		{
			name:       "forbid create of token bound to nonexistant pod",
			podsGetter: noExistingPods,
			features:   trEnabledFeature,
			attributes: admission.NewAttributesRecord(makeTokenRequest("nopod", "someuid"), nil, tokenrequestKind, "ns", "mysa", svcacctResource, "token", admission.Create, false, mynode),
			err:        "not found",
		},
		{
			name:       "forbid create of token bound to pod without uid",
			podsGetter: existingPods,
			features:   trEnabledFeature,
			attributes: admission.NewAttributesRecord(makeTokenRequest(coremypod.Name, ""), nil, tokenrequestKind, "ns", "mysa", svcacctResource, "token", admission.Create, false, mynode),
			err:        "pod binding without a uid",
		},
		{
			name:       "forbid create of token bound to pod scheduled on another node",
			podsGetter: existingPods,
			features:   trEnabledFeature,
			attributes: admission.NewAttributesRecord(makeTokenRequest(coreotherpod.Name, coreotherpod.UID), nil, tokenrequestKind, coreotherpod.Namespace, "mysa", svcacctResource, "token", admission.Create, false, mynode),
			err:        "pod scheduled on a different node",
		},
		{
			name:       "allow create of token bound to pod scheduled this node",
			podsGetter: existingPods,
			features:   trEnabledFeature,
			attributes: admission.NewAttributesRecord(makeTokenRequest(coremypod.Name, coremypod.UID), nil, tokenrequestKind, coremypod.Namespace, "mysa", svcacctResource, "token", admission.Create, false, mynode),
		},

		// Unrelated objects
		{
			name:       "allow create of unrelated object",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(&api.ConfigMap{}, nil, configmapKind, "myns", "mycm", configmapResource, "", admission.Create, false, mynode),
			err:        "",
		},
		{
			name:       "allow update of unrelated object",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(&api.ConfigMap{}, &api.ConfigMap{}, configmapKind, "myns", "mycm", configmapResource, "", admission.Update, false, mynode),
			err:        "",
		},
		{
			name:       "allow delete of unrelated object",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, configmapKind, "myns", "mycm", configmapResource, "", admission.Delete, false, mynode),
			err:        "",
		},

		// Unrelated user
		{
			name:       "allow unrelated user creating a normal pod unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Create, false, bob),
			err:        "",
		},
		{
			name:       "allow unrelated user update of normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, coreunboundpod, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Update, false, bob),
			err:        "",
		},
		{
			name:       "allow unrelated user delete of normal pod unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "", admission.Delete, false, bob),
			err:        "",
		},
		{
			name:       "allow unrelated user create of normal pod status unbound",
			podsGetter: noExistingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Create, false, bob),
			err:        "",
		},
		{
			name:       "allow unrelated user update of normal pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(coreunboundpod, coreunboundpod, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Update, false, bob),
			err:        "",
		},
		{
			name:       "allow unrelated user delete of normal pod status unbound",
			podsGetter: existingPods,
			attributes: admission.NewAttributesRecord(nil, nil, podKind, coreunboundpod.Namespace, coreunboundpod.Name, podResource, "status", admission.Delete, false, bob),
			err:        "",
		},
		// Node leases
		{
			name:       "disallowed create lease - feature disabled",
			attributes: admission.NewAttributesRecord(lease, nil, leaseKind, lease.Namespace, lease.Name, leaseResource, "", admission.Create, false, mynode),
			features:   leaseDisabledFeature,
			err:        "forbidden: disabled by feature gate NodeLease",
		},
		{
			name:       "disallowed create lease in namespace other than kube-node-lease - feature enabled",
			attributes: admission.NewAttributesRecord(leaseWrongNS, nil, leaseKind, leaseWrongNS.Namespace, leaseWrongNS.Name, leaseResource, "", admission.Create, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed update lease in namespace other than kube-node-lease - feature enabled",
			attributes: admission.NewAttributesRecord(leaseWrongNS, leaseWrongNS, leaseKind, leaseWrongNS.Namespace, leaseWrongNS.Name, leaseResource, "", admission.Update, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed delete lease in namespace other than kube-node-lease - feature enabled",
			attributes: admission.NewAttributesRecord(nil, nil, leaseKind, leaseWrongNS.Namespace, leaseWrongNS.Name, leaseResource, "", admission.Delete, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed create another node's lease - feature enabled",
			attributes: admission.NewAttributesRecord(leaseWrongName, nil, leaseKind, leaseWrongName.Namespace, leaseWrongName.Name, leaseResource, "", admission.Create, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed update another node's lease - feature enabled",
			attributes: admission.NewAttributesRecord(leaseWrongName, leaseWrongName, leaseKind, leaseWrongName.Namespace, leaseWrongName.Name, leaseResource, "", admission.Update, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed delete another node's lease - feature enabled",
			attributes: admission.NewAttributesRecord(nil, nil, leaseKind, leaseWrongName.Namespace, leaseWrongName.Name, leaseResource, "", admission.Delete, false, mynode),
			features:   leaseEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "allowed create node lease - feature enabled",
			attributes: admission.NewAttributesRecord(lease, nil, leaseKind, lease.Namespace, lease.Name, leaseResource, "", admission.Create, false, mynode),
			features:   leaseEnabledFeature,
			err:        "",
		},
		{
			name:       "allowed update node lease - feature enabled",
			attributes: admission.NewAttributesRecord(lease, lease, leaseKind, lease.Namespace, lease.Name, leaseResource, "", admission.Update, false, mynode),
			features:   leaseEnabledFeature,
			err:        "",
		},
		{
			name:       "allowed delete node lease - feature enabled",
			attributes: admission.NewAttributesRecord(nil, nil, leaseKind, lease.Namespace, lease.Name, leaseResource, "", admission.Delete, false, mynode),
			features:   leaseEnabledFeature,
			err:        "",
		},
		// CSINodeInfo
		{
			name:       "disallowed create CSINodeInfo - feature disabled",
			attributes: admission.NewAttributesRecord(nodeInfo, nil, csiNodeInfoKind, nodeInfo.Namespace, nodeInfo.Name, csiNodeInfoResource, "", admission.Create, false, mynode),
			features:   csiNodeInfoDisabledFeature,
			err:        fmt.Sprintf("forbidden: disabled by feature gates %s and %s", features.KubeletPluginsWatcher, features.CSINodeInfo),
		},
		{
			name:       "disallowed create another node's CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nodeInfoWrongName, nil, csiNodeInfoKind, nodeInfoWrongName.Namespace, nodeInfoWrongName.Name, csiNodeInfoResource, "", admission.Create, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed update another node's CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nodeInfoWrongName, nodeInfoWrongName, csiNodeInfoKind, nodeInfoWrongName.Namespace, nodeInfoWrongName.Name, csiNodeInfoResource, "", admission.Update, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "disallowed delete another node's CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nil, nil, csiNodeInfoKind, nodeInfoWrongName.Namespace, nodeInfoWrongName.Name, csiNodeInfoResource, "", admission.Delete, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "forbidden: ",
		},
		{
			name:       "allowed create node CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nodeInfo, nil, csiNodeInfoKind, nodeInfo.Namespace, nodeInfo.Name, csiNodeInfoResource, "", admission.Create, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "",
		},
		{
			name:       "allowed update node CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nodeInfo, nodeInfo, csiNodeInfoKind, nodeInfo.Namespace, nodeInfo.Name, csiNodeInfoResource, "", admission.Update, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "",
		},
		{
			name:       "allowed delete node CSINodeInfo - feature enabled",
			attributes: admission.NewAttributesRecord(nil, nil, csiNodeInfoKind, nodeInfo.Namespace, nodeInfo.Name, csiNodeInfoResource, "", admission.Delete, false, mynode),
			features:   csiNodeInfoEnabledFeature,
			err:        "",
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := NewPlugin(nodeidentifier.NewDefaultNodeIdentifier())
			if tt.features != nil {
				c.features = tt.features
			}
			c.podsGetter = tt.podsGetter
			err := c.Admit(tt.attributes)
			if (err == nil) != (len(tt.err) == 0) {
				t.Errorf("nodePlugin.Admit() error = %v, expected %v", err, tt.err)
				return
			}
			if len(tt.err) > 0 && !strings.Contains(err.Error(), tt.err) {
				t.Errorf("nodePlugin.Admit() error = %v, expected %v", err, tt.err)
			}
		})
	}
}

func Test_getModifiedLabels(t *testing.T) {
	tests := []struct {
		name string
		a    map[string]string
		b    map[string]string
		want sets.String
	}{
		{
			name: "empty",
			a:    nil,
			b:    nil,
			want: sets.NewString(),
		},
		{
			name: "no change",
			a:    map[string]string{"x": "1", "y": "2", "z": "3"},
			b:    map[string]string{"x": "1", "y": "2", "z": "3"},
			want: sets.NewString(),
		},
		{
			name: "added",
			a:    map[string]string{},
			b:    map[string]string{"a": "0"},
			want: sets.NewString("a"),
		},
		{
			name: "removed",
			a:    map[string]string{"z": "3"},
			b:    map[string]string{},
			want: sets.NewString("z"),
		},
		{
			name: "changed",
			a:    map[string]string{"z": "3"},
			b:    map[string]string{"z": "4"},
			want: sets.NewString("z"),
		},
		{
			name: "added empty",
			a:    map[string]string{},
			b:    map[string]string{"a": ""},
			want: sets.NewString("a"),
		},
		{
			name: "removed empty",
			a:    map[string]string{"z": ""},
			b:    map[string]string{},
			want: sets.NewString("z"),
		},
		{
			name: "changed to empty",
			a:    map[string]string{"z": "3"},
			b:    map[string]string{"z": ""},
			want: sets.NewString("z"),
		},
		{
			name: "changed from empty",
			a:    map[string]string{"z": ""},
			b:    map[string]string{"z": "3"},
			want: sets.NewString("z"),
		},
		{
			name: "added, removed, and changed",
			a:    map[string]string{"a": "1", "b": "2"},
			b:    map[string]string{"a": "2", "c": "3"},
			want: sets.NewString("a", "b", "c"),
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := getModifiedLabels(tt.a, tt.b); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("getModifiedLabels() = %v, want %v", got, tt.want)
			}
		})
	}
}
