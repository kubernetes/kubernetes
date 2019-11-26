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
	"context"
	"fmt"
	"io"
	"strings"

	v1 "k8s.io/api/core/v1"
	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/diff"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	apiserveradmission "k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1lister "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	authenticationapi "k8s.io/kubernetes/pkg/apis/authentication"
	coordapi "k8s.io/kubernetes/pkg/apis/coordination"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/policy"
	storage "k8s.io/kubernetes/pkg/apis/storage"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/features"
	kubeletapis "k8s.io/kubernetes/pkg/kubelet/apis"
)

// PluginName is a string with the name of the plugin
const PluginName = "NodeRestriction"

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(nodeidentifier.NewDefaultNodeIdentifier()), nil
	})
}

// NewPlugin creates a new NodeRestriction admission plugin.
// This plugin identifies requests from nodes
func NewPlugin(nodeIdentifier nodeidentifier.NodeIdentifier) *Plugin {
	return &Plugin{
		Handler:        admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		nodeIdentifier: nodeIdentifier,
	}
}

// Plugin holds state for and implements the admission plugin.
type Plugin struct {
	*admission.Handler
	nodeIdentifier nodeidentifier.NodeIdentifier
	podsGetter     corev1lister.PodLister
	nodesGetter    corev1lister.NodeLister

	tokenRequestEnabled            bool
	csiNodeInfoEnabled             bool
	expandPersistentVolumesEnabled bool
}

var (
	_ admission.Interface                                 = &Plugin{}
	_ apiserveradmission.WantsExternalKubeInformerFactory = &Plugin{}
	_ apiserveradmission.WantsFeatures                    = &Plugin{}
)

// InspectFeatureGates allows setting bools without taking a dep on a global variable
func (p *Plugin) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	p.tokenRequestEnabled = featureGates.Enabled(features.TokenRequest)
	p.csiNodeInfoEnabled = featureGates.Enabled(features.CSINodeInfo)
	p.expandPersistentVolumesEnabled = featureGates.Enabled(features.ExpandPersistentVolumes)
}

// SetExternalKubeInformerFactory registers an informer factory into Plugin
func (p *Plugin) SetExternalKubeInformerFactory(f informers.SharedInformerFactory) {
	p.podsGetter = f.Core().V1().Pods().Lister()
	p.nodesGetter = f.Core().V1().Nodes().Lister()
}

// ValidateInitialization validates the Plugin was initialized properly
func (p *Plugin) ValidateInitialization() error {
	if p.nodeIdentifier == nil {
		return fmt.Errorf("%s requires a node identifier", PluginName)
	}
	if p.podsGetter == nil {
		return fmt.Errorf("%s requires a pod getter", PluginName)
	}
	if p.nodesGetter == nil {
		return fmt.Errorf("%s requires a node getter", PluginName)
	}
	return nil
}

var (
	podResource     = api.Resource("pods")
	nodeResource    = api.Resource("nodes")
	pvcResource     = api.Resource("persistentvolumeclaims")
	svcacctResource = api.Resource("serviceaccounts")
	leaseResource   = coordapi.Resource("leases")
	csiNodeResource = storage.Resource("csinodes")
)

// Admit checks the admission policy and triggers corresponding actions
func (p *Plugin) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	nodeName, isNode := p.nodeIdentifier.NodeIdentity(a.GetUserInfo())

	// Our job is just to restrict nodes
	if !isNode {
		return nil
	}

	if len(nodeName) == 0 {
		// disallow requests we cannot match to a particular node
		return admission.NewForbidden(a, fmt.Errorf("could not determine node from user %q", a.GetUserInfo().GetName()))
	}

	// TODO: if node doesn't exist and this isn't a create node request, then reject.

	switch a.GetResource().GroupResource() {
	case podResource:
		switch a.GetSubresource() {
		case "":
			return p.admitPod(nodeName, a)
		case "status":
			return p.admitPodStatus(nodeName, a)
		case "eviction":
			return p.admitPodEviction(nodeName, a)
		default:
			return admission.NewForbidden(a, fmt.Errorf("unexpected pod subresource %q, only 'status' and 'eviction' are allowed", a.GetSubresource()))
		}

	case nodeResource:
		return p.admitNode(nodeName, a)

	case pvcResource:
		switch a.GetSubresource() {
		case "status":
			return p.admitPVCStatus(nodeName, a)
		default:
			return admission.NewForbidden(a, fmt.Errorf("may only update PVC status"))
		}

	case svcacctResource:
		if p.tokenRequestEnabled {
			return p.admitServiceAccount(nodeName, a)
		}
		return nil

	case leaseResource:
		return p.admitLease(nodeName, a)

	case csiNodeResource:
		if p.csiNodeInfoEnabled {
			return p.admitCSINode(nodeName, a)
		}
		return admission.NewForbidden(a, fmt.Errorf("disabled by feature gates %s", features.CSINodeInfo))

	default:
		return nil
	}
}

// admitPod allows creating or deleting a pod if it is assigned to the
// current node and fulfills related criteria.
func (p *Plugin) admitPod(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Create:
		return p.admitPodCreate(nodeName, a)

	case admission.Delete:
		// get the existing pod
		existingPod, err := p.podsGetter.Pods(a.GetNamespace()).Get(a.GetName())
		if errors.IsNotFound(err) {
			return err
		}
		if err != nil {
			return admission.NewForbidden(a, err)
		}
		// only allow a node to delete a pod bound to itself
		if existingPod.Spec.NodeName != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("node %q can only delete pods with spec.nodeName set to itself", nodeName))
		}
		return nil

	default:
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %q, node %q can only create and delete mirror pods", a.GetOperation(), nodeName))
	}
}

func (p *Plugin) admitPodCreate(nodeName string, a admission.Attributes) error {
	// require a pod object
	pod, ok := a.GetObject().(*api.Pod)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	// only allow nodes to create mirror pods
	if _, isMirrorPod := pod.Annotations[api.MirrorPodAnnotationKey]; !isMirrorPod {
		return admission.NewForbidden(a, fmt.Errorf("pod does not have %q annotation, node %q can only create mirror pods", api.MirrorPodAnnotationKey, nodeName))
	}

	// only allow nodes to create a pod bound to itself
	if pod.Spec.NodeName != nodeName {
		return admission.NewForbidden(a, fmt.Errorf("node %q can only create pods with spec.nodeName set to itself", nodeName))
	}
	if len(pod.OwnerReferences) > 1 {
		return admission.NewForbidden(a, fmt.Errorf("node %q can only create pods with a single owner reference set to itself", nodeName))
	}
	if len(pod.OwnerReferences) == 1 {
		owner := pod.OwnerReferences[0]
		if owner.APIVersion != v1.SchemeGroupVersion.String() ||
			owner.Kind != "Node" ||
			owner.Name != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("node %q can only create pods with an owner reference set to itself", nodeName))
		}
		if owner.Controller == nil || !*owner.Controller {
			return admission.NewForbidden(a, fmt.Errorf("node %q can only create pods with a controller owner reference set to itself", nodeName))
		}
		if owner.BlockOwnerDeletion != nil && *owner.BlockOwnerDeletion {
			return admission.NewForbidden(a, fmt.Errorf("node %q must not set blockOwnerDeletion on an owner reference", nodeName))
		}

		// Verify the node UID.
		node, err := p.nodesGetter.Get(nodeName)
		if errors.IsNotFound(err) {
			return err
		}
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("error looking up node %s to verify uid: %v", nodeName, err))
		}
		if owner.UID != node.UID {
			return admission.NewForbidden(a, fmt.Errorf("node %s UID mismatch: expected %s got %s", nodeName, owner.UID, node.UID))
		}
	}

	// don't allow a node to create a pod that references any other API objects
	if pod.Spec.ServiceAccountName != "" {
		return admission.NewForbidden(a, fmt.Errorf("node %q can not create pods that reference a service account", nodeName))
	}
	hasSecrets := false
	podutil.VisitPodSecretNames(pod, func(name string) (shouldContinue bool) { hasSecrets = true; return false })
	if hasSecrets {
		return admission.NewForbidden(a, fmt.Errorf("node %q can not create pods that reference secrets", nodeName))
	}
	hasConfigMaps := false
	podutil.VisitPodConfigmapNames(pod, func(name string) (shouldContinue bool) { hasConfigMaps = true; return false })
	if hasConfigMaps {
		return admission.NewForbidden(a, fmt.Errorf("node %q can not create pods that reference configmaps", nodeName))
	}
	for _, v := range pod.Spec.Volumes {
		if v.PersistentVolumeClaim != nil {
			return admission.NewForbidden(a, fmt.Errorf("node %q can not create pods that reference persistentvolumeclaims", nodeName))
		}
	}

	return nil
}

// admitPodStatus allows to update the status of a pod if it is
// assigned to the current node.
func (p *Plugin) admitPodStatus(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Update:
		// require an existing pod
		oldPod, ok := a.GetOldObject().(*api.Pod)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetOldObject()))
		}
		// only allow a node to update status of a pod bound to itself
		if oldPod.Spec.NodeName != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("node %q can only update pod status for pods with spec.nodeName set to itself", nodeName))
		}
		newPod, ok := a.GetObject().(*api.Pod)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}
		if !labels.Equals(oldPod.Labels, newPod.Labels) {
			return admission.NewForbidden(a, fmt.Errorf("node %q cannot update labels through pod status", nodeName))
		}
		return nil

	default:
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %q", a.GetOperation()))
	}
}

// admitPodEviction allows to evict a pod if it is assigned to the current node.
func (p *Plugin) admitPodEviction(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Create:
		// require eviction to an existing pod object
		eviction, ok := a.GetObject().(*policy.Eviction)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}
		// use pod name from the admission attributes, if set, rather than from the submitted Eviction object
		podName := a.GetName()
		if len(podName) == 0 {
			if len(eviction.Name) == 0 {
				return admission.NewForbidden(a, fmt.Errorf("could not determine pod from request data"))
			}
			podName = eviction.Name
		}
		// get the existing pod
		existingPod, err := p.podsGetter.Pods(a.GetNamespace()).Get(podName)
		if errors.IsNotFound(err) {
			return err
		}
		if err != nil {
			return admission.NewForbidden(a, err)
		}
		// only allow a node to evict a pod bound to itself
		if existingPod.Spec.NodeName != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("node %s can only evict pods with spec.nodeName set to itself", nodeName))
		}
		return nil

	default:
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %s", a.GetOperation()))
	}
}

func (p *Plugin) admitPVCStatus(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Update:
		if !p.expandPersistentVolumesEnabled {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to update persistentvolumeclaim metadata", nodeName))
		}

		oldPVC, ok := a.GetOldObject().(*api.PersistentVolumeClaim)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetOldObject()))
		}

		newPVC, ok := a.GetObject().(*api.PersistentVolumeClaim)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}

		// make copies for comparison
		oldPVC = oldPVC.DeepCopy()
		newPVC = newPVC.DeepCopy()

		// zero out resourceVersion to avoid comparing differences,
		// since the new object could leave it empty to indicate an unconditional update
		oldPVC.ObjectMeta.ResourceVersion = ""
		newPVC.ObjectMeta.ResourceVersion = ""

		oldPVC.Status.Capacity = nil
		newPVC.Status.Capacity = nil

		oldPVC.Status.Conditions = nil
		newPVC.Status.Conditions = nil

		// TODO(apelisse): We don't have a good mechanism to
		// verify that only the things that should have changed
		// have changed. Ignore it for now.
		oldPVC.ObjectMeta.ManagedFields = nil
		newPVC.ObjectMeta.ManagedFields = nil

		// ensure no metadata changed. nodes should not be able to relabel, add finalizers/owners, etc
		if !apiequality.Semantic.DeepEqual(oldPVC, newPVC) {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to update fields other than status.capacity and status.conditions: %v", nodeName, diff.ObjectReflectDiff(oldPVC, newPVC)))
		}

		return nil

	default:
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %q", a.GetOperation()))
	}
}

func (p *Plugin) admitNode(nodeName string, a admission.Attributes) error {
	requestedName := a.GetName()
	if a.GetOperation() == admission.Create {
		node, ok := a.GetObject().(*api.Node)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}

		// Don't allow a node to create its Node API object with the config source set.
		// We scope node access to things listed in the Node.Spec, so allowing this would allow a view escalation.
		if node.Spec.ConfigSource != nil {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to create pods with a non-nil configSource", nodeName))
		}

		// Don't allow a node to register with labels outside the allowed set.
		// This would allow a node to add or modify its labels in a way that would let it steer privileged workloads to itself.
		modifiedLabels := getModifiedLabels(node.Labels, nil)
		if forbiddenLabels := p.getForbiddenCreateLabels(modifiedLabels); len(forbiddenLabels) > 0 {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to set the following labels: %s", nodeName, strings.Join(forbiddenLabels.List(), ", ")))
		}
		// check and warn if nodes set labels on create that would have been forbidden on update
		// TODO(liggitt): in 1.19, expand getForbiddenCreateLabels to match getForbiddenUpdateLabels and drop this
		if forbiddenUpdateLabels := p.getForbiddenUpdateLabels(modifiedLabels); len(forbiddenUpdateLabels) > 0 {
			klog.Warningf("node %q added disallowed labels on node creation: %s", nodeName, strings.Join(forbiddenUpdateLabels.List(), ", "))
		}
	}
	if requestedName != nodeName {
		return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to modify node %q", nodeName, requestedName))
	}

	if a.GetOperation() == admission.Update {
		node, ok := a.GetObject().(*api.Node)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}
		oldNode, ok := a.GetOldObject().(*api.Node)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}

		// Don't allow a node to update the config source on its Node API object.
		// We scope node access to things listed in the Node.Spec, so allowing this would allow a view escalation.
		// We only do the check if the new node's configSource is non-nil; old kubelets might drop the field during a status update.
		if node.Spec.ConfigSource != nil && !apiequality.Semantic.DeepEqual(node.Spec.ConfigSource, oldNode.Spec.ConfigSource) {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to update configSource to a new non-nil configSource", nodeName))
		}

		// Don't allow a node to update its own taints. This would allow a node to remove or modify its
		// taints in a way that would let it steer disallowed workloads to itself.
		if !apiequality.Semantic.DeepEqual(node.Spec.Taints, oldNode.Spec.Taints) {
			return admission.NewForbidden(a, fmt.Errorf("node %q is not allowed to modify taints", nodeName))
		}

		// Don't allow a node to update labels outside the allowed set.
		// This would allow a node to add or modify its labels in a way that would let it steer privileged workloads to itself.
		modifiedLabels := getModifiedLabels(node.Labels, oldNode.Labels)
		if forbiddenUpdateLabels := p.getForbiddenUpdateLabels(modifiedLabels); len(forbiddenUpdateLabels) > 0 {
			return admission.NewForbidden(a, fmt.Errorf("is not allowed to modify labels: %s", strings.Join(forbiddenUpdateLabels.List(), ", ")))
		}
	}

	return nil
}

// getModifiedLabels returns the set of label keys that are different between the two maps
func getModifiedLabels(a, b map[string]string) sets.String {
	modified := sets.NewString()
	for k, v1 := range a {
		if v2, ok := b[k]; !ok || v1 != v2 {
			modified.Insert(k)
		}
	}
	for k, v1 := range b {
		if v2, ok := a[k]; !ok || v1 != v2 {
			modified.Insert(k)
		}
	}
	return modified
}

func isKubernetesLabel(key string) bool {
	namespace := getLabelNamespace(key)
	if namespace == "kubernetes.io" || strings.HasSuffix(namespace, ".kubernetes.io") {
		return true
	}
	if namespace == "k8s.io" || strings.HasSuffix(namespace, ".k8s.io") {
		return true
	}
	return false
}

func getLabelNamespace(key string) string {
	if parts := strings.SplitN(key, "/", 2); len(parts) == 2 {
		return parts[0]
	}
	return ""
}

// getForbiddenCreateLabels returns the set of labels that may not be set by the node.
// TODO(liggitt): in 1.19, expand to match getForbiddenUpdateLabels()
func (p *Plugin) getForbiddenCreateLabels(modifiedLabels sets.String) sets.String {
	if len(modifiedLabels) == 0 {
		return nil
	}

	forbiddenLabels := sets.NewString()
	for label := range modifiedLabels {
		namespace := getLabelNamespace(label)
		// forbid kubelets from setting node-restriction labels
		if namespace == v1.LabelNamespaceNodeRestriction || strings.HasSuffix(namespace, "."+v1.LabelNamespaceNodeRestriction) {
			forbiddenLabels.Insert(label)
		}
	}
	return forbiddenLabels
}

// getForbiddenLabels returns the set of labels that may not be set by the node on update.
func (p *Plugin) getForbiddenUpdateLabels(modifiedLabels sets.String) sets.String {
	if len(modifiedLabels) == 0 {
		return nil
	}

	forbiddenLabels := sets.NewString()
	for label := range modifiedLabels {
		namespace := getLabelNamespace(label)
		// forbid kubelets from setting node-restriction labels
		if namespace == v1.LabelNamespaceNodeRestriction || strings.HasSuffix(namespace, "."+v1.LabelNamespaceNodeRestriction) {
			forbiddenLabels.Insert(label)
		}
		// forbid kubelets from setting unknown kubernetes.io and k8s.io labels on update
		if isKubernetesLabel(label) && !kubeletapis.IsKubeletLabel(label) {
			// TODO: defer to label policy once available
			forbiddenLabels.Insert(label)
		}
	}
	return forbiddenLabels
}

func (p *Plugin) admitServiceAccount(nodeName string, a admission.Attributes) error {
	if a.GetOperation() != admission.Create {
		return nil
	}
	if a.GetSubresource() != "token" {
		return nil
	}
	tr, ok := a.GetObject().(*authenticationapi.TokenRequest)
	if !ok {
		return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
	}

	// TokenRequests from a node must have a pod binding. That pod must be
	// scheduled on the node.
	ref := tr.Spec.BoundObjectRef
	if ref == nil ||
		ref.APIVersion != "v1" ||
		ref.Kind != "Pod" ||
		ref.Name == "" {
		return admission.NewForbidden(a, fmt.Errorf("node requested token not bound to a pod"))
	}
	if ref.UID == "" {
		return admission.NewForbidden(a, fmt.Errorf("node requested token with a pod binding without a uid"))
	}
	pod, err := p.podsGetter.Pods(a.GetNamespace()).Get(ref.Name)
	if errors.IsNotFound(err) {
		return err
	}
	if err != nil {
		return admission.NewForbidden(a, err)
	}
	if ref.UID != pod.UID {
		return admission.NewForbidden(a, fmt.Errorf("the UID in the bound object reference (%s) does not match the UID in record (%s). The object might have been deleted and then recreated", ref.UID, pod.UID))
	}
	if pod.Spec.NodeName != nodeName {
		return admission.NewForbidden(a, fmt.Errorf("node requested token bound to a pod scheduled on a different node"))
	}

	return nil
}

func (p *Plugin) admitLease(nodeName string, a admission.Attributes) error {
	// the request must be against the system namespace reserved for node leases
	if a.GetNamespace() != api.NamespaceNodeLease {
		return admission.NewForbidden(a, fmt.Errorf("can only access leases in the %q system namespace", api.NamespaceNodeLease))
	}

	// the request must come from a node with the same name as the lease
	if a.GetOperation() == admission.Create {
		// a.GetName() won't return the name on create, so we drill down to the proposed object
		lease, ok := a.GetObject().(*coordapi.Lease)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}
		if lease.Name != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("can only access node lease with the same name as the requesting node"))
		}
	} else {
		if a.GetName() != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("can only access node lease with the same name as the requesting node"))
		}
	}

	return nil
}

func (p *Plugin) admitCSINode(nodeName string, a admission.Attributes) error {
	// the request must come from a node with the same name as the CSINode object
	if a.GetOperation() == admission.Create {
		// a.GetName() won't return the name on create, so we drill down to the proposed object
		accessor, err := meta.Accessor(a.GetObject())
		if err != nil {
			return admission.NewForbidden(a, fmt.Errorf("unable to access the object name"))
		}
		if accessor.GetName() != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("can only access CSINode with the same name as the requesting node"))
		}
	} else {
		if a.GetName() != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("can only access CSINode with the same name as the requesting node"))
		}
	}

	return nil
}
