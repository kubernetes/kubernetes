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
	"io"

	apiequality "k8s.io/apimachinery/pkg/api/equality"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	coreinternalversion "k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset/typed/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

const (
	PluginName = "NodeRestriction"
)

// Register registers a plugin
func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName, func(config io.Reader) (admission.Interface, error) {
		return NewPlugin(nodeidentifier.NewDefaultNodeIdentifier()), nil
	})
}

// NewPlugin creates a new NodeRestriction admission plugin.
// This plugin identifies requests from nodes
func NewPlugin(nodeIdentifier nodeidentifier.NodeIdentifier) *nodePlugin {
	return &nodePlugin{
		Handler:        admission.NewHandler(admission.Create, admission.Update, admission.Delete),
		nodeIdentifier: nodeIdentifier,
	}
}

// nodePlugin holds state for and implements the admission plugin.
type nodePlugin struct {
	*admission.Handler
	nodeIdentifier nodeidentifier.NodeIdentifier
	podsGetter     coreinternalversion.PodsGetter
}

var (
	_ = admission.Interface(&nodePlugin{})
	_ = kubeapiserveradmission.WantsInternalKubeClientSet(&nodePlugin{})
)

func (p *nodePlugin) SetInternalKubeClientSet(f internalclientset.Interface) {
	p.podsGetter = f.Core()
}

func (p *nodePlugin) Validate() error {
	if p.nodeIdentifier == nil {
		return fmt.Errorf("%s requires a node identifier", PluginName)
	}
	if p.podsGetter == nil {
		return fmt.Errorf("%s requires a pod getter", PluginName)
	}
	return nil
}

var (
	podResource  = api.Resource("pods")
	nodeResource = api.Resource("nodes")
)

func (c *nodePlugin) Admit(a admission.Attributes) error {
	nodeName, isNode := c.nodeIdentifier.NodeIdentity(a.GetUserInfo())

	// Our job is just to restrict nodes
	if !isNode {
		return nil
	}

	if len(nodeName) == 0 {
		// disallow requests we cannot match to a particular node
		return admission.NewForbidden(a, fmt.Errorf("could not determine node from user %q", a.GetUserInfo().GetName()))
	}

	switch a.GetResource().GroupResource() {
	case podResource:
		switch a.GetSubresource() {
		case "":
			return c.admitPod(nodeName, a)
		case "status":
			return c.admitPodStatus(nodeName, a)
		case "eviction":
			return c.admitPodEviction(nodeName, a)
		default:
			return admission.NewForbidden(a, fmt.Errorf("unexpected pod subresource %q", a.GetSubresource()))
		}

	case nodeResource:
		return c.admitNode(nodeName, a)

	default:
		return nil
	}
}

func (c *nodePlugin) admitPod(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Create:
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

	case admission.Delete:
		// get the existing pod from the server cache
		existingPod, err := c.podsGetter.Pods(a.GetNamespace()).Get(a.GetName(), v1.GetOptions{ResourceVersion: "0"})
		if errors.IsNotFound(err) {
			// wasn't found in the server cache, do a live lookup before forbidding
			existingPod, err = c.podsGetter.Pods(a.GetNamespace()).Get(a.GetName(), v1.GetOptions{})
			if errors.IsNotFound(err) {
				return err
			}
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
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %q", a.GetOperation()))
	}
}

func (c *nodePlugin) admitPodStatus(nodeName string, a admission.Attributes) error {
	switch a.GetOperation() {
	case admission.Update:
		// require an existing pod
		pod, ok := a.GetOldObject().(*api.Pod)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}
		// only allow a node to update status of a pod bound to itself
		if pod.Spec.NodeName != nodeName {
			return admission.NewForbidden(a, fmt.Errorf("node %q can only update pod status for pods with spec.nodeName set to itself", nodeName))
		}
		return nil

	default:
		return admission.NewForbidden(a, fmt.Errorf("unexpected operation %q", a.GetOperation()))
	}
}

func (c *nodePlugin) admitPodEviction(nodeName string, a admission.Attributes) error {
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
		// get the existing pod from the server cache
		existingPod, err := c.podsGetter.Pods(a.GetNamespace()).Get(podName, v1.GetOptions{ResourceVersion: "0"})
		if errors.IsNotFound(err) {
			// wasn't found in the server cache, do a live lookup before forbidding
			existingPod, err = c.podsGetter.Pods(a.GetNamespace()).Get(podName, v1.GetOptions{})
			if errors.IsNotFound(err) {
				return err
			}
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

func (c *nodePlugin) admitNode(nodeName string, a admission.Attributes) error {
	requestedName := a.GetName()
	if a.GetOperation() == admission.Create {
		node, ok := a.GetObject().(*api.Node)
		if !ok {
			return admission.NewForbidden(a, fmt.Errorf("unexpected type %T", a.GetObject()))
		}

		// Don't allow a node to create its Node API object with the config source set.
		// We scope node access to things listed in the Node.Spec, so allowing this would allow a view escalation.
		if node.Spec.ConfigSource != nil {
			return admission.NewForbidden(a, fmt.Errorf("cannot create with non-nil configSource"))
		}

		// On create, get name from new object if unset in admission
		if len(requestedName) == 0 {
			requestedName = node.Name
		}
	}
	if requestedName != nodeName {
		return admission.NewForbidden(a, fmt.Errorf("node %q cannot modify node %q", nodeName, requestedName))
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
			return admission.NewForbidden(a, fmt.Errorf("cannot update configSource to a new non-nil configSource"))
		}
	}

	return nil
}
