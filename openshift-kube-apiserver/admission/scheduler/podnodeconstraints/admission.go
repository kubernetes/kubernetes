package podnodeconstraints

import (
	"context"
	"fmt"
	"io"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/authorization/authorizer"
	"k8s.io/klog/v2"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/auth/nodeidentifier"

	"github.com/openshift/library-go/pkg/config/helpers"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/apis/podnodeconstraints"
	v1 "k8s.io/kubernetes/openshift-kube-apiserver/admission/scheduler/apis/podnodeconstraints/v1"
)

const PluginName = "scheduling.openshift.io/PodNodeConstraints"

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			pluginConfig, err := readConfig(config)
			if err != nil {
				return nil, err
			}
			if pluginConfig == nil {
				klog.Infof("Admission plugin %q is not configured so it will be disabled.", PluginName)
				return nil, nil
			}
			return NewPodNodeConstraints(pluginConfig, nodeidentifier.NewDefaultNodeIdentifier()), nil
		})
}

// NewPodNodeConstraints creates a new admission plugin to prevent objects that contain pod templates
// from containing node bindings by name or selector based on role permissions.
func NewPodNodeConstraints(config *podnodeconstraints.PodNodeConstraintsConfig, nodeIdentifier nodeidentifier.NodeIdentifier) admission.Interface {
	plugin := podNodeConstraints{
		config:         config,
		Handler:        admission.NewHandler(admission.Create, admission.Update),
		nodeIdentifier: nodeIdentifier,
	}
	if config != nil {
		plugin.selectorLabelBlacklist = sets.NewString(config.NodeSelectorLabelBlacklist...)
	}

	return &plugin
}

type podNodeConstraints struct {
	*admission.Handler
	selectorLabelBlacklist sets.String
	config                 *podnodeconstraints.PodNodeConstraintsConfig
	authorizer             authorizer.Authorizer
	nodeIdentifier         nodeidentifier.NodeIdentifier
}

var _ = initializer.WantsAuthorizer(&podNodeConstraints{})
var _ = admission.ValidationInterface(&podNodeConstraints{})

func shouldCheckResource(resource schema.GroupResource, kind schema.GroupKind) (bool, error) {
	expectedKind, shouldCheck := resourcesToCheck[resource]
	if !shouldCheck {
		return false, nil
	}
	if expectedKind != kind {
		return false, fmt.Errorf("Unexpected resource kind %v for resource %v", &kind, &resource)
	}
	return true, nil
}

// resourcesToCheck is a map of resources and corresponding kinds of things that we want handled in this plugin
var resourcesToCheck = map[schema.GroupResource]schema.GroupKind{
	coreapi.Resource("pods"): coreapi.Kind("Pod"),
}

func readConfig(reader io.Reader) (*podnodeconstraints.PodNodeConstraintsConfig, error) {
	obj, err := helpers.ReadYAMLToInternal(reader, podnodeconstraints.Install, v1.Install)
	if err != nil {
		return nil, err
	}
	if obj == nil {
		return nil, nil
	}
	config, ok := obj.(*podnodeconstraints.PodNodeConstraintsConfig)
	if !ok {
		return nil, fmt.Errorf("unexpected config object: %#v", obj)
	}
	// No validation needed since config is just list of strings
	return config, nil
}

func (o *podNodeConstraints) Validate(ctx context.Context, attr admission.Attributes, _ admission.ObjectInterfaces) error {
	switch {
	case o.config == nil,
		attr.GetSubresource() != "":
		return nil
	}
	shouldCheck, err := shouldCheckResource(attr.GetResource().GroupResource(), attr.GetKind().GroupKind())
	if err != nil {
		return err
	}
	if !shouldCheck {
		return nil
	}
	// Only check Create operation on pods
	if attr.GetResource().GroupResource() == coreapi.Resource("pods") && attr.GetOperation() != admission.Create {
		return nil
	}

	return o.validatePodSpec(ctx, attr, attr.GetObject().(*coreapi.Pod).Spec)
}

// validate PodSpec if NodeName or NodeSelector are specified
func (o *podNodeConstraints) validatePodSpec(ctx context.Context, attr admission.Attributes, ps coreapi.PodSpec) error {
	// a node creating a mirror pod that targets itself is allowed
	// see the NodeRestriction plugin for further details
	if o.isNodeSelfTargetWithMirrorPod(attr, ps.NodeName) {
		return nil
	}

	matchingLabels := []string{}
	// nodeSelector blacklist filter
	for nodeSelectorLabel := range ps.NodeSelector {
		if o.selectorLabelBlacklist.Has(nodeSelectorLabel) {
			matchingLabels = append(matchingLabels, nodeSelectorLabel)
		}
	}
	// nodeName constraint
	if len(ps.NodeName) > 0 || len(matchingLabels) > 0 {
		allow, err := o.checkPodsBindAccess(ctx, attr)
		if err != nil {
			return err
		}
		if !allow {
			switch {
			case len(ps.NodeName) > 0 && len(matchingLabels) == 0:
				return admission.NewForbidden(attr, fmt.Errorf("node selection by nodeName is prohibited by policy for your role"))
			case len(ps.NodeName) == 0 && len(matchingLabels) > 0:
				return admission.NewForbidden(attr, fmt.Errorf("node selection by label(s) %v is prohibited by policy for your role", matchingLabels))
			case len(ps.NodeName) > 0 && len(matchingLabels) > 0:
				return admission.NewForbidden(attr, fmt.Errorf("node selection by nodeName and label(s) %v is prohibited by policy for your role", matchingLabels))
			}
		}
	}
	return nil
}

func (o *podNodeConstraints) SetAuthorizer(a authorizer.Authorizer) {
	o.authorizer = a
}

func (o *podNodeConstraints) ValidateInitialization() error {
	if o.authorizer == nil {
		return fmt.Errorf("%s requires an authorizer", PluginName)
	}
	if o.nodeIdentifier == nil {
		return fmt.Errorf("%s requires a node identifier", PluginName)
	}
	return nil
}

// build LocalSubjectAccessReview struct to validate role via checkAccess
func (o *podNodeConstraints) checkPodsBindAccess(ctx context.Context, attr admission.Attributes) (bool, error) {
	authzAttr := authorizer.AttributesRecord{
		User:            attr.GetUserInfo(),
		Verb:            "create",
		Namespace:       attr.GetNamespace(),
		Resource:        "pods",
		Subresource:     "binding",
		APIGroup:        coreapi.GroupName,
		ResourceRequest: true,
	}
	if attr.GetResource().GroupResource() == coreapi.Resource("pods") {
		authzAttr.Name = attr.GetName()
	}
	authorized, _, err := o.authorizer.Authorize(ctx, authzAttr)
	return authorized == authorizer.DecisionAllow, err
}

func (o *podNodeConstraints) isNodeSelfTargetWithMirrorPod(attr admission.Attributes, nodeName string) bool {
	// make sure we are actually trying to target a node
	if len(nodeName) == 0 {
		return false
	}
	// this check specifically requires the object to be pod (unlike the other checks where we want any pod spec)
	pod, ok := attr.GetObject().(*coreapi.Pod)
	if !ok {
		return false
	}
	// note that anyone can create a mirror pod, but they are not privileged in any way
	// they are actually highly constrained since they cannot reference secrets
	// nodes can only create and delete them, and they will delete any "orphaned" mirror pods
	if _, isMirrorPod := pod.Annotations[coreapi.MirrorPodAnnotationKey]; !isMirrorPod {
		return false
	}
	// we are targeting a node with a mirror pod
	// confirm the user is a node that is targeting itself
	actualNodeName, isNode := o.nodeIdentifier.NodeIdentity(attr.GetUserInfo())
	return isNode && actualNodeName == nodeName
}
