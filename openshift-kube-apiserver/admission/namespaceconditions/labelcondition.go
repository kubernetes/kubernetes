package namespaceconditions

import (
	"context"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1lister "k8s.io/client-go/listers/core/v1"
)

const runLevelLabel = "openshift.io/run-level"

var (
	skipRunLevelZeroSelector labels.Selector
	skipRunLevelOneSelector  labels.Selector
)

func init() {
	var err error
	skipRunLevelZeroSelector, err = labels.Parse(runLevelLabel + " notin ( 0 )")
	if err != nil {
		panic(err)
	}
	skipRunLevelOneSelector, err = labels.Parse(runLevelLabel + " notin ( 0,1 )")
	if err != nil {
		panic(err)
	}
}

// pluginHandlerWithNamespaceLabelConditions wraps an admission plugin in a conditional skip based on namespace labels
type pluginHandlerWithNamespaceLabelConditions struct {
	admissionPlugin   admission.Interface
	namespaceClient   corev1client.NamespacesGetter
	namespaceLister   corev1lister.NamespaceLister
	namespaceSelector labels.Selector
}

var _ admission.ValidationInterface = &pluginHandlerWithNamespaceLabelConditions{}
var _ admission.MutationInterface = &pluginHandlerWithNamespaceLabelConditions{}

func (p pluginHandlerWithNamespaceLabelConditions) Handles(operation admission.Operation) bool {
	return p.admissionPlugin.Handles(operation)
}

// Admit performs a mutating admission control check and emit metrics.
func (p pluginHandlerWithNamespaceLabelConditions) Admit(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.shouldRunAdmission(a) {
		return nil
	}

	mutatingHandler, ok := p.admissionPlugin.(admission.MutationInterface)
	if !ok {
		return nil
	}
	return mutatingHandler.Admit(ctx, a, o)
}

// Validate performs a non-mutating admission control check and emits metrics.
func (p pluginHandlerWithNamespaceLabelConditions) Validate(ctx context.Context, a admission.Attributes, o admission.ObjectInterfaces) error {
	if !p.shouldRunAdmission(a) {
		return nil
	}

	validatingHandler, ok := p.admissionPlugin.(admission.ValidationInterface)
	if !ok {
		return nil
	}
	return validatingHandler.Validate(ctx, a, o)
}

// MatchNamespaceSelector decideds whether the request matches the
// namespaceSelctor of the webhook. Only when they match, the webhook is called.
func (p pluginHandlerWithNamespaceLabelConditions) shouldRunAdmission(attr admission.Attributes) bool {
	namespaceName := attr.GetNamespace()
	if len(namespaceName) == 0 && attr.GetResource().Resource != "namespaces" {
		// cluster scoped resources always run admission
		return true
	}
	namespaceLabels, err := p.getNamespaceLabels(attr)
	if err != nil {
		// default to running the hook so we don't leak namespace existence information
		return true
	}
	// TODO: adding an LRU cache to cache the match decision
	return p.namespaceSelector.Matches(labels.Set(namespaceLabels))
}

// getNamespaceLabels gets the labels of the namespace related to the attr.
func (p pluginHandlerWithNamespaceLabelConditions) getNamespaceLabels(attr admission.Attributes) (map[string]string, error) {
	// If the request itself is creating or updating a namespace, then get the
	// labels from attr.Object, because namespaceLister doesn't have the latest
	// namespace yet.
	//
	// However, if the request is deleting a namespace, then get the label from
	// the namespace in the namespaceLister, because a delete request is not
	// going to change the object, and attr.Object will be a DeleteOptions
	// rather than a namespace object.
	if attr.GetResource().Resource == "namespaces" &&
		len(attr.GetSubresource()) == 0 &&
		(attr.GetOperation() == admission.Create || attr.GetOperation() == admission.Update) {
		accessor, err := meta.Accessor(attr.GetObject())
		if err != nil {
			return nil, err
		}
		return accessor.GetLabels(), nil
	}

	namespaceName := attr.GetNamespace()
	namespace, err := p.namespaceLister.Get(namespaceName)
	if err != nil && !apierrors.IsNotFound(err) {
		return nil, err
	}
	if apierrors.IsNotFound(err) {
		// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
		namespace, err = p.namespaceClient.Namespaces().Get(namespaceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
	}
	return namespace.Labels, nil
}
