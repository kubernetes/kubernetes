package namespaceconditions

import (
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apiserver/pkg/admission"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	corev1lister "k8s.io/client-go/listers/core/v1"
)

// this is a list of namespaces with special meaning.  The kube ones are here in particular because
// we don't control their creation or labeling on their creation
var runLevelZeroNamespaces = sets.NewString("default", "kube-system", "kube-public")
var runLevelOneNamespaces = sets.NewString("openshift-node", "openshift-infra", "openshift")

func init() {
	runLevelOneNamespaces.Insert(runLevelZeroNamespaces.List()...)
}

// NamespaceLabelConditions provides a decorator that can delegate and conditionally add label conditions
type NamespaceLabelConditions struct {
	NamespaceClient corev1client.NamespacesGetter
	NamespaceLister corev1lister.NamespaceLister

	SkipLevelZeroNames sets.String
	SkipLevelOneNames  sets.String
}

func (d *NamespaceLabelConditions) WithNamespaceLabelConditions(admissionPlugin admission.Interface, name string) admission.Interface {
	switch {
	case d.SkipLevelOneNames.Has(name):
		// return a decorated admission plugin that skips runlevel 0 and 1 namespaces based on name (for known values) and
		// label.
		return &pluginHandlerWithNamespaceNameConditions{
			admissionPlugin: &pluginHandlerWithNamespaceLabelConditions{
				admissionPlugin:   admissionPlugin,
				namespaceClient:   d.NamespaceClient,
				namespaceLister:   d.NamespaceLister,
				namespaceSelector: skipRunLevelOneSelector,
			},
			namespacesToExclude: runLevelOneNamespaces,
		}

	case d.SkipLevelZeroNames.Has(name):
		// return a decorated admission plugin that skips runlevel 0 namespaces based on name (for known values) and
		// label.
		return &pluginHandlerWithNamespaceNameConditions{
			admissionPlugin: &pluginHandlerWithNamespaceLabelConditions{
				admissionPlugin:   admissionPlugin,
				namespaceClient:   d.NamespaceClient,
				namespaceLister:   d.NamespaceLister,
				namespaceSelector: skipRunLevelZeroSelector,
			},
			namespacesToExclude: runLevelZeroNamespaces,
		}

	default:
		return admissionPlugin
	}
}

// NamespaceLabelSelector provides a decorator that delegates
type NamespaceLabelSelector struct {
	namespaceClient corev1client.NamespacesGetter
	namespaceLister corev1lister.NamespaceLister

	admissionPluginNamesToDecorate sets.String
	namespaceLabelSelector         labels.Selector
}

func NewConditionalAdmissionPlugins(nsClient corev1client.NamespacesGetter, nsLister corev1lister.NamespaceLister, nsSelector labels.Selector, admissionPluginNames ...string) *NamespaceLabelSelector {
	return &NamespaceLabelSelector{
		namespaceClient:                nsClient,
		namespaceLister:                nsLister,
		admissionPluginNamesToDecorate: sets.NewString(admissionPluginNames...),
		namespaceLabelSelector:         nsSelector,
	}
}

func (d *NamespaceLabelSelector) WithNamespaceLabelSelector(admissionPlugin admission.Interface, name string) admission.Interface {
	if !d.admissionPluginNamesToDecorate.Has(name) {
		return admissionPlugin
	}

	return &pluginHandlerWithNamespaceLabelConditions{
		admissionPlugin:   admissionPlugin,
		namespaceClient:   d.namespaceClient,
		namespaceLister:   d.namespaceLister,
		namespaceSelector: d.namespaceLabelSelector,
	}
}
