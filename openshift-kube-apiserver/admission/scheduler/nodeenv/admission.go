package nodeenv

import (
	"context"
	"errors"
	"fmt"
	"io"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	coreapi "k8s.io/kubernetes/pkg/apis/core"

	projectv1 "github.com/openshift/api/project/v1"
	"github.com/openshift/apiserver-library-go/pkg/labelselector"
)

func Register(plugins *admission.Plugins) {
	plugins.Register("scheduling.openshift.io/OriginPodNodeEnvironment",
		func(config io.Reader) (admission.Interface, error) {
			return NewPodNodeEnvironment()
		})
}

const (
	timeToWaitForCacheSync  = 10 * time.Second
	kubeProjectNodeSelector = "scheduler.alpha.kubernetes.io/node-selector"
)

// podNodeEnvironment is an implementation of admission.MutationInterface.
type podNodeEnvironment struct {
	*admission.Handler
	nsLister         corev1listers.NamespaceLister
	nsListerSynced   func() bool
	nodeLister       corev1listers.NodeLister
	nodeListerSynced func() bool
	// TODO this should become a piece of config passed to the admission plugin
	defaultNodeSelector string
}

var _ = initializer.WantsExternalKubeInformerFactory(&podNodeEnvironment{})
var _ = WantsDefaultNodeSelector(&podNodeEnvironment{})
var _ = admission.ValidationInterface(&podNodeEnvironment{})
var _ = admission.MutationInterface(&podNodeEnvironment{})

// Admit enforces that pod and its project node label selectors matches at least a node in the cluster.
func (p *podNodeEnvironment) admit(ctx context.Context, a admission.Attributes, mutationAllowed bool) (err error) {
	resource := a.GetResource().GroupResource()
	if resource != corev1.Resource("pods") {
		return nil
	}
	if a.GetSubresource() != "" {
		// only run the checks below on pods proper and not subresources
		return nil
	}

	obj := a.GetObject()
	pod, ok := obj.(*coreapi.Pod)
	if !ok {
		return nil
	}

	name := pod.Name

	if !p.waitForSyncedStore(time.After(timeToWaitForCacheSync)) {
		return admission.NewForbidden(a, errors.New("scheduling.openshift.io/OriginPodNodeEnvironment: caches not synchronized"))
	}
	namespace, err := p.nsLister.Get(a.GetNamespace())
	if err != nil {
		return apierrors.NewForbidden(resource, name, err)
	}

	// If scheduler.alpha.kubernetes.io/node-selector is set on the namespace,
	// do not process the pod further.
	if _, ok := namespace.ObjectMeta.Annotations[kubeProjectNodeSelector]; ok {
		return nil
	}

	selector := p.defaultNodeSelector
	if projectNodeSelector, ok := namespace.ObjectMeta.Annotations[projectv1.ProjectNodeSelector]; ok {
		selector = projectNodeSelector
	}
	// we might consider in the future to allow advanced syntax selectors and use labels.Parse here instead
	projectNodeSelector, err := labelselector.Parse(selector)
	if err != nil {
		return err
	}

	if labelselector.Conflicts(projectNodeSelector, pod.Spec.NodeSelector) {
		return apierrors.NewForbidden(resource, name, fmt.Errorf("pod node label selector conflicts with its project node label selector"))
	}

	if !mutationAllowed && len(labelselector.Merge(projectNodeSelector, pod.Spec.NodeSelector)) != len(pod.Spec.NodeSelector) {
		// no conflict, different size => pod.Spec.NodeSelector does not contain projectNodeSelector
		return apierrors.NewForbidden(resource, name, fmt.Errorf("pod node label selector does not extend project node label selector"))
	}

	if len(pod.Spec.NodeName) > 0 && len(projectNodeSelector) > 0 {
		node, err := p.nodeLister.Get(pod.Spec.NodeName)
		if err != nil {
			return apierrors.NewForbidden(resource, name, fmt.Errorf("cannot validate project node label selector: %v", err))
		}
		projectNodeSelectorAdvanced, err := labels.Parse(selector)
		if err != nil {
			return err
		}
		if !projectNodeSelectorAdvanced.Matches(labels.Set(node.Labels)) {
			return apierrors.NewForbidden(resource, name, fmt.Errorf("pod node name conflicts with project node label selector"))
		}
	}

	// modify pod node selector = project node selector + current pod node selector
	pod.Spec.NodeSelector = labelselector.Merge(projectNodeSelector, pod.Spec.NodeSelector)

	return nil
}

func (p *podNodeEnvironment) Admit(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) (err error) {
	return p.admit(ctx, a, true)
}

func (p *podNodeEnvironment) Validate(ctx context.Context, a admission.Attributes, _ admission.ObjectInterfaces) (err error) {
	return p.admit(ctx, a, false)
}

func (p *podNodeEnvironment) SetDefaultNodeSelector(in string) {
	p.defaultNodeSelector = in
}

func (p *podNodeEnvironment) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	p.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
	p.nsListerSynced = kubeInformers.Core().V1().Namespaces().Informer().HasSynced
	p.nodeLister = kubeInformers.Core().V1().Nodes().Lister()
	p.nodeListerSynced = kubeInformers.Core().V1().Nodes().Informer().HasSynced
}

func (p *podNodeEnvironment) waitForSyncedStore(timeout <-chan time.Time) bool {
	for !p.nsListerSynced() || !p.nodeListerSynced() {
		select {
		case <-time.After(100 * time.Millisecond):
		case <-timeout:
			return p.nsListerSynced() && p.nodeListerSynced()
		}
	}

	return true
}

func (p *podNodeEnvironment) ValidateInitialization() error {
	if p.nsLister == nil {
		return fmt.Errorf("project node environment plugin needs a namespace lister")
	}
	if p.nsListerSynced == nil {
		return fmt.Errorf("project node environment plugin needs a namespace lister synced")
	}
	if p.nodeLister == nil {
		return fmt.Errorf("project node environment plugin needs a node lister")
	}
	if p.nodeListerSynced == nil {
		return fmt.Errorf("project node environment plugin needs a node lister synced")
	}
	return nil
}

func NewPodNodeEnvironment() (admission.Interface, error) {
	return &podNodeEnvironment{
		Handler: admission.NewHandler(admission.Create),
	}, nil
}
