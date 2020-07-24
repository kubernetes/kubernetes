package runonceduration

import (
	"context"
	"errors"
	"fmt"
	"io"
	"strconv"

	"k8s.io/klog"

	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	kapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/integer"

	"github.com/openshift/library-go/pkg/config/helpers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration"
	v1 "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/runonceduration/validation"
)

func Register(plugins *admission.Plugins) {
	plugins.Register("autoscaling.openshift.io/RunOnceDuration",
		func(config io.Reader) (admission.Interface, error) {
			pluginConfig, err := readConfig(config)
			if err != nil {
				return nil, err
			}
			if pluginConfig == nil {
				klog.Infof("Admission plugin %q is not configured so it will be disabled.", "autoscaling.openshift.io/RunOnceDuration")
				return nil, nil
			}
			return NewRunOnceDuration(pluginConfig), nil
		})
}

func readConfig(reader io.Reader) (*runonceduration.RunOnceDurationConfig, error) {
	obj, err := helpers.ReadYAMLToInternal(reader, runonceduration.Install, v1.Install)
	if err != nil {
		return nil, err
	}
	if obj == nil {
		return nil, nil
	}
	config, ok := obj.(*runonceduration.RunOnceDurationConfig)
	if !ok {
		return nil, fmt.Errorf("unexpected config object %#v", obj)
	}
	errs := validation.ValidateRunOnceDurationConfig(config)
	if len(errs) > 0 {
		return nil, errs.ToAggregate()
	}
	return config, nil
}

// NewRunOnceDuration creates a new RunOnceDuration admission plugin
func NewRunOnceDuration(config *runonceduration.RunOnceDurationConfig) admission.Interface {
	return &runOnceDuration{
		Handler: admission.NewHandler(admission.Create),
		config:  config,
	}
}

type runOnceDuration struct {
	*admission.Handler
	config   *runonceduration.RunOnceDurationConfig
	nsLister corev1listers.NamespaceLister
}

var _ = initializer.WantsExternalKubeInformerFactory(&runOnceDuration{})

func (a *runOnceDuration) Admit(ctx context.Context, attributes admission.Attributes, _ admission.ObjectInterfaces) error {
	switch {
	case a.config == nil,
		attributes.GetResource().GroupResource() != kapi.Resource("pods"),
		len(attributes.GetSubresource()) > 0:
		return nil
	}
	pod, ok := attributes.GetObject().(*kapi.Pod)
	if !ok {
		return admission.NewForbidden(attributes, fmt.Errorf("unexpected object: %#v", attributes.GetObject()))
	}

	// Only update pods with a restart policy of Never or OnFailure
	switch pod.Spec.RestartPolicy {
	case kapi.RestartPolicyNever,
		kapi.RestartPolicyOnFailure:
		// continue
	default:
		return nil
	}

	appliedProjectLimit, err := a.applyProjectAnnotationLimit(attributes.GetNamespace(), pod)
	if err != nil {
		return admission.NewForbidden(attributes, err)
	}

	if !appliedProjectLimit && a.config.ActiveDeadlineSecondsOverride != nil {
		pod.Spec.ActiveDeadlineSeconds = int64MinP(a.config.ActiveDeadlineSecondsOverride, pod.Spec.ActiveDeadlineSeconds)
	}
	return nil
}

func (a *runOnceDuration) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	a.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
}

func (a *runOnceDuration) ValidateInitialization() error {
	if a.nsLister == nil {
		return errors.New("autoscaling.openshift.io/RunOnceDuration plugin requires a namespace listers")
	}
	return nil
}

func (a *runOnceDuration) applyProjectAnnotationLimit(namespace string, pod *kapi.Pod) (bool, error) {
	ns, err := a.nsLister.Get(namespace)
	if err != nil {
		return false, fmt.Errorf("error looking up pod namespace: %v", err)
	}
	if ns.Annotations == nil {
		return false, nil
	}
	limit, hasLimit := ns.Annotations[runonceduration.ActiveDeadlineSecondsLimitAnnotation]
	if !hasLimit {
		return false, nil
	}
	limitInt64, err := strconv.ParseInt(limit, 10, 64)
	if err != nil {
		return false, fmt.Errorf("cannot parse the ActiveDeadlineSeconds limit (%s) for project %s: %v", limit, ns.Name, err)
	}
	pod.Spec.ActiveDeadlineSeconds = int64MinP(&limitInt64, pod.Spec.ActiveDeadlineSeconds)
	return true, nil
}

func int64MinP(a, b *int64) *int64 {
	switch {
	case a == nil:
		return b
	case b == nil:
		return a
	default:
		c := integer.Int64Min(*a, *b)
		return &c
	}
}
