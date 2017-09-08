package defaultpodlabels

import (
	"fmt"
	"io"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/util/yaml"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	informers "k8s.io/kubernetes/pkg/client/informers/informers_generated/internalversion"
	corelisters "k8s.io/kubernetes/pkg/client/listers/core/internalversion"
	kubeapiserveradmission "k8s.io/kubernetes/pkg/kubeapiserver/admission"
)

type defaultPodLabels struct {
	*admission.Handler
	Config          *DefaultPodLabelsConfig
	client          internalclientset.Interface
	namespaceLister corelisters.NamespaceLister
}

func Register(plugins *admission.Plugins) {
	plugins.Register("DefaultPodLabels", func(config io.Reader) (admission.Interface, error) {
		return NewDefaultPodLabels(config)
	})
}

var _ = kubeapiserveradmission.WantsInternalKubeInformerFactory(&defaultPodLabels{})
var _ = kubeapiserveradmission.WantsInternalKubeClientSet(&defaultPodLabels{})

func (dpl *defaultPodLabels) handlePodLabel(pod *api.Pod, label LabelConfigItem) error {
	if _, ok := pod.Labels[label.Name]; ok {
		return nil
	}

	if !label.SkipNamespace {
		namespace, err := dpl.namespaceLister.Get(pod.Namespace)
		if err != nil {
			return fmt.Errorf("Error getting namespace '%s': %s", pod.Namespace, err)
		}

		if val, ok := namespace.Labels[label.Name]; ok {
			pod.Labels[label.Name] = val
			return nil
		}
	}

	// fall through to default
	if len(label.Default) > 0 {
		pod.Labels[label.Name] = label.Default
	}

	return nil
}

func (dpl *defaultPodLabels) Admit(attributes admission.Attributes) error {
	if len(dpl.Config.Labels) == 0 {
		return nil
	}

	// Ignore all calls to subresources or resources other than pods.
	if len(attributes.GetSubresource()) != 0 || attributes.GetResource().GroupResource() != api.Resource("pods") {
		return nil
	}

	// we need to wait for our caches to warm
	if !dpl.WaitForReady() {
		return admission.NewForbidden(attributes, fmt.Errorf("not yet ready to handle request"))
	}

	pod, ok := attributes.GetObject().(*api.Pod)

	if !ok {
		return apierrors.NewBadRequest("Resource was marked with kind Pod but was unable to be converted")
	}

	for i := range dpl.Config.Labels {
		if err := dpl.handlePodLabel(pod, dpl.Config.Labels[i]); err != nil {
			return err
		}
	}

	return nil
}

func NewDefaultPodLabels(config io.Reader) (admission.Interface, error) {
	lc := DefaultPodLabelsConfig{}

	d := yaml.NewYAMLOrJSONDecoder(config, 4096)
	err := d.Decode(&lc)
	if err != nil {
		return nil, err
	}

	return &defaultPodLabels{
		Handler: admission.NewHandler(admission.Create, admission.Update),
		Config:  &lc,
	}, nil
}

func (dpl *defaultPodLabels) SetInternalKubeClientSet(client internalclientset.Interface) {
	dpl.client = client
}

func (dpl *defaultPodLabels) SetInternalKubeInformerFactory(f informers.SharedInformerFactory) {
	namespaceInformer := f.Core().InternalVersion().Namespaces()
	dpl.namespaceLister = namespaceInformer.Lister()
	dpl.SetReadyFunc(namespaceInformer.Informer().HasSynced)
}

func (dpl *defaultPodLabels) Validate() error {
	if dpl.namespaceLister == nil {
		return fmt.Errorf("missing namespaceLister")
	}
	if dpl.client == nil {
		return fmt.Errorf("missing client")
	}
	return nil
}

type LabelConfigItem struct {
	Name          string `json:"name"`
	SkipNamespace bool   `json:"skipNamespace"`
	Default       string `json:"default"`
}

type DefaultPodLabelsConfig struct {
	Labels []LabelConfigItem `json:"labels"`
}
