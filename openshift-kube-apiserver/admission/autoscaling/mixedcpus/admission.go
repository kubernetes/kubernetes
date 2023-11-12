package mixedcpus

import (
	"context"
	"fmt"
	"io"
	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/managementcpusoverride"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
)

const (
	PluginName       = "autoscaling.openshift.io/MixedCPUs"
	annotationEnable = "enable"
	// containerResourceRequestName is the name of the resource that should be specified under the container's request in the pod spec
	containerResourceRequestName = "workload.openshift.io/enable-shared-cpus"
	// runtimeAnnotationPrefix is the prefix for the annotation that is expected by the runtime
	runtimeAnnotationPrefix = "cpu-shared.crio.io"
	// namespaceAllowedAnnotation contains the namespace allowed annotation key
	namespaceAllowedAnnotation = "workload.mixedcpus.openshift.io/allowed"
)

var _ = initializer.WantsExternalKubeClientSet(&mixedCPUsMutation{})
var _ = initializer.WantsExternalKubeInformerFactory(&mixedCPUsMutation{})
var _ = admission.MutationInterface(&mixedCPUsMutation{})

type mixedCPUsMutation struct {
	*admission.Handler
	client          kubernetes.Interface
	podLister       corev1listers.PodLister
	podListerSynced func() bool
	nsLister        corev1listers.NamespaceLister
	nsListerSynced  func() bool
}

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			return &mixedCPUsMutation{
				Handler: admission.NewHandler(admission.Create),
			}, nil
		})
}

// SetExternalKubeClientSet implements the WantsExternalKubeClientSet interface.
func (s *mixedCPUsMutation) SetExternalKubeClientSet(client kubernetes.Interface) {
	s.client = client
}

func (s *mixedCPUsMutation) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	s.podLister = kubeInformers.Core().V1().Pods().Lister()
	s.podListerSynced = kubeInformers.Core().V1().Pods().Informer().HasSynced
	s.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
	s.nsListerSynced = kubeInformers.Core().V1().Namespaces().Informer().HasSynced
}

func (s *mixedCPUsMutation) ValidateInitialization() error {
	if s.client == nil {
		return fmt.Errorf("%s plugin needs a kubernetes client", PluginName)
	}
	if s.podLister == nil {
		return fmt.Errorf("%s did not get a pod lister", PluginName)
	}
	if s.podListerSynced == nil {
		return fmt.Errorf("%s plugin needs a pod lister synced", PluginName)
	}
	if s.nsLister == nil {
		return fmt.Errorf("%s did not get a namespace lister", PluginName)
	}
	if s.nsListerSynced == nil {
		return fmt.Errorf("%s plugin needs a namespace lister synced", PluginName)
	}
	return nil
}

func (s *mixedCPUsMutation) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != coreapi.Resource("pods") || attr.GetSubresource() != "" {
		return nil
	}

	pod, ok := attr.GetObject().(*coreapi.Pod)
	if !ok {
		return admission.NewForbidden(attr, fmt.Errorf("%s unexpected object: %#v", attr.GetObject(), PluginName))
	}

	for i := 0; i < len(pod.Spec.Containers); i++ {
		cnt := &pod.Spec.Containers[i]
		requested, v := isContainerRequestForSharedCPUs(cnt)
		if !requested {
			continue
		}
		ns, err := s.getPodNs(ctx, pod.Namespace)
		if err != nil {
			return fmt.Errorf("%s %w", PluginName, err)
		}
		_, found := ns.Annotations[namespaceAllowedAnnotation]
		if !found {
			return admission.NewForbidden(attr, fmt.Errorf("%s pod %s namespace %s is not allowed for %s resource request", PluginName, pod.Name, pod.Namespace, containerResourceRequestName))
		}
		if !managementcpusoverride.IsGuaranteed(pod.Spec.Containers) {
			return admission.NewForbidden(attr, fmt.Errorf("%s %s/%s requests for %q resource but pod is not Guaranteed QoS class", PluginName, pod.Name, cnt.Name, containerResourceRequestName))
		}
		if v.Value() > 1 {
			return admission.NewForbidden(attr, fmt.Errorf("%s %s/%s more than a single %q resource is forbiden, please set the request to 1 or remove it", PluginName, pod.Name, cnt.Name, containerResourceRequestName))
		}
		addRuntimeAnnotation(pod, cnt.Name)
	}
	return nil
}

func (s *mixedCPUsMutation) getPodNs(ctx context.Context, nsName string) (*v1.Namespace, error) {
	ns, err := s.nsLister.Get(nsName)
	if err != nil {
		if !errors.IsNotFound(err) {
			return nil, fmt.Errorf("%s failed to retrieve namespace %q from lister; %w", PluginName, nsName, err)
		}
		// cache didn't update fast enough
		ns, err = s.client.CoreV1().Namespaces().Get(ctx, nsName, metav1.GetOptions{})
		if err != nil {
			return nil, fmt.Errorf("%s failed to retrieve namespace %q from api server; %w", PluginName, nsName, err)
		}
	}
	return ns, nil
}

func isContainerRequestForSharedCPUs(container *coreapi.Container) (bool, resource.Quantity) {
	for rName, quan := range container.Resources.Requests {
		if rName == containerResourceRequestName {
			return true, quan
		}
	}
	return false, resource.Quantity{}
}

func addRuntimeAnnotation(pod *coreapi.Pod, cntName string) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}
	pod.Annotations[getRuntimeAnnotationName(cntName)] = annotationEnable
}

func getRuntimeAnnotationName(cntName string) string {
	return fmt.Sprintf("%s/%s", runtimeAnnotationPrefix, cntName)
}
