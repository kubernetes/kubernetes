package performantsecuritypolicy

import (
	"context"
	"fmt"
	"io"

	openshiftfeatures "github.com/openshift/api/features"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/audit"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/informers"
	corev1listers "k8s.io/client-go/listers/core/v1"
	"k8s.io/component-base/featuregate"
	"k8s.io/klog/v2"
	kapi "k8s.io/kubernetes/pkg/apis/core"
)

const (
	PluginName               = "storage.openshift.io/PerformantSecurityPolicy"
	fsGroupChangePolicyLabel = "storage.openshift.io/fsgroup-change-policy"
	selinuxChangePolicyLabel = "storage.openshift.io/selinux-change-policy"

	warningFormat = "found %s label with invalid %s: %s on %s namespace"
)

var (
	_ = initializer.WantsExternalKubeInformerFactory(&performantSecurityPolicy{})
	_ = admission.MutationInterface(&performantSecurityPolicy{})
	_ = initializer.WantsFeatures(&performantSecurityPolicy{})

	fsGroupPolicyPodAuditLabel = fmt.Sprintf("%s-pod", fsGroupChangePolicyLabel)
	selinuxPolicyPodAuditLabel = fmt.Sprintf("%s-pod", selinuxChangePolicyLabel)
)

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			return &performantSecurityPolicy{
				Handler: admission.NewHandler(admission.Create),
			}, nil
		})
}

// performantSecurityPolicy checks and applies if a default FSGroupChangePolicy and SELinuxChangePolicy
// should be applied to the pod.
type performantSecurityPolicy struct {
	*admission.Handler
	storagePerformantSecurityPolicyFeatureEnabled bool
	nsLister                                      corev1listers.NamespaceLister
}

// SetExternalKubeInformerFactory registers an informer
func (c *performantSecurityPolicy) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	c.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
	c.SetReadyFunc(func() bool {
		return kubeInformers.Core().V1().Namespaces().Informer().HasSynced()
	})
}

func (c *performantSecurityPolicy) InspectFeatureGates(featureGates featuregate.FeatureGate) {
	c.storagePerformantSecurityPolicyFeatureEnabled = featureGates.Enabled(featuregate.Feature(openshiftfeatures.FeatureGateStoragePerformantSecurityPolicy))
}

func (c *performantSecurityPolicy) ValidateInitialization() error {
	if c.nsLister == nil {
		return fmt.Errorf("%s plugin needs a namespace lister", PluginName)
	}
	return nil
}

func (c *performantSecurityPolicy) Admit(ctx context.Context, attributes admission.Attributes, _ admission.ObjectInterfaces) error {
	if !c.storagePerformantSecurityPolicyFeatureEnabled {
		return nil
	}

	if !c.WaitForReady() {
		return admission.NewForbidden(attributes, fmt.Errorf("not yet ready to handle request"))
	}

	if attributes.GetResource().GroupResource() != kapi.Resource("pods") ||
		len(attributes.GetSubresource()) > 0 {
		return nil
	}

	pod, ok := attributes.GetObject().(*kapi.Pod)
	if !ok {
		return admission.NewForbidden(attributes, fmt.Errorf("unexpected object: %#v", attributes.GetObject()))
	}

	ns, err := c.nsLister.Get(pod.Namespace)
	if err != nil {
		return fmt.Errorf("error listing pod namespace: %v", err)
	}
	podNameKey := fmt.Sprintf("%s/%s", attributes.GetName(), attributes.GetNamespace())

	currentFSGroupChangePolicy := extractCurrentFSGroupChangePolicy(pod)
	if currentFSGroupChangePolicy == nil {
		currentFSGroupChangePolicy = getDefaultFSGroupChangePolicy(ctx, ns)
		if currentFSGroupChangePolicy != nil {
			klog.V(4).Infof("Setting default FSGroupChangePolicy %s for pod %s", *currentFSGroupChangePolicy, podNameKey)
			audit.AddAuditAnnotations(ctx, fsGroupChangePolicyLabel, string(*currentFSGroupChangePolicy), fsGroupPolicyPodAuditLabel, podNameKey)
			if pod.Spec.SecurityContext != nil {
				pod.Spec.SecurityContext.FSGroupChangePolicy = currentFSGroupChangePolicy
			} else {
				pod.Spec.SecurityContext = &kapi.PodSecurityContext{
					FSGroupChangePolicy: currentFSGroupChangePolicy,
				}
			}
		}
	}

	currentSELinuxChangePolicy := extractCurrentSELinuxChangePolicy(pod)
	if currentSELinuxChangePolicy == nil {
		currentSELinuxChangePolicy = getDefaultSELinuxChangePolicy(ctx, ns)
		if currentSELinuxChangePolicy != nil {
			klog.V(4).Infof("Setting default SELinuxChangePolicy %s for pod %s", *currentSELinuxChangePolicy, podNameKey)
			audit.AddAuditAnnotations(ctx, selinuxChangePolicyLabel, string(*currentSELinuxChangePolicy), selinuxPolicyPodAuditLabel, podNameKey)
			if pod.Spec.SecurityContext != nil {
				pod.Spec.SecurityContext.SELinuxChangePolicy = currentSELinuxChangePolicy
			} else {
				pod.Spec.SecurityContext = &kapi.PodSecurityContext{
					SELinuxChangePolicy: currentSELinuxChangePolicy,
				}
			}
		}
	}
	return nil
}

func extractCurrentSELinuxChangePolicy(pod *kapi.Pod) *kapi.PodSELinuxChangePolicy {
	if pod.Spec.SecurityContext != nil {
		return pod.Spec.SecurityContext.SELinuxChangePolicy
	}

	return nil
}

func extractCurrentFSGroupChangePolicy(pod *kapi.Pod) *kapi.PodFSGroupChangePolicy {
	if pod.Spec.SecurityContext != nil {
		return pod.Spec.SecurityContext.FSGroupChangePolicy
	}
	return nil
}

func getDefaultFSGroupChangePolicy(ctx context.Context, ns *corev1.Namespace) *kapi.PodFSGroupChangePolicy {
	fsGroupPolicy, ok := ns.Labels[fsGroupChangePolicyLabel]
	if !ok {
		return nil
	}
	policy := kapi.PodFSGroupChangePolicy(fsGroupPolicy)

	if policy == kapi.FSGroupChangeOnRootMismatch || policy == kapi.FSGroupChangeAlways {
		return &policy
	}
	klog.Warningf("found %s label with invalid fsGroupPolicy: %s", fsGroupChangePolicyLabel, fsGroupPolicy)
	warning.AddWarning(ctx, "", fmt.Sprintf(warningFormat, fsGroupChangePolicyLabel, "fsGroupPolicy", fsGroupPolicy, ns.Name))
	return nil
}

func getDefaultSELinuxChangePolicy(ctx context.Context, ns *corev1.Namespace) *kapi.PodSELinuxChangePolicy {
	selinuxChangePolicy, ok := ns.Labels[selinuxChangePolicyLabel]
	if !ok {
		return nil
	}

	policy := kapi.PodSELinuxChangePolicy(selinuxChangePolicy)

	if policy == kapi.SELinuxChangePolicyMountOption || policy == kapi.SELinuxChangePolicyRecursive {
		return &policy
	}

	klog.Warningf("found %s label with invalid selinuxPolicy: %s", selinuxChangePolicyLabel, selinuxChangePolicy)
	warning.AddWarning(ctx, "", fmt.Sprintf(warningFormat, selinuxChangePolicyLabel, "selinuxPolicy", selinuxChangePolicy, ns.Name))
	return nil

}
