package clusterresourceoverride

import (
	"context"
	"fmt"
	"io"
	"strings"

	"github.com/openshift/library-go/pkg/config/helpers"
	v1 "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/v1"

	"k8s.io/klog/v2"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/plugin/pkg/admission/limitranger"

	api "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride"
	"k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/validation"
)

const (
	clusterResourceOverrideAnnotation = "autoscaling.openshift.io/cluster-resource-override-enabled"
	cpuBaseScaleFactor                = 1000.0 / (1024.0 * 1024.0 * 1024.0) // 1000 milliCores per 1GiB
)

var (
	cpuFloor = resource.MustParse("1m")
	memFloor = resource.MustParse("1Mi")
)

func Register(plugins *admission.Plugins) {
	plugins.Register(api.PluginName,
		func(config io.Reader) (admission.Interface, error) {
			pluginConfig, err := ReadConfig(config)
			if err != nil {
				return nil, err
			}
			if pluginConfig == nil {
				klog.Infof("Admission plugin %q is not configured so it will be disabled.", api.PluginName)
				return nil, nil
			}
			return newClusterResourceOverride(pluginConfig)
		})
}

type internalConfig struct {
	limitCPUToMemoryRatio     float64
	cpuRequestToLimitRatio    float64
	memoryRequestToLimitRatio float64
}
type clusterResourceOverridePlugin struct {
	*admission.Handler
	config            *internalConfig
	nsLister          corev1listers.NamespaceLister
	LimitRanger       *limitranger.LimitRanger
	limitRangesLister corev1listers.LimitRangeLister
}

var _ = initializer.WantsExternalKubeInformerFactory(&clusterResourceOverridePlugin{})
var _ = initializer.WantsExternalKubeClientSet(&clusterResourceOverridePlugin{})
var _ = admission.MutationInterface(&clusterResourceOverridePlugin{})
var _ = admission.ValidationInterface(&clusterResourceOverridePlugin{})

// newClusterResourceOverride returns an admission controller for containers that
// configurably overrides container resource request/limits
func newClusterResourceOverride(config *api.ClusterResourceOverrideConfig) (admission.Interface, error) {
	klog.V(2).Infof("%s admission controller loaded with config: %v", api.PluginName, config)
	var internal *internalConfig
	if config != nil {
		internal = &internalConfig{
			limitCPUToMemoryRatio:     float64(config.LimitCPUToMemoryPercent) / 100,
			cpuRequestToLimitRatio:    float64(config.CPURequestToLimitPercent) / 100,
			memoryRequestToLimitRatio: float64(config.MemoryRequestToLimitPercent) / 100,
		}
	}

	limitRanger, err := limitranger.NewLimitRanger(nil)
	if err != nil {
		return nil, err
	}

	return &clusterResourceOverridePlugin{
		Handler:     admission.NewHandler(admission.Create),
		config:      internal,
		LimitRanger: limitRanger,
	}, nil
}

func (d *clusterResourceOverridePlugin) SetExternalKubeClientSet(c kubernetes.Interface) {
	d.LimitRanger.SetExternalKubeClientSet(c)
}

func (d *clusterResourceOverridePlugin) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	d.LimitRanger.SetExternalKubeInformerFactory(kubeInformers)
	d.limitRangesLister = kubeInformers.Core().V1().LimitRanges().Lister()
	d.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
}

func ReadConfig(configFile io.Reader) (*api.ClusterResourceOverrideConfig, error) {
	obj, err := helpers.ReadYAMLToInternal(configFile, api.Install, v1.Install)
	if err != nil {
		klog.V(5).Infof("%s error reading config: %v", api.PluginName, err)
		return nil, err
	}
	if obj == nil {
		return nil, nil
	}
	config, ok := obj.(*api.ClusterResourceOverrideConfig)
	if !ok {
		return nil, fmt.Errorf("unexpected config object: %#v", obj)
	}
	klog.V(5).Infof("%s config is: %v", api.PluginName, config)
	if errs := validation.Validate(config); len(errs) > 0 {
		return nil, errs.ToAggregate()
	}

	return config, nil
}

func (a *clusterResourceOverridePlugin) ValidateInitialization() error {
	if a.nsLister == nil {
		return fmt.Errorf("%s did not get a namespace lister", api.PluginName)
	}
	return a.LimitRanger.ValidateInitialization()
}

// this a real shame to be special cased.
var (
	forbiddenNames    = []string{"openshift", "kubernetes", "kube"}
	forbiddenPrefixes = []string{"openshift-", "kubernetes-", "kube-"}
)

func isExemptedNamespace(name string) bool {
	for _, s := range forbiddenNames {
		if name == s {
			return true
		}
	}
	for _, s := range forbiddenPrefixes {
		if strings.HasPrefix(name, s) {
			return true
		}
	}
	return false
}

func (a *clusterResourceOverridePlugin) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	return a.admit(ctx, attr, true, o)
}

func (a *clusterResourceOverridePlugin) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	return a.admit(ctx, attr, false, o)
}

// TODO this will need to update when we have pod requests/limits
func (a *clusterResourceOverridePlugin) admit(ctx context.Context, attr admission.Attributes, mutationAllowed bool, o admission.ObjectInterfaces) error {
	klog.V(6).Infof("%s admission controller is invoked", api.PluginName)
	if a.config == nil || attr.GetResource().GroupResource() != coreapi.Resource("pods") || attr.GetSubresource() != "" {
		return nil // not applicable
	}
	pod, ok := attr.GetObject().(*coreapi.Pod)
	if !ok {
		return admission.NewForbidden(attr, fmt.Errorf("unexpected object: %#v", attr.GetObject()))
	}
	klog.V(5).Infof("%s is looking at creating pod %s in project %s", api.PluginName, pod.Name, attr.GetNamespace())

	// allow annotations on project to override
	ns, err := a.nsLister.Get(attr.GetNamespace())
	if err != nil {
		klog.Warningf("%s got an error retrieving namespace: %v", api.PluginName, err)
		return admission.NewForbidden(attr, err) // this should not happen though
	}

	projectEnabledPlugin, exists := ns.Annotations[clusterResourceOverrideAnnotation]
	if exists && projectEnabledPlugin != "true" {
		klog.V(5).Infof("%s is disabled for project %s", api.PluginName, attr.GetNamespace())
		return nil // disabled for this project, do nothing
	}

	if isExemptedNamespace(ns.Name) {
		klog.V(5).Infof("%s is skipping exempted project %s", api.PluginName, attr.GetNamespace())
		return nil // project is exempted, do nothing
	}

	namespaceLimits := []*corev1.LimitRange{}

	if a.limitRangesLister != nil {
		limits, err := a.limitRangesLister.LimitRanges(attr.GetNamespace()).List(labels.Everything())
		if err != nil {
			return err
		}
		namespaceLimits = limits
	}

	// Don't mutate resource requirements below the namespace
	// limit minimums.
	nsCPUFloor := minResourceLimits(namespaceLimits, corev1.ResourceCPU)
	nsMemFloor := minResourceLimits(namespaceLimits, corev1.ResourceMemory)

	// Reuse LimitRanger logic to apply limit/req defaults from the project. Ignore validation
	// errors, assume that LimitRanger will run after this plugin to validate.
	klog.V(5).Infof("%s: initial pod limits are: %#v", api.PluginName, pod.Spec)
	if err := a.LimitRanger.Admit(ctx, attr, o); err != nil {
		klog.V(5).Infof("%s: error from LimitRanger: %#v", api.PluginName, err)
	}
	klog.V(5).Infof("%s: pod limits after LimitRanger: %#v", api.PluginName, pod.Spec)
	for i := range pod.Spec.InitContainers {
		if err := updateContainerResources(a.config, &pod.Spec.InitContainers[i], nsCPUFloor, nsMemFloor, mutationAllowed); err != nil {
			return admission.NewForbidden(attr, fmt.Errorf("spec.initContainers[%d].%v", i, err))
		}
	}
	for i := range pod.Spec.Containers {
		if err := updateContainerResources(a.config, &pod.Spec.Containers[i], nsCPUFloor, nsMemFloor, mutationAllowed); err != nil {
			return admission.NewForbidden(attr, fmt.Errorf("spec.containers[%d].%v", i, err))
		}
	}
	klog.V(5).Infof("%s: pod limits after overrides are: %#v", api.PluginName, pod.Spec)
	return nil
}

func updateContainerResources(config *internalConfig, container *coreapi.Container, nsCPUFloor, nsMemFloor *resource.Quantity, mutationAllowed bool) error {
	resources := container.Resources
	memLimit, memFound := resources.Limits[coreapi.ResourceMemory]
	if memFound && config.memoryRequestToLimitRatio != 0 {
		// memory is measured in whole bytes.
		// the plugin rounds down to the nearest MiB rather than bytes to improve ease of use for end-users.
		amount := memLimit.Value() * int64(config.memoryRequestToLimitRatio*100) / 100
		// TODO: move into resource.Quantity
		var mod int64
		switch memLimit.Format {
		case resource.BinarySI:
			mod = 1024 * 1024
		default:
			mod = 1000 * 1000
		}
		if rem := amount % mod; rem != 0 {
			amount = amount - rem
		}
		q := resource.NewQuantity(int64(amount), memLimit.Format)
		if memFloor.Cmp(*q) > 0 {
			clone := memFloor.DeepCopy()
			q = &clone
		}
		if nsMemFloor != nil && q.Cmp(*nsMemFloor) < 0 {
			klog.V(5).Infof("%s: %s pod limit %q below namespace limit; setting limit to %q", api.PluginName, corev1.ResourceMemory, q.String(), nsMemFloor.String())
			clone := nsMemFloor.DeepCopy()
			q = &clone
		}
		if err := applyQuantity(resources.Requests, corev1.ResourceMemory, *q, mutationAllowed); err != nil {
			return fmt.Errorf("resources.requests.%s %v", corev1.ResourceMemory, err)
		}
	}
	if memFound && config.limitCPUToMemoryRatio != 0 {
		amount := float64(memLimit.Value()) * config.limitCPUToMemoryRatio * cpuBaseScaleFactor
		q := resource.NewMilliQuantity(int64(amount), resource.DecimalSI)
		if cpuFloor.Cmp(*q) > 0 {
			clone := cpuFloor.DeepCopy()
			q = &clone
		}
		if nsCPUFloor != nil && q.Cmp(*nsCPUFloor) < 0 {
			klog.V(5).Infof("%s: %s pod limit %q below namespace limit; setting limit to %q", api.PluginName, corev1.ResourceCPU, q.String(), nsCPUFloor.String())
			clone := nsCPUFloor.DeepCopy()
			q = &clone
		}
		if err := applyQuantity(resources.Limits, corev1.ResourceCPU, *q, mutationAllowed); err != nil {
			return fmt.Errorf("resources.limits.%s %v", corev1.ResourceCPU, err)
		}
	}

	cpuLimit, cpuFound := resources.Limits[coreapi.ResourceCPU]
	if cpuFound && config.cpuRequestToLimitRatio != 0 {
		amount := float64(cpuLimit.MilliValue()) * config.cpuRequestToLimitRatio
		q := resource.NewMilliQuantity(int64(amount), cpuLimit.Format)
		if cpuFloor.Cmp(*q) > 0 {
			clone := cpuFloor.DeepCopy()
			q = &clone
		}
		if nsCPUFloor != nil && q.Cmp(*nsCPUFloor) < 0 {
			klog.V(5).Infof("%s: %s pod limit %q below namespace limit; setting limit to %q", api.PluginName, corev1.ResourceCPU, q.String(), nsCPUFloor.String())
			clone := nsCPUFloor.DeepCopy()
			q = &clone
		}
		if err := applyQuantity(resources.Requests, corev1.ResourceCPU, *q, mutationAllowed); err != nil {
			return fmt.Errorf("resources.requests.%s %v", corev1.ResourceCPU, err)
		}
	}

	return nil
}

func applyQuantity(l coreapi.ResourceList, r corev1.ResourceName, v resource.Quantity, mutationAllowed bool) error {
	if mutationAllowed {
		l[coreapi.ResourceName(r)] = v
		return nil
	}

	if oldValue, ok := l[coreapi.ResourceName(r)]; !ok {
		return fmt.Errorf("mutated, expected: %v, now absent", v)
	} else if oldValue.Cmp(v) != 0 {
		return fmt.Errorf("mutated, expected: %v, got %v", v, oldValue)
	}

	return nil
}

// minResourceLimits finds the Min limit for resourceName. Nil is
// returned if limitRanges is empty or limits contains no resourceName
// limits.
func minResourceLimits(limitRanges []*corev1.LimitRange, resourceName corev1.ResourceName) *resource.Quantity {
	limits := []*resource.Quantity{}

	for _, limitRange := range limitRanges {
		for _, limit := range limitRange.Spec.Limits {
			if limit.Type == corev1.LimitTypeContainer {
				if limit, found := limit.Min[resourceName]; found {
					clone := limit.DeepCopy()
					limits = append(limits, &clone)
				}
			}
		}
	}

	if len(limits) == 0 {
		return nil
	}

	return minQuantity(limits)
}

func minQuantity(quantities []*resource.Quantity) *resource.Quantity {
	min := quantities[0].DeepCopy()

	for i := range quantities {
		if quantities[i].Cmp(min) < 0 {
			min = quantities[i].DeepCopy()
		}
	}

	return &min
}
