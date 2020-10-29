package managementcpusoverride

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"reflect"
	"strings"
	"time"

	configv1 "github.com/openshift/api/config/v1"
	configv1informer "github.com/openshift/client-go/config/informers/externalversions/config/v1"
	configv1listers "github.com/openshift/client-go/config/listers/config/v1"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/admission"
	"k8s.io/apiserver/pkg/admission/initializer"
	"k8s.io/apiserver/pkg/warning"
	"k8s.io/client-go/informers"
	"k8s.io/client-go/kubernetes"
	corev1listers "k8s.io/client-go/listers/core/v1"
	coreapi "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubelet/cm"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

const (
	PluginName = "autoscaling.openshift.io/ManagementCPUsOverride"
	// timeToWaitForCacheSync contains the time how long to wait for caches to be synchronize
	timeToWaitForCacheSync = 10 * time.Second
	// containerWorkloadResourceSuffix contains the suffix for the container workload resource
	containerWorkloadResourceSuffix = "workload.openshift.io/cores"
	// podWorkloadTargetAnnotationPrefix contains the prefix for the pod workload target annotation
	podWorkloadTargetAnnotationPrefix = "target.workload.openshift.io/"
	// podWorkloadAnnotationEffect contains the effect key for the workload annotation value
	podWorkloadAnnotationEffect = "effect"
	// workloadEffectPreferredDuringScheduling contains the PreferredDuringScheduling effect value
	workloadEffectPreferredDuringScheduling = "PreferredDuringScheduling"
	// containerResourcesAnnotationPrefix contains resource annotation prefix that will be used by CRI-O to set cpu shares
	containerResourcesAnnotationPrefix = "resources.workload.openshift.io/"
	// containerResourcesAnnotationValueKeyCPUShares contains resource annotation value cpushares key
	containerResourcesAnnotationValueKeyCPUShares = "cpushares"
	// namespaceAllowedAnnotation contains the namespace allowed annotation key
	namespaceAllowedAnnotation = "workload.openshift.io/allowed"
	// workloadAdmissionWarning contains the admission warning annotation key
	workloadAdmissionWarning = "workload.openshift.io/warning"
	// infraClusterName contains the name of the cluster infrastructure resource
	infraClusterName = "cluster"
	// debugSourceResourceAnnotation contains the debug annotation that refers to the pod resource
	debugSourceResourceAnnotation = "debug.openshift.io/source-resource"
)

var _ = initializer.WantsExternalKubeInformerFactory(&managementCPUsOverride{})
var _ = initializer.WantsExternalKubeClientSet(&managementCPUsOverride{})
var _ = admission.MutationInterface(&managementCPUsOverride{})
var _ = admission.ValidationInterface(&managementCPUsOverride{})
var _ = WantsInfraInformer(&managementCPUsOverride{})

func Register(plugins *admission.Plugins) {
	plugins.Register(PluginName,
		func(config io.Reader) (admission.Interface, error) {
			return &managementCPUsOverride{
				Handler: admission.NewHandler(admission.Create),
			}, nil
		})
}

// managementCPUsOverride presents admission plugin that should replace pod container CPU requests with a new management resource.
// It applies to all pods that:
// 1. are in an allowed namespace
// 2. and have the workload annotation.
//
// It also sets the new management resource request and limit and  set resource annotation that CRI-O can
// recognize and apply the relevant changes.
// For more information, see - https://github.com/openshift/enhancements/pull/703
//
// Conditions for CPUs requests deletion:
// 1. The namespace should have allowed annotation "workload.openshift.io/allowed": "management"
// 2. The pod should have management annotation: "workload.openshift.io/management": "{"effect": "PreferredDuringScheduling"}"
// 3. All nodes under the cluster should have new management resource - "management.workload.openshift.io/cores"
// 4. The CPU request deletion will not change the pod QoS class
type managementCPUsOverride struct {
	*admission.Handler
	client                kubernetes.Interface
	nsLister              corev1listers.NamespaceLister
	nsListerSynced        func() bool
	nodeLister            corev1listers.NodeLister
	nodeListSynced        func() bool
	infraConfigLister     configv1listers.InfrastructureLister
	infraConfigListSynced func() bool
}

func (a *managementCPUsOverride) SetExternalKubeInformerFactory(kubeInformers informers.SharedInformerFactory) {
	a.nsLister = kubeInformers.Core().V1().Namespaces().Lister()
	a.nsListerSynced = kubeInformers.Core().V1().Namespaces().Informer().HasSynced
	a.nodeLister = kubeInformers.Core().V1().Nodes().Lister()
	a.nodeListSynced = kubeInformers.Core().V1().Nodes().Informer().HasSynced
}

// SetExternalKubeClientSet implements the WantsExternalKubeClientSet interface.
func (a *managementCPUsOverride) SetExternalKubeClientSet(client kubernetes.Interface) {
	a.client = client
}

func (a *managementCPUsOverride) SetInfraInformer(informer configv1informer.InfrastructureInformer) {
	a.infraConfigLister = informer.Lister()
	a.infraConfigListSynced = informer.Informer().HasSynced
}

func (a *managementCPUsOverride) ValidateInitialization() error {
	if a.client == nil {
		return fmt.Errorf("%s plugin needs a kubernetes client", PluginName)
	}
	if a.nsLister == nil {
		return fmt.Errorf("%s did not get a namespace lister", PluginName)
	}
	if a.nsListerSynced == nil {
		return fmt.Errorf("%s plugin needs a namespace lister synced", PluginName)
	}
	if a.nodeLister == nil {
		return fmt.Errorf("%s did not get a node lister", PluginName)
	}
	if a.nodeListSynced == nil {
		return fmt.Errorf("%s plugin needs a node lister synced", PluginName)
	}
	if a.infraConfigLister == nil {
		return fmt.Errorf("%s did not get a config infrastructure lister", PluginName)
	}
	if a.infraConfigListSynced == nil {
		return fmt.Errorf("%s plugin needs a config infrastructure lister synced", PluginName)
	}
	return nil
}

func (a *managementCPUsOverride) Admit(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) error {
	if attr.GetResource().GroupResource() != coreapi.Resource("pods") || attr.GetSubresource() != "" {
		return nil
	}

	pod, ok := attr.GetObject().(*coreapi.Pod)
	if !ok {
		return admission.NewForbidden(attr, fmt.Errorf("unexpected object: %#v", attr.GetObject()))
	}

	// do not mutate mirror pods at all
	if isStaticPod(pod.Annotations) {
		return nil
	}

	podAnnotations := map[string]string{}
	for k, v := range pod.Annotations {
		podAnnotations[k] = v
	}

	// strip any resource annotations specified by a user
	stripResourcesAnnotations(pod.Annotations)
	// strip any workload annotation to prevent from underlying components(CRI-O, kubelet) to apply any changes
	// according to the workload annotation
	stripWorkloadAnnotations(pod.Annotations)

	workloadType, err := getWorkloadType(podAnnotations)
	if err != nil {
		invalidError := getPodInvalidWorkloadAnnotationError(podAnnotations, err.Error())
		return errors.NewInvalid(coreapi.Kind("Pod"), pod.Name, field.ErrorList{invalidError})
	}

	// no workload annotation is specified under the pod
	if len(workloadType) == 0 {
		return nil
	}

	if !a.waitForSyncedStore(time.After(timeToWaitForCacheSync)) {
		return admission.NewForbidden(attr, fmt.Errorf("%s node or namespace or infra config cache not synchronized", PluginName))
	}

	nodes, err := a.nodeLister.List(labels.Everything())
	if err != nil {
		return admission.NewForbidden(attr, err) // can happen due to informer latency
	}

	// we still need to have nodes under the cluster to decide if the management resource enabled or not
	if len(nodes) == 0 {
		return admission.NewForbidden(attr, fmt.Errorf("%s the cluster does not have any nodes", PluginName))
	}

	clusterInfra, err := a.infraConfigLister.Get(infraClusterName)
	if err != nil {
		return admission.NewForbidden(attr, err) // can happen due to informer latency
	}

	// the infrastructure status is empty, so we can not decide the cluster type
	if reflect.DeepEqual(clusterInfra.Status, configv1.InfrastructureStatus{}) {
		return admission.NewForbidden(attr, fmt.Errorf("%s infrastructure resource has empty status", PluginName))
	}

	// the infrastructure status is not empty, but topology related fields do not have any values indicates that
	// the cluster is during the roll-back process to the version that does not support the topology fields
	// the upgrade to 4.8 handled by the CR defaulting
	if clusterInfra.Status.ControlPlaneTopology == "" && clusterInfra.Status.InfrastructureTopology == "" {
		return nil
	}

	// Check if we are in CPU Partitioning mode for AllNodes
	if !isCPUPartitioning(clusterInfra.Status, nodes, workloadType) {
		return nil
	}

	// allow annotations on project to override management pods CPUs requests
	ns, err := a.getPodNamespace(attr)
	if err != nil {
		return err
	}

	if !doesNamespaceAllowWorkloadType(ns.Annotations, workloadType) {
		return admission.NewForbidden(attr, fmt.Errorf("%s the pod namespace %q does not allow the workload type %s", PluginName, ns.Name, workloadType))
	}

	workloadAnnotation := fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadType)
	effect, err := getWorkloadAnnotationEffect(podAnnotations[workloadAnnotation])
	if err != nil {
		invalidError := getPodInvalidWorkloadAnnotationError(podAnnotations, fmt.Sprintf("failed to get workload annotation effect: %v", err))
		return errors.NewInvalid(coreapi.Kind("Pod"), pod.Name, field.ErrorList{invalidError})
	}

	// TODO: currently we support only PreferredDuringScheduling effect
	if effect != workloadEffectPreferredDuringScheduling {
		invalidError := getPodInvalidWorkloadAnnotationError(podAnnotations, fmt.Sprintf("only %q effect is supported", workloadEffectPreferredDuringScheduling))
		return errors.NewInvalid(coreapi.Kind("Pod"), pod.Name, field.ErrorList{invalidError})
	}

	allContainers := append([]coreapi.Container{}, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	podQoSClass := getPodQoSClass(allContainers)

	// we do not want to change guaranteed pods resource allocation, because it should be managed by
	// relevant managers(CPU and memory) under the kubelet
	if podQoSClass == coreapi.PodQOSGuaranteed {
		pod.Annotations[workloadAdmissionWarning] = "skip pod CPUs requests modifications because it has guaranteed QoS class"
		return nil
	}

	// we should skip mutation of the pod that has container with both CPU limit and request because once we will remove
	// the request, the defaulter will set the request back with the CPU limit value
	if podHasBothCPULimitAndRequest(allContainers) {
		pod.Annotations[workloadAdmissionWarning] = "skip pod CPUs requests modifications because pod container has both CPU limit and request"
		return nil
	}

	// before we update the pod available under admission attributes, we need to verify that deletion of the CPU request
	// will not change the pod QoS class, otherwise skip pod mutation
	// 1. Copy the pod
	// 2. Delete CPUs requests for all containers under the pod
	// 3. Get modified pod QoS class
	// 4. Verify that the pod QoS class before and after the modification stay the same
	// 5. Update the pod under admission attributes
	podCopy := pod.DeepCopy()
	updatePodResources(podCopy, workloadType, podQoSClass)

	allContainersCopy := append([]coreapi.Container{}, podCopy.Spec.InitContainers...)
	allContainersCopy = append(allContainersCopy, podCopy.Spec.Containers...)
	podQoSClassAfterModification := getPodQoSClass(allContainersCopy)

	if podQoSClass != podQoSClassAfterModification {
		pod.Annotations[workloadAdmissionWarning] = fmt.Sprintf("skip pod CPUs requests modifications because it will change the pod QoS class from %s to %s", podQoSClass, podQoSClassAfterModification)
		return nil
	}

	updatePodResources(pod, workloadType, podQoSClass)

	return nil
}

func isCPUPartitioning(infraStatus configv1.InfrastructureStatus, nodes []*corev1.Node, workloadType string) bool {
	// If status is not for CPU partitioning and we're single node we also check nodes to support upgrade event
	// TODO: This should not be needed after 4.13 as all clusters after should have this feature on at install time, or updated by migration in NTO.
	if infraStatus.CPUPartitioning != configv1.CPUPartitioningAllNodes && infraStatus.ControlPlaneTopology == configv1.SingleReplicaTopologyMode {
		managedResource := fmt.Sprintf("%s.%s", workloadType, containerWorkloadResourceSuffix)
		for _, node := range nodes {
			// We only expect a single node to exist, so we return on first hit
			if _, ok := node.Status.Allocatable[corev1.ResourceName(managedResource)]; ok {
				return true
			}
		}
	}
	return infraStatus.CPUPartitioning == configv1.CPUPartitioningAllNodes
}

func (a *managementCPUsOverride) getPodNamespace(attr admission.Attributes) (*corev1.Namespace, error) {
	ns, err := a.nsLister.Get(attr.GetNamespace())
	if err == nil {
		return ns, nil
	}

	if !errors.IsNotFound(err) {
		return nil, admission.NewForbidden(attr, err)
	}

	// in case of latency in our caches, make a call direct to storage to verify that it truly exists or not
	ns, err = a.client.CoreV1().Namespaces().Get(context.TODO(), attr.GetNamespace(), metav1.GetOptions{})
	if err == nil {
		return ns, nil
	}

	if !errors.IsNotFound(err) {
		return nil, admission.NewForbidden(attr, err)
	}

	return nil, err
}

func (a *managementCPUsOverride) waitForSyncedStore(timeout <-chan time.Time) bool {
	for !a.nsListerSynced() || !a.nodeListSynced() || !a.infraConfigListSynced() {
		select {
		case <-time.After(100 * time.Millisecond):
		case <-timeout:
			return a.nsListerSynced() && a.nodeListSynced() && a.infraConfigListSynced()
		}
	}

	return true
}

func updatePodResources(pod *coreapi.Pod, workloadType string, class coreapi.PodQOSClass) {
	if pod.Annotations == nil {
		pod.Annotations = map[string]string{}
	}

	// update init containers resources
	updateContainersResources(pod.Spec.InitContainers, pod.Annotations, workloadType, class)

	// update app containers resources
	updateContainersResources(pod.Spec.Containers, pod.Annotations, workloadType, class)

	// re-add workload annotation
	addWorkloadAnnotations(pod.Annotations, workloadType)
}

func updateContainersResources(containers []coreapi.Container, podAnnotations map[string]string, workloadType string, podQoSClass coreapi.PodQOSClass) {
	for i := range containers {
		c := &containers[i]
		cpusharesAnnotationKey := fmt.Sprintf("%s%s", containerResourcesAnnotationPrefix, c.Name)

		// make sure best effort is always 2 shares, it the minimal shares that supported
		// see - https://github.com/kubernetes/kubernetes/blob/46563b0abebbb00e21db967950a1343e83a0c6a2/pkg/kubelet/cm/qos_container_manager_linux.go#L99
		if podQoSClass == coreapi.PodQOSBestEffort {
			podAnnotations[cpusharesAnnotationKey] = fmt.Sprintf(`{"%s": 2}`, containerResourcesAnnotationValueKeyCPUShares)
			continue
		}

		if c.Resources.Requests != nil {
			if _, ok := c.Resources.Requests[coreapi.ResourceCPU]; !ok {
				continue
			}

			cpuRequest := c.Resources.Requests[coreapi.ResourceCPU]
			cpuRequestInMilli := cpuRequest.MilliValue()

			cpuShares := cm.MilliCPUToShares(cpuRequestInMilli)
			podAnnotations[cpusharesAnnotationKey] = fmt.Sprintf(`{"%s": %d}`, containerResourcesAnnotationValueKeyCPUShares, cpuShares)
			delete(c.Resources.Requests, coreapi.ResourceCPU)

			if c.Resources.Limits == nil {
				c.Resources.Limits = coreapi.ResourceList{}
			}

			// multiply the CPU request by 1000, to make sure that the resource will pass integer validation
			managedResource := fmt.Sprintf("%s.%s", workloadType, containerWorkloadResourceSuffix)
			newCPURequest := resource.NewMilliQuantity(cpuRequestInMilli*1000, cpuRequest.Format)
			c.Resources.Requests[coreapi.ResourceName(managedResource)] = *newCPURequest
			c.Resources.Limits[coreapi.ResourceName(managedResource)] = *newCPURequest
		}
	}
}

func isGuaranteed(containers []coreapi.Container) bool {
	for _, c := range containers {
		// only memory and CPU resources are relevant to decide pod QoS class
		for _, r := range []coreapi.ResourceName{coreapi.ResourceMemory, coreapi.ResourceCPU} {
			limit := c.Resources.Limits[r]
			request, requestExist := c.Resources.Requests[r]

			if limit.IsZero() {
				return false
			}

			if !requestExist {
				continue
			}

			// it some corner case, when you set CPU request to 0 the k8s will change it to the value
			// specified under the limit
			if r == coreapi.ResourceCPU && request.IsZero() {
				continue
			}

			if !limit.Equal(request) {
				return false
			}
		}
	}

	return true
}

func isBestEffort(containers []coreapi.Container) bool {
	for _, c := range containers {
		// only memory and CPU resources are relevant to decide pod QoS class
		for _, r := range []coreapi.ResourceName{coreapi.ResourceMemory, coreapi.ResourceCPU} {
			limit := c.Resources.Limits[r]
			request := c.Resources.Requests[r]

			if !limit.IsZero() || !request.IsZero() {
				return false
			}
		}
	}

	return true
}

func getPodQoSClass(containers []coreapi.Container) coreapi.PodQOSClass {
	if isGuaranteed(containers) {
		return coreapi.PodQOSGuaranteed
	}

	if isBestEffort(containers) {
		return coreapi.PodQOSBestEffort
	}

	return coreapi.PodQOSBurstable
}

func podHasBothCPULimitAndRequest(containers []coreapi.Container) bool {
	for _, c := range containers {
		_, cpuRequestExists := c.Resources.Requests[coreapi.ResourceCPU]
		_, cpuLimitExists := c.Resources.Limits[coreapi.ResourceCPU]

		if cpuRequestExists && cpuLimitExists {
			return true
		}
	}

	return false
}

func doesNamespaceAllowWorkloadType(annotations map[string]string, workloadType string) bool {
	v, found := annotations[namespaceAllowedAnnotation]
	if !found {
		return false
	}

	for _, t := range strings.Split(v, ",") {
		if workloadType == t {
			return true
		}
	}

	return false
}

func getWorkloadType(annotations map[string]string) (string, error) {
	var workloadAnnotationsKeys []string
	for k := range annotations {
		if strings.HasPrefix(k, podWorkloadTargetAnnotationPrefix) {
			workloadAnnotationsKeys = append(workloadAnnotationsKeys, k)
		}
	}

	// no workload annotation is specified under the pod
	if len(workloadAnnotationsKeys) == 0 {
		return "", nil
	}

	// more than one workload annotation exists under the pod and we do not support different workload types
	// under the same pod
	if len(workloadAnnotationsKeys) > 1 {
		return "", fmt.Errorf("the pod can not have more than one workload annotations")
	}

	workloadType := strings.TrimPrefix(workloadAnnotationsKeys[0], podWorkloadTargetAnnotationPrefix)
	if len(workloadType) == 0 {
		return "", fmt.Errorf("the workload annotation key should have format %s<workload_type>, when <workload_type> is non empty string", podWorkloadTargetAnnotationPrefix)
	}

	return workloadType, nil
}

func getWorkloadAnnotationEffect(workloadAnnotationKey string) (string, error) {
	managementAnnotationValue := map[string]string{}
	if err := json.Unmarshal([]byte(workloadAnnotationKey), &managementAnnotationValue); err != nil {
		return "", fmt.Errorf("failed to parse %q annotation value: %v", workloadAnnotationKey, err)
	}

	if len(managementAnnotationValue) > 1 {
		return "", fmt.Errorf("the workload annotation value %q has more than one key", managementAnnotationValue)
	}

	effect, ok := managementAnnotationValue[podWorkloadAnnotationEffect]
	if !ok {
		return "", fmt.Errorf("the workload annotation value %q does not have %q key", managementAnnotationValue, podWorkloadAnnotationEffect)
	}
	return effect, nil
}

func stripResourcesAnnotations(annotations map[string]string) {
	for k := range annotations {
		if strings.HasPrefix(k, containerResourcesAnnotationPrefix) {
			delete(annotations, k)
		}
	}
}

func stripWorkloadAnnotations(annotations map[string]string) {
	for k := range annotations {
		if strings.HasPrefix(k, podWorkloadTargetAnnotationPrefix) {
			delete(annotations, k)
		}
	}
}

func addWorkloadAnnotations(annotations map[string]string, workloadType string) {
	if annotations == nil {
		annotations = map[string]string{}
	}

	workloadAnnotation := fmt.Sprintf("%s%s", podWorkloadTargetAnnotationPrefix, workloadType)
	annotations[workloadAnnotation] = fmt.Sprintf(`{"%s":"%s"}`, podWorkloadAnnotationEffect, workloadEffectPreferredDuringScheduling)
}

func (a *managementCPUsOverride) Validate(ctx context.Context, attr admission.Attributes, o admission.ObjectInterfaces) (err error) {
	if attr.GetResource().GroupResource() != coreapi.Resource("pods") || attr.GetSubresource() != "" {
		return nil
	}

	pod, ok := attr.GetObject().(*coreapi.Pod)
	if !ok {
		return admission.NewForbidden(attr, fmt.Errorf("unexpected object: %#v", attr.GetObject()))
	}

	// do not validate mirror pods at all
	if isStaticPod(pod.Annotations) {
		return nil
	}

	ns, err := a.getPodNamespace(attr)
	if err != nil {
		return err
	}

	var allErrs field.ErrorList
	workloadType, err := getWorkloadType(pod.Annotations)
	if err != nil {
		allErrs = append(allErrs, getPodInvalidWorkloadAnnotationError(pod.Annotations, err.Error()))
	}

	workloadResourceAnnotations := map[string]map[string]int{}
	for k, v := range pod.Annotations {
		if !strings.HasPrefix(k, containerResourcesAnnotationPrefix) {
			continue
		}

		resourceAnnotationValue := map[string]int{}
		if err := json.Unmarshal([]byte(v), &resourceAnnotationValue); err != nil {
			allErrs = append(allErrs, getPodInvalidWorkloadAnnotationError(pod.Annotations, err.Error()))
		}
		workloadResourceAnnotations[k] = resourceAnnotationValue
	}

	containersWorkloadResources := map[string]*coreapi.Container{}
	allContainers := append([]coreapi.Container{}, pod.Spec.InitContainers...)
	allContainers = append(allContainers, pod.Spec.Containers...)
	for i := range allContainers {
		c := &allContainers[i]
		// we interested only in request because only the request affects the scheduler
		for r := range c.Resources.Requests {
			resourceName := string(r)
			if strings.HasSuffix(resourceName, containerWorkloadResourceSuffix) {
				containersWorkloadResources[resourceName] = c
			}
		}
	}

	// the pod does not have workload annotation
	if len(workloadType) == 0 {
		if len(workloadResourceAnnotations) > 0 {
			allErrs = append(allErrs, getPodInvalidWorkloadAnnotationError(pod.Annotations, "the pod without workload annotation can not have resource annotation"))
		}

		for resourceName, c := range containersWorkloadResources {
			if isDebugPod(pod.Annotations) {
				warning.AddWarning(ctx, "", "You must pass --keep-annotations parameter to the debug command or upgrade the oc tool to the latest version when trying to debug a pod with workload partitioning resources.")
			}

			allErrs = append(allErrs, field.Invalid(field.NewPath("spec.containers.resources.requests"), c.Resources.Requests, fmt.Sprintf("the pod without workload annotations can not have containers with workload resources %q", resourceName)))
		}
	} else {
		if !doesNamespaceAllowWorkloadType(ns.Annotations, workloadType) { // pod has workload annotation, but the pod does not have workload annotation
			allErrs = append(allErrs, getPodInvalidWorkloadAnnotationError(pod.Annotations, fmt.Sprintf("the pod can not have workload annotation, when the namespace %q does not allow it", ns.Name)))
		}

		for _, v := range workloadResourceAnnotations {
			if len(v) > 1 {
				allErrs = append(allErrs, field.Invalid(field.NewPath("metadata.annotations"), pod.Annotations, "the pod resource annotation value can not have more than one key"))
			}

			// the pod should not have any resource annotations with the value that includes keys different from cpushares
			if _, ok := v[containerResourcesAnnotationValueKeyCPUShares]; len(v) == 1 && !ok {
				allErrs = append(allErrs, field.Invalid(field.NewPath("metadata.annotations"), pod.Annotations, "the pod resource annotation value should have only cpushares key"))
			}
		}
	}

	if len(allErrs) == 0 {
		return nil
	}

	return errors.NewInvalid(coreapi.Kind("Pod"), pod.Name, allErrs)
}

func getPodInvalidWorkloadAnnotationError(annotations map[string]string, message string) *field.Error {
	return field.Invalid(field.NewPath("metadata.Annotations"), annotations, message)
}

// isStaticPod returns true if the pod is a static pod.
func isStaticPod(annotations map[string]string) bool {
	source, ok := annotations[kubetypes.ConfigSourceAnnotationKey]
	return ok && source != kubetypes.ApiserverSource
}

func isDebugPod(annotations map[string]string) bool {
	_, ok := annotations[debugSourceResourceAnnotation]
	return ok
}
