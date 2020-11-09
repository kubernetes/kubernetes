/*
Copyright 2014 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package pod

import (
	"context"
	"fmt"
	"net"
	"net/http"
	"net/url"
	"strconv"
	"strings"
	"time"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	genericfeatures "k8s.io/apiserver/pkg/features"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper/qos"
	"k8s.io/kubernetes/pkg/apis/core/validation"
	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/pkg/kubelet/client"
	proxyutil "k8s.io/kubernetes/pkg/proxy/util"
)

// podStrategy implements behavior for Pods
type podStrategy struct {
	runtime.ObjectTyper
	names.NameGenerator
}

// Strategy is the default logic that applies when creating and updating Pod
// objects via the REST API.
var Strategy = podStrategy{legacyscheme.Scheme, names.SimpleNameGenerator}

// NamespaceScoped is true for pods.
func (podStrategy) NamespaceScoped() bool {
	return true
}

// PrepareForCreate clears fields that are not allowed to be set by end users on creation.
func (podStrategy) PrepareForCreate(ctx context.Context, obj runtime.Object) {
	pod := obj.(*api.Pod)
	pod.Status = api.PodStatus{
		Phase:    api.PodPending,
		QOSClass: qos.GetPodQOS(pod),
	}

	podutil.DropDisabledPodFields(pod, nil)

	applySeccompVersionSkew(pod)
}

// PrepareForUpdate clears fields that are not allowed to be set by end users on update.
func (podStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPod := obj.(*api.Pod)
	oldPod := old.(*api.Pod)
	newPod.Status = oldPod.Status

	podutil.DropDisabledPodFields(newPod, oldPod)
}

// Validate validates a new pod.
func (podStrategy) Validate(ctx context.Context, obj runtime.Object) field.ErrorList {
	pod := obj.(*api.Pod)
	opts := validation.PodValidationOptions{
		// Allow multiple huge pages on pod create if feature is enabled
		AllowMultipleHugePageResources: utilfeature.DefaultFeatureGate.Enabled(features.HugePageStorageMediumSize),
	}
	return validation.ValidatePodCreate(pod, opts)
}

// Canonicalize normalizes the object after validation.
func (podStrategy) Canonicalize(obj runtime.Object) {
}

// AllowCreateOnUpdate is false for pods.
func (podStrategy) AllowCreateOnUpdate() bool {
	return false
}

// ValidateUpdate is the default update validation for an end user.
func (podStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	oldFailsSingleHugepagesValidation := len(validation.ValidatePodSingleHugePageResources(old.(*api.Pod), field.NewPath("spec"))) > 0
	opts := validation.PodValidationOptions{
		// Allow multiple huge pages on pod create if feature is enabled or if the old pod already has multiple hugepages specified
		AllowMultipleHugePageResources: oldFailsSingleHugepagesValidation || utilfeature.DefaultFeatureGate.Enabled(features.HugePageStorageMediumSize),
	}
	return validation.ValidatePodUpdate(obj.(*api.Pod), old.(*api.Pod), opts)
}

// AllowUnconditionalUpdate allows pods to be overwritten
func (podStrategy) AllowUnconditionalUpdate() bool {
	return true
}

// CheckGracefulDelete allows a pod to be gracefully deleted. It updates the DeleteOptions to
// reflect the desired grace value.
func (podStrategy) CheckGracefulDelete(ctx context.Context, obj runtime.Object, options *metav1.DeleteOptions) bool {
	if options == nil {
		return false
	}
	pod := obj.(*api.Pod)
	period := int64(0)
	// user has specified a value
	if options.GracePeriodSeconds != nil {
		period = *options.GracePeriodSeconds
	} else {
		// use the default value if set, or deletes the pod immediately (0)
		if pod.Spec.TerminationGracePeriodSeconds != nil {
			period = *pod.Spec.TerminationGracePeriodSeconds
		}
	}
	// if the pod is not scheduled, delete immediately
	if len(pod.Spec.NodeName) == 0 {
		period = 0
	}
	// if the pod is already terminated, delete immediately
	if pod.Status.Phase == api.PodFailed || pod.Status.Phase == api.PodSucceeded {
		period = 0
	}
	// ensure the options and the pod are in sync
	options.GracePeriodSeconds = &period
	return true
}

type podStatusStrategy struct {
	podStrategy
}

// StatusStrategy wraps and exports the used podStrategy for the storage package.
var StatusStrategy = podStatusStrategy{Strategy}

func (podStatusStrategy) PrepareForUpdate(ctx context.Context, obj, old runtime.Object) {
	newPod := obj.(*api.Pod)
	oldPod := old.(*api.Pod)
	newPod.Spec = oldPod.Spec
	newPod.DeletionTimestamp = nil

	// don't allow the pods/status endpoint to touch owner references since old kubelets corrupt them in a way
	// that breaks garbage collection
	newPod.OwnerReferences = oldPod.OwnerReferences
}

func (podStatusStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodStatusUpdate(obj.(*api.Pod), old.(*api.Pod))
}

type podEphemeralContainersStrategy struct {
	podStrategy
}

// EphemeralContainersStrategy wraps and exports the used podStrategy for the storage package.
var EphemeralContainersStrategy = podEphemeralContainersStrategy{Strategy}

func (podEphemeralContainersStrategy) ValidateUpdate(ctx context.Context, obj, old runtime.Object) field.ErrorList {
	return validation.ValidatePodEphemeralContainersUpdate(obj.(*api.Pod), old.(*api.Pod))
}

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), ToSelectableFields(pod), nil
}

// MatchPod returns a generic matcher for a given label and field selector.
func MatchPod(label labels.Selector, field fields.Selector) storage.SelectionPredicate {
	return storage.SelectionPredicate{
		Label:       label,
		Field:       field,
		GetAttrs:    GetAttrs,
		IndexFields: []string{"spec.nodeName"},
	}
}

// NodeNameTriggerFunc returns value spec.nodename of given object.
func NodeNameTriggerFunc(obj runtime.Object) string {
	return obj.(*api.Pod).Spec.NodeName
}

// NodeNameIndexFunc return value spec.nodename of given object.
func NodeNameIndexFunc(obj interface{}) ([]string, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return nil, fmt.Errorf("not a pod")
	}
	return []string{pod.Spec.NodeName}, nil
}

// Indexers returns the indexers for pod storage.
func Indexers() *cache.Indexers {
	if utilfeature.DefaultFeatureGate.Enabled(genericfeatures.SelectorIndex) {
		return &cache.Indexers{
			storage.FieldIndex("spec.nodeName"): NodeNameIndexFunc,
		}
	}
	return nil
}

// ToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func ToSelectableFields(pod *api.Pod) fields.Set {
	// The purpose of allocation with a given number of elements is to reduce
	// amount of allocations needed to create the fields.Set. If you add any
	// field here or the number of object-meta related fields changes, this should
	// be adjusted.
	podSpecificFieldsSet := make(fields.Set, 9)
	podSpecificFieldsSet["spec.nodeName"] = pod.Spec.NodeName
	podSpecificFieldsSet["spec.restartPolicy"] = string(pod.Spec.RestartPolicy)
	podSpecificFieldsSet["spec.schedulerName"] = string(pod.Spec.SchedulerName)
	podSpecificFieldsSet["spec.serviceAccountName"] = string(pod.Spec.ServiceAccountName)
	podSpecificFieldsSet["status.phase"] = string(pod.Status.Phase)
	// TODO: add podIPs as a downward API value(s) with proper format
	podIP := ""
	if len(pod.Status.PodIPs) > 0 {
		podIP = string(pod.Status.PodIPs[0].IP)
	}
	podSpecificFieldsSet["status.podIP"] = podIP
	podSpecificFieldsSet["status.nominatedNodeName"] = string(pod.Status.NominatedNodeName)
	return generic.AddObjectMetaFieldsSet(podSpecificFieldsSet, &pod.ObjectMeta, true)
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(context.Context, string, *metav1.GetOptions) (runtime.Object, error)
}

func getPod(ctx context.Context, getter ResourceGetter, name string) (*api.Pod, error) {
	obj, err := getter.Get(ctx, name, &metav1.GetOptions{})
	if err != nil {
		return nil, err
	}
	pod := obj.(*api.Pod)
	if pod == nil {
		return nil, fmt.Errorf("Unexpected object type: %#v", pod)
	}
	return pod, nil
}

// getPodIP returns primary IP for a Pod
func getPodIP(pod *api.Pod) string {
	if pod == nil {
		return ""
	}
	if len(pod.Status.PodIPs) > 0 {
		return pod.Status.PodIPs[0].IP
	}

	return ""
}

// ResourceLocation returns a URL to which one can send traffic for the specified pod.
func ResourceLocation(ctx context.Context, getter ResourceGetter, rt http.RoundTripper, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "podname" or "podname:port" or "scheme:podname:port".
	// If port is not specified, try to use the first defined port on the pod.
	scheme, name, port, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid pod request %q", id))
	}

	pod, err := getPod(ctx, getter, name)
	if err != nil {
		return nil, nil, err
	}

	// Try to figure out a port.
	if port == "" {
		for i := range pod.Spec.Containers {
			if len(pod.Spec.Containers[i].Ports) > 0 {
				port = fmt.Sprintf("%d", pod.Spec.Containers[i].Ports[0].ContainerPort)
				break
			}
		}
	}
	podIP := getPodIP(pod)
	if err := proxyutil.IsProxyableIP(podIP); err != nil {
		return nil, nil, errors.NewBadRequest(err.Error())
	}

	loc := &url.URL{
		Scheme: scheme,
	}
	if port == "" {
		// when using an ipv6 IP as a hostname in a URL, it must be wrapped in [...]
		// net.JoinHostPort does this for you.
		if strings.Contains(podIP, ":") {
			loc.Host = "[" + podIP + "]"
		} else {
			loc.Host = podIP
		}
	} else {
		loc.Host = net.JoinHostPort(podIP, port)
	}
	return loc, rt, nil
}

// LogLocation returns the log URL for a pod container. If opts.Container is blank
// and only one container is present in the pod, that container is used.
func LogLocation(
	ctx context.Context, getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	name string,
	opts *api.PodLogOptions,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(ctx, getter, name)
	if err != nil {
		return nil, nil, err
	}

	// Try to figure out a container
	// If a container was provided, it must be valid
	container := opts.Container
	container, err = validateContainer(container, pod)
	if err != nil {
		return nil, nil, err
	}
	nodeName := types.NodeName(pod.Spec.NodeName)
	if len(nodeName) == 0 {
		// If pod has not been assigned a host, return an empty location
		return nil, nil, nil
	}
	nodeInfo, err := connInfo.GetConnectionInfo(ctx, nodeName)
	if err != nil {
		return nil, nil, err
	}
	params := url.Values{}
	if opts.Follow {
		params.Add("follow", "true")
	}
	if opts.Previous {
		params.Add("previous", "true")
	}
	if opts.Timestamps {
		params.Add("timestamps", "true")
	}
	if opts.SinceSeconds != nil {
		params.Add("sinceSeconds", strconv.FormatInt(*opts.SinceSeconds, 10))
	}
	if opts.SinceTime != nil {
		params.Add("sinceTime", opts.SinceTime.Format(time.RFC3339))
	}
	if opts.TailLines != nil {
		params.Add("tailLines", strconv.FormatInt(*opts.TailLines, 10))
	}
	if opts.LimitBytes != nil {
		params.Add("limitBytes", strconv.FormatInt(*opts.LimitBytes, 10))
	}
	loc := &url.URL{
		Scheme:   nodeInfo.Scheme,
		Host:     net.JoinHostPort(nodeInfo.Hostname, nodeInfo.Port),
		Path:     fmt.Sprintf("/containerLogs/%s/%s/%s", pod.Namespace, pod.Name, container),
		RawQuery: params.Encode(),
	}

	if opts.InsecureSkipTLSVerifyBackend && utilfeature.DefaultFeatureGate.Enabled(features.AllowInsecureBackendProxy) {
		return loc, nodeInfo.InsecureSkipTLSVerifyTransport, nil
	}
	return loc, nodeInfo.Transport, nil
}

func podHasContainerWithName(pod *api.Pod, containerName string) bool {
	var hasContainer bool
	podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(), func(c *api.Container, containerType podutil.ContainerType) bool {
		if c.Name == containerName {
			hasContainer = true
			return false
		}
		return true
	})
	return hasContainer
}

func streamParams(params url.Values, opts runtime.Object) error {
	switch opts := opts.(type) {
	case *api.PodExecOptions:
		if opts.Stdin {
			params.Add(api.ExecStdinParam, "1")
		}
		if opts.Stdout {
			params.Add(api.ExecStdoutParam, "1")
		}
		if opts.Stderr {
			params.Add(api.ExecStderrParam, "1")
		}
		if opts.TTY {
			params.Add(api.ExecTTYParam, "1")
		}
		for _, c := range opts.Command {
			params.Add("command", c)
		}
	case *api.PodAttachOptions:
		if opts.Stdin {
			params.Add(api.ExecStdinParam, "1")
		}
		if opts.Stdout {
			params.Add(api.ExecStdoutParam, "1")
		}
		if opts.Stderr {
			params.Add(api.ExecStderrParam, "1")
		}
		if opts.TTY {
			params.Add(api.ExecTTYParam, "1")
		}
	case *api.PodPortForwardOptions:
		if len(opts.Ports) > 0 {
			ports := make([]string, len(opts.Ports))
			for i, p := range opts.Ports {
				ports[i] = strconv.FormatInt(int64(p), 10)
			}
			params.Add(api.PortHeader, strings.Join(ports, ","))
		}
	default:
		return fmt.Errorf("Unknown object for streaming: %v", opts)
	}
	return nil
}

// AttachLocation returns the attach URL for a pod container. If opts.Container is blank
// and only one container is present in the pod, that container is used.
func AttachLocation(
	ctx context.Context,
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	name string,
	opts *api.PodAttachOptions,
) (*url.URL, http.RoundTripper, error) {
	return streamLocation(ctx, getter, connInfo, name, opts, opts.Container, "attach")
}

// ExecLocation returns the exec URL for a pod container. If opts.Container is blank
// and only one container is present in the pod, that container is used.
func ExecLocation(
	ctx context.Context,
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	name string,
	opts *api.PodExecOptions,
) (*url.URL, http.RoundTripper, error) {
	return streamLocation(ctx, getter, connInfo, name, opts, opts.Container, "exec")
}

func streamLocation(
	ctx context.Context,
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	name string,
	opts runtime.Object,
	container,
	path string,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(ctx, getter, name)
	if err != nil {
		return nil, nil, err
	}

	// Try to figure out a container
	// If a container was provided, it must be valid
	container, err = validateContainer(container, pod)
	if err != nil {
		return nil, nil, err
	}

	nodeName := types.NodeName(pod.Spec.NodeName)
	if len(nodeName) == 0 {
		// If pod has not been assigned a host, return an empty location
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("pod %s does not have a host assigned", name))
	}
	nodeInfo, err := connInfo.GetConnectionInfo(ctx, nodeName)
	if err != nil {
		return nil, nil, err
	}
	params := url.Values{}
	if err := streamParams(params, opts); err != nil {
		return nil, nil, err
	}
	loc := &url.URL{
		Scheme:   nodeInfo.Scheme,
		Host:     net.JoinHostPort(nodeInfo.Hostname, nodeInfo.Port),
		Path:     fmt.Sprintf("/%s/%s/%s/%s", path, pod.Namespace, pod.Name, container),
		RawQuery: params.Encode(),
	}
	return loc, nodeInfo.Transport, nil
}

// PortForwardLocation returns the port-forward URL for a pod.
func PortForwardLocation(
	ctx context.Context,
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	name string,
	opts *api.PodPortForwardOptions,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(ctx, getter, name)
	if err != nil {
		return nil, nil, err
	}

	nodeName := types.NodeName(pod.Spec.NodeName)
	if len(nodeName) == 0 {
		// If pod has not been assigned a host, return an empty location
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("pod %s does not have a host assigned", name))
	}
	nodeInfo, err := connInfo.GetConnectionInfo(ctx, nodeName)
	if err != nil {
		return nil, nil, err
	}
	params := url.Values{}
	if err := streamParams(params, opts); err != nil {
		return nil, nil, err
	}
	loc := &url.URL{
		Scheme:   nodeInfo.Scheme,
		Host:     net.JoinHostPort(nodeInfo.Hostname, nodeInfo.Port),
		Path:     fmt.Sprintf("/portForward/%s/%s", pod.Namespace, pod.Name),
		RawQuery: params.Encode(),
	}
	return loc, nodeInfo.Transport, nil
}

// validateContainer validate container is valid for pod, return valid container
func validateContainer(container string, pod *api.Pod) (string, error) {
	if len(container) == 0 {
		switch len(pod.Spec.Containers) {
		case 1:
			container = pod.Spec.Containers[0].Name
		case 0:
			return "", errors.NewBadRequest(fmt.Sprintf("a container name must be specified for pod %s", pod.Name))
		default:
			var containerNames []string
			podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(), func(c *api.Container, containerType podutil.ContainerType) bool {
				containerNames = append(containerNames, c.Name)
				return true
			})
			errStr := fmt.Sprintf("a container name must be specified for pod %s, choose one of: %s", pod.Name, containerNames)
			return "", errors.NewBadRequest(errStr)
		}
	} else {
		if !podHasContainerWithName(pod, container) {
			return "", errors.NewBadRequest(fmt.Sprintf("container %s is not valid for pod %s", container, pod.Name))
		}
	}

	return container, nil
}

// applySeccompVersionSkew implements the version skew behavior described in:
// https://github.com/kubernetes/enhancements/blob/master/keps/sig-node/20190717-seccomp-ga.md#version-skew-strategy
func applySeccompVersionSkew(pod *api.Pod) {
	// get possible annotation and field
	annotation, hasAnnotation := pod.Annotations[v1.SeccompPodAnnotationKey]
	field, hasField := (*api.SeccompProfile)(nil), false

	if pod.Spec.SecurityContext != nil && pod.Spec.SecurityContext.SeccompProfile != nil {
		field = pod.Spec.SecurityContext.SeccompProfile
		hasField = true
	}

	// sync field and annotation
	if hasField && !hasAnnotation {
		newAnnotation := podutil.SeccompAnnotationForField(field)

		if newAnnotation != "" {
			if pod.Annotations == nil {
				pod.Annotations = map[string]string{}
			}
			pod.Annotations[v1.SeccompPodAnnotationKey] = newAnnotation
		}
	} else if hasAnnotation && !hasField {
		newField := podutil.SeccompFieldForAnnotation(annotation)

		if newField != nil {
			if pod.Spec.SecurityContext == nil {
				pod.Spec.SecurityContext = &api.PodSecurityContext{}
			}
			pod.Spec.SecurityContext.SeccompProfile = newField
		}
	}

	// Handle the containers of the pod
	podutil.VisitContainers(&pod.Spec, podutil.AllFeatureEnabledContainers(),
		func(ctr *api.Container, _ podutil.ContainerType) bool {
			// get possible annotation and field
			key := api.SeccompContainerAnnotationKeyPrefix + ctr.Name
			annotation, hasAnnotation := pod.Annotations[key]

			field, hasField := (*api.SeccompProfile)(nil), false
			if ctr.SecurityContext != nil && ctr.SecurityContext.SeccompProfile != nil {
				field = ctr.SecurityContext.SeccompProfile
				hasField = true
			}

			// sync field and annotation
			if hasField && !hasAnnotation {
				newAnnotation := podutil.SeccompAnnotationForField(field)

				if newAnnotation != "" {
					if pod.Annotations == nil {
						pod.Annotations = map[string]string{}
					}
					pod.Annotations[key] = newAnnotation
				}
			} else if hasAnnotation && !hasField {
				newField := podutil.SeccompFieldForAnnotation(annotation)

				if newField != nil {
					if ctr.SecurityContext == nil {
						ctr.SecurityContext = &api.SecurityContext{}
					}
					ctr.SecurityContext.SeccompProfile = newField
				}
			}

			return true
		})
}
