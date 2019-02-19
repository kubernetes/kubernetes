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

	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	utilnet "k8s.io/apimachinery/pkg/util/net"
	"k8s.io/apimachinery/pkg/util/validation/field"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/names"
	"k8s.io/kubernetes/pkg/api/legacyscheme"
	podutil "k8s.io/kubernetes/pkg/api/pod"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/apis/core/helper/qos"
	"k8s.io/kubernetes/pkg/apis/core/validation"
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
	allErrs := validation.ValidatePod(pod)
	allErrs = append(allErrs, validation.ValidateConditionalPod(pod, nil, field.NewPath(""))...)
	return allErrs
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
	errorList := validation.ValidatePod(obj.(*api.Pod))
	errorList = append(errorList, validation.ValidatePodUpdate(obj.(*api.Pod), old.(*api.Pod))...)
	errorList = append(errorList, validation.ValidateConditionalPod(obj.(*api.Pod), old.(*api.Pod), field.NewPath(""))...)
	return errorList
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

type podStrategyWithoutGraceful struct {
	podStrategy
}

// CheckGracefulDelete prohibits graceful deletion.
func (podStrategyWithoutGraceful) CheckGracefulDelete(ctx context.Context, obj runtime.Object, options *metav1.DeleteOptions) bool {
	return false
}

// StrategyWithoutGraceful implements the legacy instant delele behavior.
var StrategyWithoutGraceful = podStrategyWithoutGraceful{Strategy}

type podStatusStrategy struct {
	podStrategy
}

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

// GetAttrs returns labels and fields of a given object for filtering purposes.
func GetAttrs(obj runtime.Object) (labels.Set, fields.Set, error) {
	pod, ok := obj.(*api.Pod)
	if !ok {
		return nil, nil, fmt.Errorf("not a pod")
	}
	return labels.Set(pod.ObjectMeta.Labels), PodToSelectableFields(pod), nil
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

func NodeNameTriggerFunc(obj runtime.Object) []storage.MatchValue {
	pod := obj.(*api.Pod)
	result := storage.MatchValue{IndexName: "spec.nodeName", Value: pod.Spec.NodeName}
	return []storage.MatchValue{result}
}

// PodToSelectableFields returns a field set that represents the object
// TODO: fields are not labels, and the validation rules for them do not apply.
func PodToSelectableFields(pod *api.Pod) fields.Set {
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
	podSpecificFieldsSet["status.podIP"] = string(pod.Status.PodIP)
	podSpecificFieldsSet["status.nominatedNodeName"] = string(pod.Status.NominatedNodeName)
	return generic.AddObjectMetaFieldsSet(podSpecificFieldsSet, &pod.ObjectMeta, true)
}

// ResourceGetter is an interface for retrieving resources by ResourceLocation.
type ResourceGetter interface {
	Get(context.Context, string, *metav1.GetOptions) (runtime.Object, error)
}

func getPod(getter ResourceGetter, ctx context.Context, name string) (*api.Pod, error) {
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

// ResourceLocation returns a URL to which one can send traffic for the specified pod.
func ResourceLocation(getter ResourceGetter, rt http.RoundTripper, ctx context.Context, id string) (*url.URL, http.RoundTripper, error) {
	// Allow ID as "podname" or "podname:port" or "scheme:podname:port".
	// If port is not specified, try to use the first defined port on the pod.
	scheme, name, port, valid := utilnet.SplitSchemeNamePort(id)
	if !valid {
		return nil, nil, errors.NewBadRequest(fmt.Sprintf("invalid pod request %q", id))
	}
	// TODO: if port is not a number but a "(container)/(portname)", do a name lookup.

	pod, err := getPod(getter, ctx, name)
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

	if err := proxyutil.IsProxyableIP(pod.Status.PodIP); err != nil {
		return nil, nil, errors.NewBadRequest(err.Error())
	}

	loc := &url.URL{
		Scheme: scheme,
	}
	if port == "" {
		loc.Host = pod.Status.PodIP
	} else {
		loc.Host = net.JoinHostPort(pod.Status.PodIP, port)
	}
	return loc, rt, nil
}

// getContainerNames returns a formatted string containing the container names
func getContainerNames(containers []api.Container) string {
	names := []string{}
	for _, c := range containers {
		names = append(names, c.Name)
	}
	return strings.Join(names, " ")
}

// LogLocation returns the log URL for a pod container. If opts.Container is blank
// and only one container is present in the pod, that container is used.
func LogLocation(
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	ctx context.Context,
	name string,
	opts *api.PodLogOptions,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(getter, ctx, name)
	if err != nil {
		return nil, nil, err
	}

	// Try to figure out a container
	// If a container was provided, it must be valid
	container := opts.Container
	if len(container) == 0 {
		switch len(pod.Spec.Containers) {
		case 1:
			container = pod.Spec.Containers[0].Name
		case 0:
			return nil, nil, errors.NewBadRequest(fmt.Sprintf("a container name must be specified for pod %s", name))
		default:
			containerNames := getContainerNames(pod.Spec.Containers)
			initContainerNames := getContainerNames(pod.Spec.InitContainers)
			err := fmt.Sprintf("a container name must be specified for pod %s, choose one of: [%s]", name, containerNames)
			if len(initContainerNames) > 0 {
				err += fmt.Sprintf(" or one of the init containers: [%s]", initContainerNames)
			}
			return nil, nil, errors.NewBadRequest(err)
		}
	} else {
		if !podHasContainerWithName(pod, container) {
			return nil, nil, errors.NewBadRequest(fmt.Sprintf("container %s is not valid for pod %s", container, name))
		}
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
	return loc, nodeInfo.Transport, nil
}

func podHasContainerWithName(pod *api.Pod, containerName string) bool {
	for _, c := range pod.Spec.Containers {
		if c.Name == containerName {
			return true
		}
	}
	for _, c := range pod.Spec.InitContainers {
		if c.Name == containerName {
			return true
		}
	}
	return false
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
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	ctx context.Context,
	name string,
	opts *api.PodAttachOptions,
) (*url.URL, http.RoundTripper, error) {
	return streamLocation(getter, connInfo, ctx, name, opts, opts.Container, "attach")
}

// ExecLocation returns the exec URL for a pod container. If opts.Container is blank
// and only one container is present in the pod, that container is used.
func ExecLocation(
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	ctx context.Context,
	name string,
	opts *api.PodExecOptions,
) (*url.URL, http.RoundTripper, error) {
	return streamLocation(getter, connInfo, ctx, name, opts, opts.Container, "exec")
}

func streamLocation(
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	ctx context.Context,
	name string,
	opts runtime.Object,
	container,
	path string,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(getter, ctx, name)
	if err != nil {
		return nil, nil, err
	}

	// Try to figure out a container
	// If a container was provided, it must be valid
	if container == "" {
		switch len(pod.Spec.Containers) {
		case 1:
			container = pod.Spec.Containers[0].Name
		case 0:
			return nil, nil, errors.NewBadRequest(fmt.Sprintf("a container name must be specified for pod %s", name))
		default:
			containerNames := getContainerNames(pod.Spec.Containers)
			initContainerNames := getContainerNames(pod.Spec.InitContainers)
			err := fmt.Sprintf("a container name must be specified for pod %s, choose one of: [%s]", name, containerNames)
			if len(initContainerNames) > 0 {
				err += fmt.Sprintf(" or one of the init containers: [%s]", initContainerNames)
			}
			return nil, nil, errors.NewBadRequest(err)
		}
	} else {
		if !podHasContainerWithName(pod, container) {
			return nil, nil, errors.NewBadRequest(fmt.Sprintf("container %s is not valid for pod %s", container, name))
		}
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
	getter ResourceGetter,
	connInfo client.ConnectionInfoGetter,
	ctx context.Context,
	name string,
	opts *api.PodPortForwardOptions,
) (*url.URL, http.RoundTripper, error) {
	pod, err := getPod(getter, ctx, name)
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
