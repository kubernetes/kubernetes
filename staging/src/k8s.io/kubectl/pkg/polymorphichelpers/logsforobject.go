/*
Copyright 2018 The Kubernetes Authors.

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

package polymorphichelpers

import (
	"errors"
	"fmt"
	"os"
	"sort"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/reference"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/podutils"
)

// defaultLogsContainerAnnotationName is an annotation name that can be used to preselect the interesting container
// from a pod when running kubectl logs.
const defaultLogsContainerAnnotationName = "kubectl.kubernetes.io/default-logs-container"

func logsForObject(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool) (map[corev1.ObjectReference]rest.ResponseWrapper, error) {
	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	clientset, err := corev1client.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	return logsForObjectWithClient(clientset, object, options, timeout, allContainers)
}

// this is split for easy test-ability
func logsForObjectWithClient(clientset corev1client.CoreV1Interface, object, options runtime.Object, timeout time.Duration, allContainers bool) (map[corev1.ObjectReference]rest.ResponseWrapper, error) {
	opts, ok := options.(*corev1.PodLogOptions)
	if !ok {
		return nil, errors.New("provided options object is not a PodLogOptions")
	}

	switch t := object.(type) {
	case *corev1.PodList:
		ret := make(map[corev1.ObjectReference]rest.ResponseWrapper)
		for i := range t.Items {
			currRet, err := logsForObjectWithClient(clientset, &t.Items[i], options, timeout, allContainers)
			if err != nil {
				return nil, err
			}
			for k, v := range currRet {
				ret[k] = v
			}
		}
		return ret, nil

	case *corev1.Pod:
		// in case the "kubectl.kubernetes.io/default-logs-container" annotation is present, we preset the opts.Containers to default to selected
		// container. This gives users ability to preselect the most interesting container in pod.
		if annotations := t.GetAnnotations(); annotations != nil && len(opts.Container) == 0 && len(annotations[defaultLogsContainerAnnotationName]) > 0 {
			containerName := annotations[defaultLogsContainerAnnotationName]
			if exists, _ := findContainerByName(t, containerName); exists != nil {
				opts.Container = containerName
			} else {
				fmt.Fprintf(os.Stderr, "Default container name %q not found in a pod\n", containerName)
			}
		}
		// if allContainers is true, then we're going to locate all containers and then iterate through them. At that point, "allContainers" is false
		if !allContainers {
			var containerName string
			if opts == nil || len(opts.Container) == 0 {
				// We don't know container name. In this case we expect only one container to be present in the pod (ignoring InitContainers).
				// If there is more than one container we should return an error showing all container names.
				if len(t.Spec.Containers) != 1 {
					containerNames := getContainerNames(t.Spec.Containers)
					initContainerNames := getContainerNames(t.Spec.InitContainers)
					ephemeralContainerNames := getContainerNames(ephemeralContainersToContainers(t.Spec.EphemeralContainers))
					err := fmt.Sprintf("a container name must be specified for pod %s, choose one of: [%s]", t.Name, containerNames)
					if len(initContainerNames) > 0 {
						err += fmt.Sprintf(" or one of the init containers: [%s]", initContainerNames)
					}
					if len(ephemeralContainerNames) > 0 {
						err += fmt.Sprintf(" or one of the ephemeral containers: [%s]", ephemeralContainerNames)
					}

					return nil, errors.New(err)
				}
				containerName = t.Spec.Containers[0].Name
			} else {
				containerName = opts.Container
			}

			container, fieldPath := findContainerByName(t, containerName)
			if container == nil {
				return nil, fmt.Errorf("container %s is not valid for pod %s", opts.Container, t.Name)
			}
			ref, err := reference.GetPartialReference(scheme.Scheme, t, fieldPath)
			if err != nil {
				return nil, fmt.Errorf("Unable to construct reference to '%#v': %v", t, err)
			}

			ret := make(map[corev1.ObjectReference]rest.ResponseWrapper, 1)
			ret[*ref] = clientset.Pods(t.Namespace).GetLogs(t.Name, opts)
			return ret, nil
		}

		ret := make(map[corev1.ObjectReference]rest.ResponseWrapper)
		for _, c := range t.Spec.InitContainers {
			currOpts := opts.DeepCopy()
			currOpts.Container = c.Name
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false)
			if err != nil {
				return nil, err
			}
			for k, v := range currRet {
				ret[k] = v
			}
		}
		for _, c := range t.Spec.Containers {
			currOpts := opts.DeepCopy()
			currOpts.Container = c.Name
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false)
			if err != nil {
				return nil, err
			}
			for k, v := range currRet {
				ret[k] = v
			}
		}
		for _, c := range t.Spec.EphemeralContainers {
			currOpts := opts.DeepCopy()
			currOpts.Container = c.Name
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false)
			if err != nil {
				return nil, err
			}
			for k, v := range currRet {
				ret[k] = v
			}
		}

		return ret, nil
	}

	namespace, selector, err := SelectorsForObject(object)
	if err != nil {
		return nil, fmt.Errorf("cannot get the logs from %T: %v", object, err)
	}

	sortBy := func(pods []*corev1.Pod) sort.Interface { return podutils.ByLogging(pods) }
	pod, numPods, err := GetFirstPod(clientset, namespace, selector.String(), timeout, sortBy)
	if err != nil {
		return nil, err
	}
	if numPods > 1 {
		fmt.Fprintf(os.Stderr, "Found %v pods, using pod/%v\n", numPods, pod.Name)
	}

	return logsForObjectWithClient(clientset, pod, options, timeout, allContainers)
}

// findContainerByName searches for a container by name amongst all containers in a pod.
// Returns a pointer to a container and a field path.
func findContainerByName(pod *corev1.Pod, name string) (container *corev1.Container, fieldPath string) {
	for _, c := range pod.Spec.InitContainers {
		if c.Name == name {
			return &c, fmt.Sprintf("spec.initContainers{%s}", c.Name)
		}
	}
	for _, c := range pod.Spec.Containers {
		if c.Name == name {
			return &c, fmt.Sprintf("spec.containers{%s}", c.Name)
		}
	}
	for _, c := range pod.Spec.EphemeralContainers {
		if c.Name == name {
			containerCommon := corev1.Container(c.EphemeralContainerCommon)
			return &containerCommon, fmt.Sprintf("spec.ephemeralContainers{%s}", containerCommon.Name)
		}
	}
	return nil, ""
}

// getContainerNames returns a formatted string containing the container names
func getContainerNames(containers []corev1.Container) string {
	names := []string{}
	for _, c := range containers {
		names = append(names, c.Name)
	}
	return strings.Join(names, " ")
}

func ephemeralContainersToContainers(containers []corev1.EphemeralContainer) []corev1.Container {
	var ec []corev1.Container
	for i := range containers {
		ec = append(ec, corev1.Container(containers[i].EphemeralContainerCommon))
	}
	return ec
}
