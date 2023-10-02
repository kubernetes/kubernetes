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
	"context"
	"errors"
	"fmt"
	"os"
	"sort"
	"time"

	corev1 "k8s.io/api/core/v1"
	errorsapi "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/client-go/rest"
	"k8s.io/client-go/tools/cache"
	"k8s.io/client-go/tools/reference"
	watchtools "k8s.io/client-go/tools/watch"
	"k8s.io/klog/v2"
	"k8s.io/kubectl/pkg/cmd/util/podcmd"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/interrupt"
	"k8s.io/kubectl/pkg/util/podutils"
)

func logsForObject(restClientGetter genericclioptions.RESTClientGetter, object, options runtime.Object, timeout time.Duration, allContainers bool, wait bool) (map[corev1.ObjectReference]rest.ResponseWrapper, error) {
	clientConfig, err := restClientGetter.ToRESTConfig()
	if err != nil {
		return nil, err
	}

	clientset, err := corev1client.NewForConfig(clientConfig)
	if err != nil {
		return nil, err
	}
	return logsForObjectWithClient(clientset, object, options, timeout, allContainers, wait)
}

// this is split for easy test-ability
func logsForObjectWithClient(clientset corev1client.CoreV1Interface, object, options runtime.Object, timeout time.Duration, allContainers bool, wait bool) (map[corev1.ObjectReference]rest.ResponseWrapper, error) {
	opts, ok := options.(*corev1.PodLogOptions)
	if !ok {
		return nil, errors.New("provided options object is not a PodLogOptions")
	}

	switch t := object.(type) {
	case *corev1.PodList:
		ret := make(map[corev1.ObjectReference]rest.ResponseWrapper)
		for i := range t.Items {
			currRet, err := logsForObjectWithClient(clientset, &t.Items[i], options, timeout, allContainers, wait)
			if err != nil {
				return nil, err
			}
			for k, v := range currRet {
				ret[k] = v
			}
		}
		return ret, nil

	case *corev1.Pod:
		// if allContainers is true, then we're going to locate all containers and then iterate through them. At that point, "allContainers" is false
		if !allContainers {
			currOpts := new(corev1.PodLogOptions)
			if opts != nil {
				opts.DeepCopyInto(currOpts)
			}
			// in case the "kubectl.kubernetes.io/default-container" annotation is present, we preset the opts.Containers to default to selected
			// container. This gives users ability to preselect the most interesting container in pod.
			if annotations := t.GetAnnotations(); annotations != nil && currOpts.Container == "" {
				var defaultContainer string
				if len(annotations[podcmd.DefaultContainerAnnotationName]) > 0 {
					defaultContainer = annotations[podcmd.DefaultContainerAnnotationName]
				}
				if len(defaultContainer) > 0 {
					if exists, _ := podcmd.FindContainerByName(t, defaultContainer); exists == nil {
						fmt.Fprintf(os.Stderr, "Default container name %q not found in pod %s\n", defaultContainer, t.Name)
					} else {
						currOpts.Container = defaultContainer
					}
				}
			}

			if currOpts.Container == "" {
				// Default to the first container name(aligning behavior with `kubectl exec').
				currOpts.Container = t.Spec.Containers[0].Name
				if len(t.Spec.Containers) > 1 || len(t.Spec.InitContainers) > 0 || len(t.Spec.EphemeralContainers) > 0 {
					fmt.Fprintf(os.Stderr, "Defaulted container %q out of: %s\n", currOpts.Container, podcmd.AllContainerNames(t))
				}
			}
			container, fieldPath := podcmd.FindContainerByName(t, currOpts.Container)
			if container == nil {
				return nil, fmt.Errorf("container %s is not valid for pod %s", currOpts.Container, t.Name)
			}
			ref, err := reference.GetPartialReference(scheme.Scheme, t, fieldPath)
			if err != nil {
				return nil, fmt.Errorf("Unable to construct reference to '%#v': %v", t, err)
			}

			if wait {
				errFromContainer := waitForContainer(clientset, t.Namespace, t.Name, currOpts.Container)
				if errFromContainer != nil {
					return nil, errFromContainer
				}
			}

			ret := make(map[corev1.ObjectReference]rest.ResponseWrapper, 1)
			ret[*ref] = clientset.Pods(t.Namespace).GetLogs(t.Name, currOpts)
			return ret, nil
		}

		ret := make(map[corev1.ObjectReference]rest.ResponseWrapper)
		for _, c := range t.Spec.InitContainers {
			currOpts := opts.DeepCopy()
			currOpts.Container = c.Name
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false, wait)
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
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false, wait)
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
			currRet, err := logsForObjectWithClient(clientset, t, currOpts, timeout, false, wait)
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

	return logsForObjectWithClient(clientset, pod, options, timeout, allContainers, wait)
}

// waitForContainer watches the given pod until the container is running
func waitForContainer(clientset corev1client.CoreV1Interface, ns string, podName string, containerName string) error {
	// TODO: expose the timeout
	ctx, cancel := watchtools.ContextWithOptionalTimeout(context.Background(), 0*time.Second)
	defer cancel()

	fieldSelector := fields.OneTermEqualSelector("metadata.name", podName).String()
	lw := &cache.ListWatch{
		ListFunc: func(options metav1.ListOptions) (runtime.Object, error) {
			options.FieldSelector = fieldSelector
			return clientset.Pods(ns).List(ctx, options)
		},
		WatchFunc: func(options metav1.ListOptions) (watch.Interface, error) {
			options.FieldSelector = fieldSelector
			return clientset.Pods(ns).Watch(ctx, options)
		},
	}

	intr := interrupt.New(nil, cancel)
	err := intr.Run(func() error {
		_, err := watchtools.UntilWithSync(ctx, lw, &corev1.Pod{}, nil, func(ev watch.Event) (bool, error) {
			klog.V(2).Infof("watch received event %q with object %T", ev.Type, ev.Object)
			if ev.Type == watch.Deleted {
				return false, errorsapi.NewNotFound(schema.GroupResource{Resource: "pods"}, "")
			}

			p, ok := ev.Object.(*corev1.Pod)
			if !ok {
				return false, fmt.Errorf("watch did not return a pod: %v", ev.Object)
			}

			s := podcmd.GetContainerStatusByName(p, containerName)
			if s == nil {
				return false, nil
			}
			klog.V(2).Infof("debug container status is %v", s)
			if s.State.Running != nil || s.State.Terminated != nil {
				return true, nil
			}
			if s.State.Waiting != nil && s.State.Waiting.Message != "" {
				klog.V(2).Infof("container %s: %s", containerName, s.State.Waiting.Message)
			}
			return false, nil
		})
		return err
	})

	return err
}
