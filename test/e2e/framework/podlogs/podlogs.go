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

// Package podlogs enables live capturing of all events and log
// messages for some or all pods in a namespace as they get generated.
// This helps debugging both a running test (what is currently going
// on?) and the output of a CI run (events appear in chronological
// order and output that normally isn't available like the command
// stdout messages are available).
package podlogs

import (
	"bytes"
	"context"
	"fmt"
	"github.com/pkg/errors"
	"io"

	"k8s.io/api/core/v1"
	meta "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
)

// LogsForPod starts reading the logs for a certain pod. If the pod has more than one
// container, opts.Container must be set. Reading stops when the context is done.
// The stream includes formatted error messages and ends with
//    rpc error: code = Unknown desc = Error: No such container: 41a...
// when the pod gets deleted while streaming.
func LogsForPod(ctx context.Context, cs clientset.Interface, ns, pod string, opts *v1.PodLogOptions) (io.ReadCloser, error) {
	req := cs.CoreV1().Pods(ns).GetLogs(pod, opts)
	return req.Context(ctx).Stream()
}

// WatchPods prints pod status events for a certain namespace or all namespaces
// when namespace name is empty.
func WatchPods(ctx context.Context, cs clientset.Interface, ns string, to io.Writer) error {
	watcher, err := cs.CoreV1().Pods(ns).Watch(meta.ListOptions{})
	if err != nil {
		return errors.Wrap(err, "cannot create Pod event watcher")
	}

	go func() {
		defer watcher.Stop()
		for {
			select {
			case e := <-watcher.ResultChan():
				if e.Object == nil {
					continue
				}

				pod, ok := e.Object.(*v1.Pod)
				if !ok {
					continue
				}
				buffer := new(bytes.Buffer)
				fmt.Fprintf(buffer,
					"pod event: %s: %s/%s %s: %s %s\n",
					e.Type,
					pod.Namespace,
					pod.Name,
					pod.Status.Phase,
					pod.Status.Reason,
					pod.Status.Conditions,
				)
				for _, cst := range pod.Status.ContainerStatuses {
					fmt.Fprintf(buffer, "   %s: ", cst.Name)
					if cst.State.Waiting != nil {
						fmt.Fprintf(buffer, "WAITING: %s - %s",
							cst.State.Waiting.Reason,
							cst.State.Waiting.Message,
						)
					} else if cst.State.Running != nil {
						fmt.Fprintf(buffer, "RUNNING")
					} else if cst.State.Terminated != nil {
						fmt.Fprintf(buffer, "TERMINATED: %s - %s",
							cst.State.Terminated.Reason,
							cst.State.Terminated.Message,
						)
					}
					fmt.Fprintf(buffer, "\n")
				}
				to.Write(buffer.Bytes())
			case <-ctx.Done():
				return
			}
		}
	}()

	return nil
}
