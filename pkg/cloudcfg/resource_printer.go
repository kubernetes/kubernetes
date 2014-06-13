/*
Copyright 2014 Google Inc. All rights reserved.

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

package cloudcfg

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"text/tabwriter"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"gopkg.in/v1/yaml"
)

// ResourcePrinter is an interface that knows how to print API resources
type ResourcePrinter interface {
	// Print receives an arbitrary JSON body, formats it and prints it to a writer
	Print(string, io.Writer) error
}

// Identity printer simply copies the body out to the output stream
type IdentityPrinter struct{}

func (i *IdentityPrinter) Print(data string, w io.Writer) error {
	_, err := fmt.Fprint(w, data)
	return err
}

// YAMLPrinter parses JSON, and re-formats as YAML
type YAMLPrinter struct{}

func (y *YAMLPrinter) Print(data string, w io.Writer) error {
	var obj interface{}
	if err := json.Unmarshal([]byte(data), &obj); err != nil {
		return err
	}
	output, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(output))
	return err
}

// HumanReadablePrinter attempts to provide more elegant output
type HumanReadablePrinter struct{}

var podColumns = []string{"Name", "Image(s)", "Host", "Labels"}
var replicationControllerColumns = []string{"Name", "Image(s)", "Label Query", "Replicas"}
var serviceColumns = []string{"Name", "Label Query", "Port"}

func (h *HumanReadablePrinter) unknown(data string, w io.Writer) error {
	_, err := fmt.Fprintf(w, "Unknown object: %s", data)
	return err
}

func (h *HumanReadablePrinter) printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	var lines []string
	for _ = range columnNames {
		lines = append(lines, "----------")
	}
	_, err := fmt.Fprintf(w, "%s\n", strings.Join(lines, "\t"))
	return err
}

func (h *HumanReadablePrinter) makeImageList(manifest api.ContainerManifest) string {
	var images []string
	for _, container := range manifest.Containers {
		images = append(images, container.Image)
	}
	return strings.Join(images, ",")
}

func (h *HumanReadablePrinter) makeLabelsList(labels map[string]string) string {
	var vals []string
	for key, value := range labels {
		vals = append(vals, fmt.Sprintf("%s=%s", key, value))
	}
	return strings.Join(vals, ",")
}

func (h *HumanReadablePrinter) printPod(pod api.Pod, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n",
		pod.ID, h.makeImageList(pod.DesiredState.Manifest), pod.CurrentState.Host, h.makeLabelsList(pod.Labels))
	return err
}

func (h *HumanReadablePrinter) printPodList(podList api.PodList, w io.Writer) error {
	for _, pod := range podList.Items {
		if err := h.printPod(pod, w); err != nil {
			return err
		}
	}
	return nil
}

func (h *HumanReadablePrinter) printReplicationController(ctrl api.ReplicationController, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%d\n",
		ctrl.ID, h.makeImageList(ctrl.DesiredState.PodTemplate.DesiredState.Manifest), h.makeLabelsList(ctrl.DesiredState.ReplicasInSet), ctrl.DesiredState.Replicas)
	return err
}

func (h *HumanReadablePrinter) printReplicationControllerList(list api.ReplicationControllerList, w io.Writer) error {
	for _, ctrl := range list.Items {
		if err := h.printReplicationController(ctrl, w); err != nil {
			return err
		}
	}
	return nil
}

func (h *HumanReadablePrinter) printService(svc api.Service, w io.Writer) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%d\n", svc.ID, h.makeLabelsList(svc.Labels), svc.Port)
	return err
}

func (h *HumanReadablePrinter) printServiceList(list api.ServiceList, w io.Writer) error {
	for _, svc := range list.Items {
		if err := h.printService(svc, w); err != nil {
			return err
		}
	}
	return nil
}

// TODO replace this with something that returns a concrete printer object, rather than
//  having the secondary switch below.
func (h *HumanReadablePrinter) extractObject(data, kind string) (interface{}, error) {
	// TODO: I think this can be replaced with some reflection and a map[string]type
	switch kind {
	case "cluster#pod":
		var obj api.Pod
		if err := json.Unmarshal([]byte(data), &obj); err != nil {
			return nil, err
		}
		return obj, nil
	case "cluster#podList":
		var list api.PodList
		if err := json.Unmarshal([]byte(data), &list); err != nil {
			return nil, err
		}
		return list, nil
	case "cluster#replicationController":
		var ctrl api.ReplicationController
		if err := json.Unmarshal([]byte(data), &ctrl); err != nil {
			return nil, err
		}
		return ctrl, nil
	case "cluster#replicationControllerList":
		var list api.ReplicationControllerList
		if err := json.Unmarshal([]byte(data), &list); err != nil {
			return nil, err
		}
		return list, nil
	case "cluster#service":
		var ctrl api.Service
		if err := json.Unmarshal([]byte(data), &ctrl); err != nil {
			return nil, err
		}
		return ctrl, nil
	case "cluster#serviceList":
		var list api.ServiceList
		if err := json.Unmarshal([]byte(data), &list); err != nil {
			return nil, err
		}
		return list, nil
	default:
		return nil, fmt.Errorf("Unknown kind: %s", kind)
	}
}

func (h *HumanReadablePrinter) Print(data string, output io.Writer) error {
	w := tabwriter.NewWriter(output, 20, 5, 3, ' ', 0)
	defer w.Flush()
	var obj interface{}
	if err := json.Unmarshal([]byte(data), &obj); err != nil {
		return err
	}

	if _, contains := obj.(map[string]interface{})["kind"]; !contains {
		return fmt.Errorf("Unexpected object with no 'kind' field: %s", data)
	}
	kind := (obj.(map[string]interface{})["kind"]).(string)
	obj, err := h.extractObject(data, kind)
	if err != nil {
		return err
	}
	switch obj.(type) {
	case api.Pod:
		h.printHeader(podColumns, w)
		return h.printPod(obj.(api.Pod), w)
	case api.PodList:
		h.printHeader(podColumns, w)
		return h.printPodList(obj.(api.PodList), w)
	case api.ReplicationController:
		h.printHeader(replicationControllerColumns, w)
		return h.printReplicationController(obj.(api.ReplicationController), w)
	case api.ReplicationControllerList:
		h.printHeader(replicationControllerColumns, w)
		return h.printReplicationControllerList(obj.(api.ReplicationControllerList), w)
	case api.Service:
		h.printHeader(serviceColumns, w)
		return h.printService(obj.(api.Service), w)
	case api.ServiceList:
		h.printHeader(serviceColumns, w)
		return h.printServiceList(obj.(api.ServiceList), w)
	default:
		return h.unknown(data, w)
	}
}
