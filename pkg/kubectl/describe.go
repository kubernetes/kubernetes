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

package kubectl

import (
	"fmt"
	"io"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

// Describer generates output for the named resource or an error
// if the output could not be generated.
type Describer interface {
	Describe(namespace, name string) (output string, err error)
}

// Describer returns the default describe functions for each of the standard
// Kubernetes types.
func DescriberFor(kind string, c *client.Client) (Describer, bool) {
	switch kind {
	case "Pod":
		return &PodDescriber{c}, true
	case "ReplicationController":
		return &ReplicationControllerDescriber{c}, true
	case "Service":
		return &ServiceDescriber{c}, true
	case "Minion", "Node":
		return &MinionDescriber{c}, true
	case "LimitRange":
		return &LimitRangeDescriber{c}, true
	case "ResourceQuota":
		return &ResourceQuotaDescriber{c}, true
	}
	return nil, false
}

// LimitRangeDescriber generates information about a limit range
type LimitRangeDescriber struct {
	client.Interface
}

func (d *LimitRangeDescriber) Describe(namespace, name string) (string, error) {
	lr := d.LimitRanges(namespace)

	limitRange, err := lr.Get(name)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", limitRange.Name)
		fmt.Fprintf(out, "Type\tResource\tMin\tMax\n")
		fmt.Fprintf(out, "----\t--------\t---\t---\n")
		for i := range limitRange.Spec.Limits {
			item := limitRange.Spec.Limits[i]
			maxResources := item.Max
			minResources := item.Min

			set := map[api.ResourceName]bool{}
			for k := range maxResources {
				set[k] = true
			}
			for k := range minResources {
				set[k] = true
			}

			for k := range set {
				// if no value is set, we output -
				maxValue := "-"
				minValue := "-"

				maxQuantity, maxQuantityFound := maxResources[k]
				if maxQuantityFound {
					maxValue = maxQuantity.String()
				}

				minQuantity, minQuantityFound := minResources[k]
				if minQuantityFound {
					minValue = minQuantity.String()
				}

				msg := "%v\t%v\t%v\t%v\n"
				fmt.Fprintf(out, msg, item.Type, k, minValue, maxValue)
			}
		}
		return nil
	})
}

// ResourceQuotaDescriber generates information about a resource quota
type ResourceQuotaDescriber struct {
	client.Interface
}

func (d *ResourceQuotaDescriber) Describe(namespace, name string) (string, error) {
	rq := d.ResourceQuotas(namespace)

	resourceQuota, err := rq.Get(name)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", resourceQuota.Name)
		fmt.Fprintf(out, "Resource\tUsed\tHard\n")
		fmt.Fprintf(out, "--------\t----\t----\n")

		resources := []api.ResourceName{}
		for resource := range resourceQuota.Status.Hard {
			resources = append(resources, resource)
		}
		sort.Sort(SortableResourceNames(resources))

		msg := "%v\t%v\t%v\n"
		for i := range resources {
			resource := resources[i]
			hardQuantity := resourceQuota.Status.Hard[resource]
			usedQuantity := resourceQuota.Status.Used[resource]
			fmt.Fprintf(out, msg, resource, usedQuantity.String(), hardQuantity.String())
		}
		return nil
	})
}

// PodDescriber generates information about a pod and the replication controllers that
// create it.
type PodDescriber struct {
	client.Interface
}

func (d *PodDescriber) Describe(namespace, name string) (string, error) {
	rc := d.ReplicationControllers(namespace)
	pc := d.Pods(namespace)

	pod, err := pc.Get(name)
	if err != nil {
		events, err2 := d.Events(namespace).List(
			labels.Everything(),
			labels.Set{
				"involvedObject.name":      name,
				"involvedObject.namespace": namespace,
			}.AsSelector(),
		)
		if err2 == nil && len(events.Items) > 0 {
			return tabbedString(func(out io.Writer) error {
				fmt.Fprintf(out, "Pod '%v': error '%v', but found events.\n", name, err)
				describeEvents(events, out)
				return nil
			})
		}
		return "", err
	}

	// TODO: remove me when pods are converted
	spec := &api.PodSpec{}
	if err := api.Scheme.Convert(&pod.Spec, spec); err != nil {
		glog.Errorf("Unable to convert pod manifest: %v", err)
	}

	var events *api.EventList
	if ref, err := api.GetReference(pod); err != nil {
		glog.Errorf("Unable to construct reference to '%#v': %v", pod, err)
	} else {
		ref.Kind = "" // Find BoundPod objects, too!
		events, _ = d.Events(namespace).Search(ref)
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pod.Name)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(spec))
		fmt.Fprintf(out, "Host:\t%s\n", pod.Status.Host+"/"+pod.Status.HostIP)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(pod.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", string(pod.Status.Phase))
		fmt.Fprintf(out, "Replication Controllers:\t%s\n", getReplicationControllersForLabels(rc, labels.Set(pod.Labels)))
		if len(pod.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Kind\tStatus\n")
			for _, c := range pod.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \n",
					c.Kind,
					c.Status)
			}
		}
		if events != nil {
			describeEvents(events, out)
		}
		return nil
	})
}

// ReplicationControllerDescriber generates information about a replication controller
// and the pods it has created.
type ReplicationControllerDescriber struct {
	client.Interface
}

func (d *ReplicationControllerDescriber) Describe(namespace, name string) (string, error) {
	rc := d.ReplicationControllers(namespace)
	pc := d.Pods(namespace)

	controller, err := rc.Get(name)
	if err != nil {
		return "", err
	}

	running, waiting, succeeded, failed, err := getPodStatusForReplicationController(pc, controller)
	if err != nil {
		return "", err
	}

	events, _ := d.Events(namespace).Search(controller)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", controller.Name)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&controller.Spec.Template.Spec))
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(controller.Spec.Selector))
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(controller.Labels))
		fmt.Fprintf(out, "Replicas:\t%d current / %d desired\n", controller.Status.Replicas, controller.Spec.Replicas)
		fmt.Fprintf(out, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		if events != nil {
			describeEvents(events, out)
		}
		return nil
	})
}

// ServiceDescriber generates information about a service.
type ServiceDescriber struct {
	client.Interface
}

func (d *ServiceDescriber) Describe(namespace, name string) (string, error) {
	c := d.Services(namespace)

	service, err := c.Get(name)
	if err != nil {
		return "", err
	}

	endpoints, err := d.Endpoints(namespace).Get(name)
	if err != nil {
		endpoints = &api.Endpoints{}
	}

	events, _ := d.Events(namespace).Search(service)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", service.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(service.Labels))
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(service.Spec.Selector))
		fmt.Fprintf(out, "Port:\t%d\n", service.Spec.Port)
		fmt.Fprintf(out, "Endpoints:\t%s\n", stringList(endpoints.Endpoints))
		if events != nil {
			describeEvents(events, out)
		}
		return nil
	})
}

// MinionDescriber generates information about a minion.
type MinionDescriber struct {
	client.Interface
}

func (d *MinionDescriber) Describe(namespace, name string) (string, error) {
	mc := d.Nodes()
	minion, err := mc.Get(name)
	if err != nil {
		return "", err
	}

	events, _ := d.Events(namespace).Search(minion)

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", minion.Name)
		if len(minion.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Kind\tStatus\tLastProbeTime\tLastTransitionTime\tReason\tMessage\n")
			for _, c := range minion.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \t%s \t%s \t%v \t%v\n",
					c.Kind,
					c.Status,
					c.LastProbeTime.Time.Format(time.RFC1123Z),
					c.LastTransitionTime.Time.Format(time.RFC1123Z),
					c.Reason,
					c.Message)
			}
		}
		if events != nil {
			describeEvents(events, out)
		}
		return nil
	})
}

func describeEvents(el *api.EventList, w io.Writer) {
	if len(el.Items) == 0 {
		fmt.Fprint(w, "No events.")
		return
	}
	sort.Sort(SortableEvents(el.Items))
	fmt.Fprint(w, "Events:\nFirstSeen\tLastSeen\tCount\tFrom\tSubobjectPath\tReason\tMessage\n")
	for _, e := range el.Items {
		fmt.Fprintf(w, "%s\t%s\t%d\t%v\t%v\t%v\t%v\n",
			e.FirstTimestamp.Time.Format(time.RFC1123Z),
			e.LastTimestamp.Time.Format(time.RFC1123Z),
			e.Count,
			e.Source,
			e.InvolvedObject.FieldPath,
			e.Reason,
			e.Message)
	}
}

// Get all replication controllers whose selectors would match a given set of
// labels.
// TODO Move this to pkg/client and ideally implement it server-side (instead
// of getting all RC's and searching through them manually).
func getReplicationControllersForLabels(c client.ReplicationControllerInterface, labelsToMatch labels.Labels) string {
	// Get all replication controllers.
	// TODO this needs a namespace scope as argument
	rcs, err := c.List(labels.Everything())
	if err != nil {
		glog.Fatalf("Error getting replication controllers: %v\n", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingRCs []api.ReplicationController
	for _, controller := range rcs.Items {
		selector := labels.SelectorFromSet(controller.Spec.Selector)
		if selector.Matches(labelsToMatch) {
			matchingRCs = append(matchingRCs, controller)
		}
	}

	// Format the matching RC's into strings.
	var rcStrings []string
	for _, controller := range matchingRCs {
		rcStrings = append(rcStrings, fmt.Sprintf("%s (%d/%d replicas created)", controller.Name, controller.Status.Replicas, controller.Spec.Replicas))
	}

	list := strings.Join(rcStrings, ", ")
	if list == "" {
		return "<none>"
	}
	return list
}

func getPodStatusForReplicationController(c client.PodInterface, controller *api.ReplicationController) (running, waiting, succeeded, failed int, err error) {
	rcPods, err := c.List(labels.SelectorFromSet(controller.Spec.Selector))
	if err != nil {
		return
	}
	for _, pod := range rcPods.Items {
		switch pod.Status.Phase {
		case api.PodRunning:
			running++
		case api.PodPending:
			waiting++
		case api.PodSucceeded:
			succeeded++
		case api.PodFailed:
			failed++
		}
	}
	return
}
