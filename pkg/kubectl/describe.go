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
	"strings"
	"text/tabwriter"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

func Describe(w io.Writer, c client.Interface, resource, id string) error {
	var str string
	var err error
	path, err := resolveResource(resolveToPath, resource)
	if err != nil {
		return err
	}
	switch path {
	case "pods":
		str, err = describePod(w, c, id)
	case "replicationControllers":
		str, err = describeReplicationController(w, c, id)
	case "services":
		str, err = describeService(w, c, id)
	case "minions":
		str, err = describeMinion(w, c, id)
	}

	if err != nil {
		return err
	}

	_, err = fmt.Fprintf(w, str)
	return err
}

func describePod(w io.Writer, c client.Interface, id string) (string, error) {
	pod, err := c.GetPod(api.NewDefaultContext(), id)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out *tabwriter.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pod.Name)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(pod.DesiredState.Manifest))
		fmt.Fprintf(out, "Host:\t%s\n", pod.CurrentState.Host+"/"+pod.CurrentState.HostIP)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(pod.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", string(pod.CurrentState.Status))
		fmt.Fprintf(out, "Replication Controllers:\t%s\n", getReplicationControllersForLabels(c, labels.Set(pod.Labels)))
		return nil
	})
}

func describeReplicationController(w io.Writer, c client.Interface, id string) (string, error) {
	controller, err := c.GetReplicationController(api.NewDefaultContext(), id)
	if err != nil {
		return "", err
	}

	running, waiting, terminated, err := getPodStatusForReplicationController(c, controller)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out *tabwriter.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", controller.Name)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(controller.DesiredState.PodTemplate.DesiredState.Manifest))
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(controller.DesiredState.ReplicaSelector))
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(controller.Labels))
		fmt.Fprintf(out, "Replicas:\t%d current / %d desired\n", controller.CurrentState.Replicas, controller.DesiredState.Replicas)
		fmt.Fprintf(out, "Pods Status:\t%d Running / %d Waiting / %d Terminated\n", running, waiting, terminated)
		return nil
	})
}

func describeService(w io.Writer, c client.Interface, id string) (string, error) {
	service, err := c.GetService(api.NewDefaultContext(), id)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out *tabwriter.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", service.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(service.Labels))
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(service.Selector))
		fmt.Fprintf(out, "Port:\t%d\n", service.Port)
		return nil
	})
}

func describeMinion(w io.Writer, c client.Interface, id string) (string, error) {
	minion, err := getMinion(c, id)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out *tabwriter.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", minion.Name)
		return nil
	})
}

// client.Interface doesn't have GetMinion(id) yet so we hack it up.
func getMinion(c client.Interface, id string) (*api.Minion, error) {
	minionList, err := c.ListMinions()
	if err != nil {
		glog.Fatalf("Error getting minion info: %v\n", err)
	}

	for _, minion := range minionList.Items {
		if id == minion.Name {
			return &minion, nil
		}
	}
	return nil, fmt.Errorf("Minion %s not found", id)
}

// Get all replication controllers whose selectors would match a given set of
// labels.
// TODO Move this to pkg/client and ideally implement it server-side (instead
// of getting all RC's and searching through them manually).
func getReplicationControllersForLabels(c client.Interface, labelsToMatch labels.Labels) string {
	// Get all replication controllers.
	rcs, err := c.ListReplicationControllers(api.NewDefaultContext(), labels.Everything())
	if err != nil {
		glog.Fatalf("Error getting replication controllers: %v\n", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingRCs []api.ReplicationController
	for _, controller := range rcs.Items {
		selector := labels.SelectorFromSet(controller.DesiredState.ReplicaSelector)
		if selector.Matches(labelsToMatch) {
			matchingRCs = append(matchingRCs, controller)
		}
	}

	// Format the matching RC's into strings.
	var rcStrings []string
	for _, controller := range matchingRCs {
		rcStrings = append(rcStrings, fmt.Sprintf("%s (%d/%d replicas created)", controller.Name, controller.CurrentState.Replicas, controller.DesiredState.Replicas))
	}

	list := strings.Join(rcStrings, ", ")
	if list == "" {
		return "<none>"
	}
	return list
}

func getPodStatusForReplicationController(kubeClient client.Interface, controller *api.ReplicationController) (running, waiting, terminated int, err error) {
	rcPods, err := kubeClient.ListPods(api.NewDefaultContext(), labels.SelectorFromSet(controller.DesiredState.ReplicaSelector))
	if err != nil {
		return
	}
	for _, pod := range rcPods.Items {
		if pod.CurrentState.Status == api.PodRunning {
			running++
		} else if pod.CurrentState.Status == api.PodWaiting {
			waiting++
		} else if pod.CurrentState.Status == api.PodTerminated {
			terminated++
		}
	}
	return
}
