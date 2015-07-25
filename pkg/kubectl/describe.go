/*
Copyright 2014 The Kubernetes Authors All rights reserved.

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
	"bytes"
	"fmt"
	"io"
	"reflect"
	"sort"
	"strings"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fieldpath"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

// Describer generates output for the named resource or an error
// if the output could not be generated. Implementors typically
// abstract the retrieval of the named object from a remote server.
type Describer interface {
	Describe(namespace, name string) (output string, err error)
}

// ObjectDescriber is an interface for displaying arbitrary objects with extra
// information. Use when an object is in hand (on disk, or already retrieved).
// Implementors may ignore the additional information passed on extra, or use it
// by default. ObjectDescribers may return ErrNoDescriber if no suitable describer
// is found.
type ObjectDescriber interface {
	DescribeObject(object interface{}, extra ...interface{}) (output string, err error)
}

// ErrNoDescriber is a structured error indicating the provided object or objects
// cannot be described.
type ErrNoDescriber struct {
	Types []string
}

// Error implements the error interface.
func (e ErrNoDescriber) Error() string {
	return fmt.Sprintf("no describer has been defined for %v", e.Types)
}

func describerMap(c *client.Client) map[string]Describer {
	m := map[string]Describer{
		"Pod": &PodDescriber{c},
		"ReplicationController": &ReplicationControllerDescriber{c},
		"Secret":                &SecretDescriber{c},
		"Service":               &ServiceDescriber{c},
		"ServiceAccount":        &ServiceAccountDescriber{c},
		"Minion":                &NodeDescriber{c},
		"Node":                  &NodeDescriber{c},
		"LimitRange":            &LimitRangeDescriber{c},
		"ResourceQuota":         &ResourceQuotaDescriber{c},
		"PersistentVolume":      &PersistentVolumeDescriber{c},
		"PersistentVolumeClaim": &PersistentVolumeClaimDescriber{c},
		"Namespace":             &NamespaceDescriber{c},
	}
	return m
}

// List of all resource types we can describe
func DescribableResources() []string {
	keys := make([]string, 0)

	for k := range describerMap(nil) {
		resource := strings.ToLower(k)
		keys = append(keys, resource)
	}
	return keys
}

// Describer returns the default describe functions for each of the standard
// Kubernetes types.
func DescriberFor(kind string, c *client.Client) (Describer, bool) {
	f, ok := describerMap(c)[kind]
	if ok {
		return f, true
	}
	return nil, false
}

// DefaultObjectDescriber can describe the default Kubernetes objects.
var DefaultObjectDescriber ObjectDescriber

func init() {
	d := &Describers{}
	err := d.Add(
		describeLimitRange,
		describeQuota,
		describePod,
		describeService,
		describeReplicationController,
		describeNode,
		describeNamespace,
	)
	if err != nil {
		glog.Fatalf("Cannot register describers: %v", err)
	}
	DefaultObjectDescriber = d
}

// NamespaceDescriber generates information about a namespace
type NamespaceDescriber struct {
	client.Interface
}

func (d *NamespaceDescriber) Describe(namespace, name string) (string, error) {
	ns, err := d.Namespaces().Get(name)
	if err != nil {
		return "", err
	}
	resourceQuotaList, err := d.ResourceQuotas(name).List(labels.Everything())
	if err != nil {
		return "", err
	}
	limitRangeList, err := d.LimitRanges(name).List(labels.Everything())
	if err != nil {
		return "", err
	}

	return describeNamespace(ns, resourceQuotaList, limitRangeList)
}

func describeNamespace(namespace *api.Namespace, resourceQuotaList *api.ResourceQuotaList, limitRangeList *api.LimitRangeList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", namespace.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(namespace.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", string(namespace.Status.Phase))
		if resourceQuotaList != nil {
			fmt.Fprintf(out, "\n")
			DescribeResourceQuotas(resourceQuotaList, out)
		}
		if limitRangeList != nil {
			fmt.Fprintf(out, "\n")
			DescribeLimitRanges(limitRangeList, out)
		}
		return nil
	})
}

// DescribeLimitRanges merges a set of limit range items into a single tabular description
func DescribeLimitRanges(limitRanges *api.LimitRangeList, w io.Writer) {
	if len(limitRanges.Items) == 0 {
		fmt.Fprint(w, "No resource limits.\n")
		return
	}
	fmt.Fprintf(w, "Resource Limits\n Type\tResource\tMin\tMax\tDefault\n")
	fmt.Fprintf(w, " ----\t--------\t---\t---\t---\n")
	for _, limitRange := range limitRanges.Items {
		for i := range limitRange.Spec.Limits {
			item := limitRange.Spec.Limits[i]
			maxResources := item.Max
			minResources := item.Min
			defaultResources := item.Default

			set := map[api.ResourceName]bool{}
			for k := range maxResources {
				set[k] = true
			}
			for k := range minResources {
				set[k] = true
			}
			for k := range defaultResources {
				set[k] = true
			}

			for k := range set {
				// if no value is set, we output -
				maxValue := "-"
				minValue := "-"
				defaultValue := "-"

				maxQuantity, maxQuantityFound := maxResources[k]
				if maxQuantityFound {
					maxValue = maxQuantity.String()
				}

				minQuantity, minQuantityFound := minResources[k]
				if minQuantityFound {
					minValue = minQuantity.String()
				}

				defaultQuantity, defaultQuantityFound := defaultResources[k]
				if defaultQuantityFound {
					defaultValue = defaultQuantity.String()
				}

				msg := " %v\t%v\t%v\t%v\t%v\n"
				fmt.Fprintf(w, msg, item.Type, k, minValue, maxValue, defaultValue)
			}
		}
	}
}

// DescribeResourceQuotas merges a set of quota items into a single tabular description of all quotas
func DescribeResourceQuotas(quotas *api.ResourceQuotaList, w io.Writer) {
	if len(quotas.Items) == 0 {
		fmt.Fprint(w, "No resource quota.\n")
		return
	}
	resources := []api.ResourceName{}
	hard := map[api.ResourceName]resource.Quantity{}
	used := map[api.ResourceName]resource.Quantity{}
	for _, q := range quotas.Items {
		for resource := range q.Status.Hard {
			resources = append(resources, resource)
			hardQuantity := q.Status.Hard[resource]
			usedQuantity := q.Status.Used[resource]

			// if for some reason there are multiple quota documents, we take least permissive
			prevQuantity, ok := hard[resource]
			if ok {
				if hardQuantity.Value() < prevQuantity.Value() {
					hard[resource] = hardQuantity
				}
			} else {
				hard[resource] = hardQuantity
			}
			used[resource] = usedQuantity
		}
	}

	sort.Sort(SortableResourceNames(resources))
	fmt.Fprint(w, "Resource Quotas\n Resource\tUsed\tHard\n")
	fmt.Fprint(w, " ---\t---\t---\n")
	for _, resource := range resources {
		hardQuantity := hard[resource]
		usedQuantity := used[resource]
		fmt.Fprintf(w, " %s\t%s\t%s\n", string(resource), usedQuantity.String(), hardQuantity.String())
	}
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
	return describeLimitRange(limitRange)
}

func describeLimitRange(limitRange *api.LimitRange) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", limitRange.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", limitRange.Namespace)
		fmt.Fprintf(out, "Type\tResource\tMin\tMax\tDefault\n")
		fmt.Fprintf(out, "----\t--------\t---\t---\t---\n")
		for i := range limitRange.Spec.Limits {
			item := limitRange.Spec.Limits[i]
			maxResources := item.Max
			minResources := item.Min
			defaultResources := item.Default

			set := map[api.ResourceName]bool{}
			for k := range maxResources {
				set[k] = true
			}
			for k := range minResources {
				set[k] = true
			}
			for k := range defaultResources {
				set[k] = true
			}

			for k := range set {
				// if no value is set, we output -
				maxValue := "-"
				minValue := "-"
				defaultValue := "-"

				maxQuantity, maxQuantityFound := maxResources[k]
				if maxQuantityFound {
					maxValue = maxQuantity.String()
				}

				minQuantity, minQuantityFound := minResources[k]
				if minQuantityFound {
					minValue = minQuantity.String()
				}

				defaultQuantity, defaultQuantityFound := defaultResources[k]
				if defaultQuantityFound {
					defaultValue = defaultQuantity.String()
				}

				msg := "%v\t%v\t%v\t%v\t%v\n"
				fmt.Fprintf(out, msg, item.Type, k, minValue, maxValue, defaultValue)
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

	return describeQuota(resourceQuota)
}

func describeQuota(resourceQuota *api.ResourceQuota) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", resourceQuota.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", resourceQuota.Namespace)
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
		eventsInterface := d.Events(namespace)
		events, err2 := eventsInterface.List(
			labels.Everything(),
			eventsInterface.GetFieldSelector(&name, &namespace, nil, nil))
		if err2 == nil && len(events.Items) > 0 {
			return tabbedString(func(out io.Writer) error {
				fmt.Fprintf(out, "Pod '%v': error '%v', but found events.\n", name, err)
				DescribeEvents(events, out)
				return nil
			})
		}
		return "", err
	}

	var events *api.EventList
	if ref, err := api.GetReference(pod); err != nil {
		glog.Errorf("Unable to construct reference to '%#v': %v", pod, err)
	} else {
		ref.Kind = ""
		events, _ = d.Events(namespace).Search(ref)
	}

	rcs, err := getReplicationControllersForLabels(rc, labels.Set(pod.Labels))
	if err != nil {
		return "", err
	}

	return describePod(pod, rcs, events)
}

func describePod(pod *api.Pod, rcs []api.ReplicationController, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pod.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", pod.Namespace)
		fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&pod.Spec))
		fmt.Fprintf(out, "Node:\t%s\n", pod.Spec.NodeName+"/"+pod.Status.HostIP)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(pod.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", string(pod.Status.Phase))
		fmt.Fprintf(out, "Reason:\t%s\n", pod.Status.Reason)
		fmt.Fprintf(out, "Message:\t%s\n", pod.Status.Message)
		fmt.Fprintf(out, "IP:\t%s\n", pod.Status.PodIP)
		fmt.Fprintf(out, "Replication Controllers:\t%s\n", printReplicationControllersByLabels(rcs))
		fmt.Fprintf(out, "Containers:\n")
		describeContainers(pod, out)
		if len(pod.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Type\tStatus\n")
			for _, c := range pod.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \n",
					c.Type,
					c.Status)
			}
		}
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

type PersistentVolumeDescriber struct {
	client.Interface
}

func (d *PersistentVolumeDescriber) Describe(namespace, name string) (string, error) {
	c := d.PersistentVolumes()

	pv, err := c.Get(name)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pv.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(pv.Labels))
		fmt.Fprintf(out, "Status:\t%s\n", pv.Status.Phase)
		if pv.Spec.ClaimRef != nil {
			fmt.Fprintf(out, "Claim:\t%s\n", pv.Spec.ClaimRef.Namespace+"/"+pv.Spec.ClaimRef.Name)
		} else {
			fmt.Fprintf(out, "Claim:\t%s\n", "")
		}
		fmt.Fprintf(out, "Reclaim Policy:\t%d\n", pv.Spec.PersistentVolumeReclaimPolicy)
		fmt.Fprintf(out, "Message:\t%d\n", pv.Status.Message)
		return nil
	})
}

type PersistentVolumeClaimDescriber struct {
	client.Interface
}

func (d *PersistentVolumeClaimDescriber) Describe(namespace, name string) (string, error) {
	c := d.PersistentVolumeClaims(namespace)

	pvc, err := c.Get(name)
	if err != nil {
		return "", err
	}

	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", pvc.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", pvc.Namespace)
		fmt.Fprintf(out, "Status:\t%d\n", pvc.Status.Phase)
		fmt.Fprintf(out, "Volume:\t%d\n", pvc.Spec.VolumeName)

		return nil
	})
}

func describeContainers(pod *api.Pod, out io.Writer) {
	statuses := map[string]api.ContainerStatus{}
	for _, status := range pod.Status.ContainerStatuses {
		statuses[status.Name] = status
	}

	for _, container := range pod.Spec.Containers {
		status := statuses[container.Name]
		state := status.State

		fmt.Fprintf(out, "  %v:\n", container.Name)
		fmt.Fprintf(out, "    Image:\t%s\n", container.Image)

		if len(container.Resources.Limits) > 0 {
			fmt.Fprintf(out, "    Limits:\n")
		}
		for name, quantity := range container.Resources.Limits {
			fmt.Fprintf(out, "      %s:\t%s\n", name, quantity.String())
		}

		switch {
		case state.Running != nil:
			fmt.Fprintf(out, "    State:\tRunning\n")
			fmt.Fprintf(out, "      Started:\t%v\n", state.Running.StartedAt.Time.Format(time.RFC1123Z))
		case state.Waiting != nil:
			fmt.Fprintf(out, "    State:\tWaiting\n")
			if state.Waiting.Reason != "" {
				fmt.Fprintf(out, "      Reason:\t%s\n", state.Waiting.Reason)
			}
		case state.Terminated != nil:
			fmt.Fprintf(out, "    State:\tTerminated\n")
			if state.Terminated.Reason != "" {
				fmt.Fprintf(out, "      Reason:\t%s\n", state.Terminated.Reason)
			}
			if state.Terminated.Message != "" {
				fmt.Fprintf(out, "      Message:\t%s\n", state.Terminated.Message)
			}
			fmt.Fprintf(out, "      Exit Code:\t%d\n", state.Terminated.ExitCode)
			if state.Terminated.Signal > 0 {
				fmt.Fprintf(out, "      Signal:\t%d\n", state.Terminated.Signal)
			}
			fmt.Fprintf(out, "      Started:\t%s\n", state.Terminated.StartedAt.Time.Format(time.RFC1123Z))
			fmt.Fprintf(out, "      Finished:\t%s\n", state.Terminated.FinishedAt.Time.Format(time.RFC1123Z))
		default:
			fmt.Fprintf(out, "    State:\tWaiting\n")
		}

		fmt.Fprintf(out, "    Ready:\t%v\n", printBool(status.Ready))
		fmt.Fprintf(out, "    Restart Count:\t%d\n", status.RestartCount)
		fmt.Fprintf(out, "    Variables:\n")
		for _, e := range container.Env {
			if e.ValueFrom != nil && e.ValueFrom.FieldRef != nil {
				valueFrom := envValueFrom(pod, e)
				fmt.Fprintf(out, "      %s:\t%s (%s:%s)\n", e.Name, valueFrom, e.ValueFrom.FieldRef.APIVersion, e.ValueFrom.FieldRef.FieldPath)
			} else {
				fmt.Fprintf(out, "      %s:\t%s\n", e.Name, e.Value)
			}
		}
	}
}

func envValueFrom(pod *api.Pod, e api.EnvVar) string {
	internalFieldPath, _, err := api.Scheme.ConvertFieldLabel(e.ValueFrom.FieldRef.APIVersion, "Pod", e.ValueFrom.FieldRef.FieldPath, "")
	if err != nil {
		return "" // pod validation should catch this on create
	}

	valueFrom, err := fieldpath.ExtractFieldPathAsString(pod, internalFieldPath)
	if err != nil {
		return "" // pod validation should catch this on create
	}

	return valueFrom
}

func printBool(value bool) string {
	if value {
		return "True"
	}

	return "False"
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

	return describeReplicationController(controller, events, running, waiting, succeeded, failed)
}

func describeReplicationController(controller *api.ReplicationController, events *api.EventList, running, waiting, succeeded, failed int) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", controller.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", controller.Namespace)
		if controller.Spec.Template != nil {
			fmt.Fprintf(out, "Image(s):\t%s\n", makeImageList(&controller.Spec.Template.Spec))
		} else {
			fmt.Fprintf(out, "Image(s):\t%s\n", "<no template>")
		}
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(controller.Spec.Selector))
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(controller.Labels))
		fmt.Fprintf(out, "Replicas:\t%d current / %d desired\n", controller.Status.Replicas, controller.Spec.Replicas)
		fmt.Fprintf(out, "Pods Status:\t%d Running / %d Waiting / %d Succeeded / %d Failed\n", running, waiting, succeeded, failed)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// SecretDescriber generates information about a secret
type SecretDescriber struct {
	client.Interface
}

func (d *SecretDescriber) Describe(namespace, name string) (string, error) {
	c := d.Secrets(namespace)

	secret, err := c.Get(name)
	if err != nil {
		return "", err
	}

	return describeSecret(secret)
}

func describeSecret(secret *api.Secret) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", secret.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", secret.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(secret.Labels))
		fmt.Fprintf(out, "Annotations:\t%s\n", formatLabels(secret.Annotations))

		fmt.Fprintf(out, "\nType:\t%s\n", secret.Type)

		fmt.Fprintf(out, "\nData\n====\n")
		for k, v := range secret.Data {
			switch {
			case k == api.ServiceAccountTokenKey && secret.Type == api.SecretTypeServiceAccountToken:
				fmt.Fprintf(out, "%s:\t%s\n", k, string(v))
			default:
				fmt.Fprintf(out, "%s:\t%d bytes\n", k, len(v))
			}
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

	endpoints, _ := d.Endpoints(namespace).Get(name)
	events, _ := d.Events(namespace).Search(service)

	return describeService(service, endpoints, events)
}

func buildIngressString(ingress []api.LoadBalancerIngress) string {
	var buffer bytes.Buffer

	for i := range ingress {
		if i != 0 {
			buffer.WriteString(", ")
		}
		if ingress[i].IP != "" {
			buffer.WriteString(ingress[i].IP)
		} else {
			buffer.WriteString(ingress[i].Hostname)
		}
	}
	return buffer.String()
}

func describeService(service *api.Service, endpoints *api.Endpoints, events *api.EventList) (string, error) {
	if endpoints == nil {
		endpoints = &api.Endpoints{}
	}
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", service.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", service.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(service.Labels))
		fmt.Fprintf(out, "Selector:\t%s\n", formatLabels(service.Spec.Selector))
		fmt.Fprintf(out, "Type:\t%s\n", service.Spec.Type)
		fmt.Fprintf(out, "IP:\t%s\n", service.Spec.ClusterIP)
		if len(service.Status.LoadBalancer.Ingress) > 0 {
			list := buildIngressString(service.Status.LoadBalancer.Ingress)
			fmt.Fprintf(out, "LoadBalancer Ingress:\t%s\n", list)
		}
		for i := range service.Spec.Ports {
			sp := &service.Spec.Ports[i]

			name := sp.Name
			if name == "" {
				name = "<unnamed>"
			}
			fmt.Fprintf(out, "Port:\t%s\t%d/%s\n", name, sp.Port, sp.Protocol)
			if sp.NodePort != 0 {
				fmt.Fprintf(out, "NodePort:\t%s\t%d/%s\n", name, sp.NodePort, sp.Protocol)
			}
			fmt.Fprintf(out, "Endpoints:\t%s\n", formatEndpoints(endpoints, util.NewStringSet(sp.Name)))
		}
		fmt.Fprintf(out, "Session Affinity:\t%s\n", service.Spec.SessionAffinity)
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

// ServiceAccountDescriber generates information about a service.
type ServiceAccountDescriber struct {
	client.Interface
}

func (d *ServiceAccountDescriber) Describe(namespace, name string) (string, error) {
	c := d.ServiceAccounts(namespace)

	serviceAccount, err := c.Get(name)
	if err != nil {
		return "", err
	}

	tokens := []api.Secret{}

	tokenSelector := fields.SelectorFromSet(map[string]string{client.SecretType: string(api.SecretTypeServiceAccountToken)})
	secrets, err := d.Secrets(namespace).List(labels.Everything(), tokenSelector)
	if err == nil {
		for _, s := range secrets.Items {
			name, _ := s.Annotations[api.ServiceAccountNameKey]
			uid, _ := s.Annotations[api.ServiceAccountUIDKey]
			if name == serviceAccount.Name && uid == string(serviceAccount.UID) {
				tokens = append(tokens, s)
			}
		}
	}

	return describeServiceAccount(serviceAccount, tokens)
}

func describeServiceAccount(serviceAccount *api.ServiceAccount, tokens []api.Secret) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", serviceAccount.Name)
		fmt.Fprintf(out, "Namespace:\t%s\n", serviceAccount.Namespace)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(serviceAccount.Labels))
		fmt.Fprintln(out)

		var (
			emptyHeader = "                   "
			pullHeader  = "Image pull secrets:"
			mountHeader = "Mountable secrets: "
			tokenHeader = "Tokens:            "

			pullSecretNames  = []string{}
			mountSecretNames = []string{}
			tokenSecretNames = []string{}
		)

		for _, s := range serviceAccount.ImagePullSecrets {
			pullSecretNames = append(pullSecretNames, s.Name)
		}
		for _, s := range serviceAccount.Secrets {
			mountSecretNames = append(mountSecretNames, s.Name)
		}
		for _, s := range tokens {
			tokenSecretNames = append(tokenSecretNames, s.Name)
		}

		types := map[string][]string{
			pullHeader:  pullSecretNames,
			mountHeader: mountSecretNames,
			tokenHeader: tokenSecretNames,
		}
		for header, names := range types {
			if len(names) == 0 {
				fmt.Fprintf(out, "%s\t<none>\n", header)
			} else {
				prefix := header
				for _, name := range names {
					fmt.Fprintf(out, "%s\t%s\n", prefix, name)
					prefix = emptyHeader
				}
			}
			fmt.Fprintln(out)
		}

		return nil
	})
}

// NodeDescriber generates information about a node.
type NodeDescriber struct {
	client.Interface
}

func (d *NodeDescriber) Describe(namespace, name string) (string, error) {
	mc := d.Nodes()
	node, err := mc.Get(name)
	if err != nil {
		return "", err
	}

	var pods []*api.Pod
	allPods, err := d.Pods(namespace).List(labels.Everything(), fields.Everything())
	if err != nil {
		return "", err
	}
	for i := range allPods.Items {
		pod := &allPods.Items[i]
		if pod.Spec.NodeName != name {
			continue
		}
		pods = append(pods, pod)
	}

	var events *api.EventList
	if ref, err := api.GetReference(node); err != nil {
		glog.Errorf("Unable to construct reference to '%#v': %v", node, err)
	} else {
		// TODO: We haven't decided the namespace for Node object yet.
		ref.UID = types.UID(ref.Name)
		events, _ = d.Events("").Search(ref)
	}

	return describeNode(node, pods, events)
}

func describeNode(node *api.Node, pods []*api.Pod, events *api.EventList) (string, error) {
	return tabbedString(func(out io.Writer) error {
		fmt.Fprintf(out, "Name:\t%s\n", node.Name)
		fmt.Fprintf(out, "Labels:\t%s\n", formatLabels(node.Labels))
		fmt.Fprintf(out, "CreationTimestamp:\t%s\n", node.CreationTimestamp.Time.Format(time.RFC1123Z))
		if len(node.Status.Conditions) > 0 {
			fmt.Fprint(out, "Conditions:\n  Type\tStatus\tLastHeartbeatTime\tLastTransitionTime\tReason\tMessage\n")
			for _, c := range node.Status.Conditions {
				fmt.Fprintf(out, "  %v \t%v \t%s \t%s \t%v \t%v\n",
					c.Type,
					c.Status,
					c.LastHeartbeatTime.Time.Format(time.RFC1123Z),
					c.LastTransitionTime.Time.Format(time.RFC1123Z),
					c.Reason,
					c.Message)
			}
		}
		var addresses []string
		for _, address := range node.Status.Addresses {
			addresses = append(addresses, address.Address)
		}
		fmt.Fprintf(out, "Addresses:\t%s\n", strings.Join(addresses, ","))
		if len(node.Status.Capacity) > 0 {
			fmt.Fprintf(out, "Capacity:\n")
			for resource, value := range node.Status.Capacity {
				fmt.Fprintf(out, " %s:\t%s\n", resource, value.String())
			}
		}

		fmt.Fprintf(out, "Version:\n")
		fmt.Fprintf(out, " Kernel Version:\t%s\n", node.Status.NodeInfo.KernelVersion)
		fmt.Fprintf(out, " OS Image:\t%s\n", node.Status.NodeInfo.OsImage)
		fmt.Fprintf(out, " Container Runtime Version:\t%s\n", node.Status.NodeInfo.ContainerRuntimeVersion)
		fmt.Fprintf(out, " Kubelet Version:\t%s\n", node.Status.NodeInfo.KubeletVersion)
		fmt.Fprintf(out, " Kube-Proxy Version:\t%s\n", node.Status.NodeInfo.KubeProxyVersion)

		if len(node.Spec.PodCIDR) > 0 {
			fmt.Fprintf(out, "PodCIDR:\t%s\n", node.Spec.PodCIDR)
		}
		if len(node.Spec.ExternalID) > 0 {
			fmt.Fprintf(out, "ExternalID:\t%s\n", node.Spec.ExternalID)
		}
		fmt.Fprintf(out, "Pods:\t(%d in total)\n", len(pods))
		fmt.Fprint(out, "  Namespace\tName\n")
		for _, pod := range pods {
			fmt.Fprintf(out, "  %s\t%s\n", pod.Namespace, pod.Name)
		}
		if events != nil {
			DescribeEvents(events, out)
		}
		return nil
	})
}

func DescribeEvents(el *api.EventList, w io.Writer) {
	if len(el.Items) == 0 {
		fmt.Fprint(w, "No events.")
		return
	}
	sort.Sort(SortableEvents(el.Items))
	fmt.Fprint(w, "Events:\n  FirstSeen\tLastSeen\tCount\tFrom\tSubobjectPath\tReason\tMessage\n")
	for _, e := range el.Items {
		fmt.Fprintf(w, "  %s\t%s\t%d\t%v\t%v\t%v\t%v\n",
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
func getReplicationControllersForLabels(c client.ReplicationControllerInterface, labelsToMatch labels.Labels) ([]api.ReplicationController, error) {
	// Get all replication controllers.
	// TODO this needs a namespace scope as argument
	rcs, err := c.List(labels.Everything())
	if err != nil {
		return nil, fmt.Errorf("error getting replication controllers: %v", err)
	}

	// Find the ones that match labelsToMatch.
	var matchingRCs []api.ReplicationController
	for _, controller := range rcs.Items {
		selector := labels.SelectorFromSet(controller.Spec.Selector)
		if selector.Matches(labelsToMatch) {
			matchingRCs = append(matchingRCs, controller)
		}
	}
	return matchingRCs, nil
}

func printReplicationControllersByLabels(matchingRCs []api.ReplicationController) string {
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
	rcPods, err := c.List(labels.SelectorFromSet(controller.Spec.Selector), fields.Everything())
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

// newErrNoDescriber creates a new ErrNoDescriber with the names of the provided types.
func newErrNoDescriber(types ...reflect.Type) error {
	names := []string{}
	for _, t := range types {
		names = append(names, t.String())
	}
	return ErrNoDescriber{Types: names}
}

// Describers implements ObjectDescriber against functions registered via Add. Those functions can
// be strongly typed. Types are exactly matched (no conversion or assignable checks).
type Describers struct {
	searchFns map[reflect.Type][]typeFunc
}

// DescribeObject implements ObjectDescriber and will attempt to print the provided object to a string,
// if at least one describer function has been registered with the exact types passed, or if any
// describer can print the exact object in its first argument (the remainder will be provided empty
// values). If no function registered with Add can satisfy the passed objects, an ErrNoDescriber will
// be returned
// TODO: reorder and partial match extra.
func (d *Describers) DescribeObject(exact interface{}, extra ...interface{}) (string, error) {
	exactType := reflect.TypeOf(exact)
	fns, ok := d.searchFns[exactType]
	if !ok {
		return "", newErrNoDescriber(exactType)
	}
	if len(extra) == 0 {
		for _, typeFn := range fns {
			if len(typeFn.Extra) == 0 {
				return typeFn.Describe(exact, extra...)
			}
		}
		typeFn := fns[0]
		for _, t := range typeFn.Extra {
			v := reflect.New(t).Elem()
			extra = append(extra, v.Interface())
		}
		return fns[0].Describe(exact, extra...)
	}

	types := []reflect.Type{}
	for _, obj := range extra {
		types = append(types, reflect.TypeOf(obj))
	}
	for _, typeFn := range fns {
		if typeFn.Matches(types) {
			return typeFn.Describe(exact, extra...)
		}
	}
	return "", newErrNoDescriber(append([]reflect.Type{exactType}, types...)...)
}

// Add adds one or more describer functions to the Describer. The passed function must
// match the signature:
//
//     func(...) (string, error)
//
// Any number of arguments may be provided.
func (d *Describers) Add(fns ...interface{}) error {
	for _, fn := range fns {
		fv := reflect.ValueOf(fn)
		ft := fv.Type()
		if ft.Kind() != reflect.Func {
			return fmt.Errorf("expected func, got: %v", ft)
		}
		if ft.NumIn() == 0 {
			return fmt.Errorf("expected at least one 'in' params, got: %v", ft)
		}
		if ft.NumOut() != 2 {
			return fmt.Errorf("expected two 'out' params - (string, error), got: %v", ft)
		}
		types := []reflect.Type{}
		for i := 0; i < ft.NumIn(); i++ {
			types = append(types, ft.In(i))
		}
		if ft.Out(0) != reflect.TypeOf(string("")) {
			return fmt.Errorf("expected string return, got: %v", ft)
		}
		var forErrorType error
		// This convolution is necessary, otherwise TypeOf picks up on the fact
		// that forErrorType is nil.
		errorType := reflect.TypeOf(&forErrorType).Elem()
		if ft.Out(1) != errorType {
			return fmt.Errorf("expected error return, got: %v", ft)
		}

		exact := types[0]
		extra := types[1:]
		if d.searchFns == nil {
			d.searchFns = make(map[reflect.Type][]typeFunc)
		}
		fns := d.searchFns[exact]
		fn := typeFunc{Extra: extra, Fn: fv}
		fns = append(fns, fn)
		d.searchFns[exact] = fns
	}
	return nil
}

// typeFunc holds information about a describer function and the types it accepts
type typeFunc struct {
	Extra []reflect.Type
	Fn    reflect.Value
}

// Matches returns true when the passed types exactly match the Extra list.
// TODO: allow unordered types to be matched and reorderd.
func (fn typeFunc) Matches(types []reflect.Type) bool {
	if len(fn.Extra) != len(types) {
		return false
	}
	for i := range types {
		if fn.Extra[i] != types[i] {
			return false
		}
	}
	return true
}

// Describe invokes the nested function with the exact number of arguments.
func (fn typeFunc) Describe(exact interface{}, extra ...interface{}) (string, error) {
	values := []reflect.Value{reflect.ValueOf(exact)}
	for _, obj := range extra {
		values = append(values, reflect.ValueOf(obj))
	}
	out := fn.Fn.Call(values)
	s := out[0].Interface().(string)
	var err error
	if !out[1].IsNil() {
		err = out[1].Interface().(error)
	}
	return s, err
}
