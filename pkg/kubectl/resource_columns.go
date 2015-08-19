/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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
	"reflect"
	"strings"
	"time"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume"
	"k8s.io/kubernetes/pkg/watch"
)

type ColumnPrinter interface {
	Title() string
	Content(runtime.Object) ([]string, error)
}

// Contains information about what columns should be included.
type printContext struct {
	withNamespace bool
	withEvent     bool
	wide          bool
	columnLabels  []string
}

// Simple implementation of ColumnPrinter that has also an indicator whether
// the printer should be included based on the given configuration (printContext)
type funcColumnPrinter struct {
	title     string
	content   interface{}
	condition func(printContext) bool
}

func (f funcColumnPrinter) Title() string {
	return f.title
}

func (f funcColumnPrinter) shouldInclude(c printContext) bool {
	return f.condition == nil || f.condition(c)
}

func (f funcColumnPrinter) Content(obj runtime.Object) ([]string, error) {
	objType := reflect.TypeOf(obj)
	contentValue := reflect.ValueOf(f.content)

	if contentValue.Kind() != reflect.Func {
		return nil, fmt.Errorf("invalid print handler. %#v is not a function", f.content)
	}
	funcType := contentValue.Type()

	args := []reflect.Value{}
	if funcType.NumIn() > 1 {
		return nil, fmt.Errorf("invalid print handler. %#v takes too many args", f.content)
	}
	// Zero or 1 arg
	if funcType.NumIn() == 1 {
		if funcType.In(0) == objType {
			args = append(args, reflect.ValueOf(obj))
		} else if funcType.In(0) == reflect.TypeOf((*runtime.Object)(nil)).Elem() {
			args = append(args, reflect.ValueOf(obj))
		} else {
			return nil, fmt.Errorf("Cannot populate type %v for %s", funcType.In(0), f.title)
		}
	}

	resultValue := contentValue.Call(args)
	var result []string

	if funcType.Out(0) == reflect.TypeOf((*string)(nil)).Elem() {
		result = []string{resultValue[0].String()}
	} else if funcType.Out(0) == reflect.TypeOf((*[]string)(nil)).Elem() {
		result = []string{}
		for i := 0; i < resultValue[0].Len(); i++ {
			result = append(result, resultValue[0].Index(i).String())
		}
	} else if funcType.Out(0) == reflect.TypeOf((*error)(nil)).Elem() {
		if !resultValue[0].IsNil() {
			return nil, resultValue[0].Interface().(error)
		}
	} else {
		return nil, fmt.Errorf("Wrong return type %v for %s", funcType.In(0), f.title)
	}

	if funcType.NumOut() == 2 {
		if funcType.Out(1) == reflect.TypeOf((*error)(nil)).Elem() {
			if !resultValue[1].IsNil() {
				return nil, resultValue[1].Interface().(error)
			}
		} else {
			return nil, fmt.Errorf("Wrong return type %v for %s", funcType.In(1), f.title)
		}
	}
	return result, nil
}

// Helper functions for getting fields viar reflection
func getStringField(obj runtime.Object, name string) string {
	return reflect.ValueOf(obj).Elem().FieldByName(name).String()
}

func getInterfaceField(obj runtime.Object, name string) interface{} {
	return reflect.ValueOf(obj).Elem().FieldByName(name).Interface()
}

func makeUpdateColumn(event *watch.EventType) funcColumnPrinter {
	return funcColumnPrinter{
		title: "UPDATE",
		content: func(c printContext) string {
			if event == nil {
				return "-"
			}
			return fmt.Sprintf("%s", *event)
		}}
}

func makeNameColumn(name string) funcColumnPrinter {
	return funcColumnPrinter{
		title:   name,
		content: func(obj runtime.Object) string { return getStringField(obj, "Name") }}
}

var namespaceColumn = funcColumnPrinter{
	title:     "NAMESPACE",
	content:   func(obj runtime.Object) string { return getStringField(obj, "Namespace") },
	condition: func(c printContext) bool { return c.withNamespace }}

var noNamespaceColumn = funcColumnPrinter{
	title:     "NAMESPACE",
	content:   func(obj runtime.Object) error { return fmt.Errorf("Namespace not supported") },
	condition: func(c printContext) bool { return c.withNamespace }}

var ageColumn = funcColumnPrinter{
	title: "AGE",
	content: func(obj runtime.Object) string {
		return translateTimestamp(getInterfaceField(obj, "CreationTimestamp").(util.Time))
	}}

func makeLabelColumn(labelName string) funcColumnPrinter {
	splitLabelName := strings.Split(labelName, "/")
	columnLabel := strings.ToUpper(splitLabelName[len(splitLabelName)-1])
	return funcColumnPrinter{
		title: columnLabel,
		content: func(obj runtime.Object) string {
			labelsMap := reflect.ValueOf(obj).Elem().FieldByName("Labels")
			val := labelsMap.MapIndex(reflect.ValueOf(labelName))
			if val.IsValid() {
				return val.String()
			} else {
				return "<n/a>"
			}
		}}
}

var podColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "READY", content: func(pod *api.Pod) string {
		readyContainers := 0
		for i := 0; i < len(pod.Status.ContainerStatuses); i++ {
			container := pod.Status.ContainerStatuses[i]
			if container.Ready && container.State.Running != nil {
				readyContainers++
			}
		}
		totalContainers := len(pod.Spec.Containers)
		return fmt.Sprintf("%d/%d", readyContainers, totalContainers)
	}},
	{title: "STATUS", content: func(pod *api.Pod) string {
		reason := string(pod.Status.Phase)
		if pod.Status.Reason != "" {
			reason = pod.Status.Reason
		}
		for i := len(pod.Status.ContainerStatuses) - 1; i >= 0; i-- {
			container := pod.Status.ContainerStatuses[i]

			if container.State.Waiting != nil && container.State.Waiting.Reason != "" {
				reason = container.State.Waiting.Reason
			} else if container.State.Terminated != nil && container.State.Terminated.Reason != "" {
				reason = container.State.Terminated.Reason
			} else if container.State.Terminated != nil && container.State.Terminated.Reason == "" {
				if container.State.Terminated.Signal != 0 {
					reason = fmt.Sprintf("Signal:%d", container.State.Terminated.Signal)
				} else {
					reason = fmt.Sprintf("ExitCode:%d", container.State.Terminated.ExitCode)
				}
			}
		}
		return reason
	}},
	{title: "RESTARTS", content: func(pod *api.Pod) string {
		restarts := 0
		for i := 0; i < len(pod.Status.ContainerStatuses); i++ {
			restarts += pod.Status.ContainerStatuses[i].RestartCount
		}
		return fmt.Sprintf("%d", restarts)
	}},
	ageColumn,
	{title: "NODE", content: func(pod *api.Pod) string { return pod.Spec.NodeName },
		condition: func(c printContext) bool { return c.wide }},
}

var podTemplateColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("TEMPLATE"),
	{title: "CONTAINER(S)", content: func(pod *api.PodTemplate) []string {
		result := []string{}
		for _, container := range pod.Template.Spec.Containers {
			result = append(result, container.Name)
		}
		return result
	}},
	{title: "IMAGE(S)", content: func(pod *api.PodTemplate) []string {
		result := []string{}
		for _, container := range pod.Template.Spec.Containers {
			result = append(result, container.Image)
		}
		return result
	}},
	{title: "PODLABELS", content: func(pod *api.PodTemplate) string { return formatLabels(pod.Template.Labels) }},
}

var replicationControllerColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("CONTROLLER"),
	{title: "CONTAINER(S)", content: func(rc *api.ReplicationController) []string {
		result := []string{}
		for _, container := range rc.Spec.Template.Spec.Containers {
			result = append(result, container.Name)
		}
		return result
	}},
	{title: "IMAGE(S)", content: func(rc *api.ReplicationController) []string {
		result := []string{}
		for _, container := range rc.Spec.Template.Spec.Containers {
			result = append(result, container.Image)
		}
		return result
	}},
	{title: "SELECTOR", content: func(rc *api.ReplicationController) string { return formatLabels(rc.Spec.Selector) }},
	{title: "REPLICAS", content: func(rc *api.ReplicationController) string { return fmt.Sprintf("%d", rc.Spec.Replicas) }},
	ageColumn,
}

var serviceColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "CLUSTER_IP", content: func(s *api.Service) string {
		return s.Spec.ClusterIP
	}},
	{title: "EXTERNAL_IP", content: func(s *api.Service) string {
		switch s.Spec.Type {
		case api.ServiceTypeClusterIP:
			return "<none>"
		case api.ServiceTypeNodePort:
			return "nodes"
		case api.ServiceTypeLoadBalancer:
			ingress := s.Status.LoadBalancer.Ingress
			result := []string{}
			for i := range ingress {
				if ingress[i].IP != "" {
					result = append(result, ingress[i].IP)
				}
			}
			return strings.Join(result, ",")
		}
		return "unknown"
	}},
	{title: "PORT(S)", content: func(s *api.Service) string {
		ports := []string{}
		for _, port := range s.Spec.Ports {
			ports = append(ports, fmt.Sprintf("%d/%s", port.Port, port.Protocol))
		}
		return strings.Join(ports, ",")
	}},
	{title: "SELECTOR", content: func(s *api.Service) string { return formatLabels(s.Spec.Selector) }},
	ageColumn,
}

var endpointsColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "ENDPOINTS", content: func(e *api.Endpoints) string { return formatEndpoints(e, nil) }},
	ageColumn,
}

var nodeColumns = []funcColumnPrinter{
	noNamespaceColumn,
	makeNameColumn("NAME"),
	{title: "LABELS", content: func(node *api.Node) string { return formatLabels(node.Labels) }},
	{title: "STATUS", content: func(node *api.Node) string {
		conditionMap := make(map[api.NodeConditionType]*api.NodeCondition)
		NodeAllConditions := []api.NodeConditionType{api.NodeReady}
		for i := range node.Status.Conditions {
			cond := node.Status.Conditions[i]
			conditionMap[cond.Type] = &cond
		}
		var status []string
		for _, validCondition := range NodeAllConditions {
			if condition, ok := conditionMap[validCondition]; ok {
				if condition.Status == api.ConditionTrue {
					status = append(status, string(condition.Type))
				} else {
					status = append(status, "Not"+string(condition.Type))
				}
			}
		}
		if len(status) == 0 {
			status = append(status, "Unknown")
		}
		if node.Spec.Unschedulable {
			status = append(status, "SchedulingDisabled")
		}
		return strings.Join(status, ",")
	}},
	ageColumn,
}

var eventColumns = []funcColumnPrinter{
	namespaceColumn,
	{title: "FIRSTSEEN", content: func(event *api.Event) string { return translateTimestamp(event.FirstTimestamp) }},
	{title: "LASTSEEN", content: func(event *api.Event) string { return translateTimestamp(event.LastTimestamp) }},
	{title: "COUNT", content: func(event *api.Event) string { return fmt.Sprintf("%d", event.Count) }},
	{title: "NAME", content: func(event *api.Event) string { return event.InvolvedObject.Name }},
	{title: "KIND", content: func(event *api.Event) string { return event.InvolvedObject.Kind }},
	{title: "SUBOBJECT", content: func(event *api.Event) string { return event.InvolvedObject.FieldPath }},
	{title: "REASON", content: func(event *api.Event) string { return event.Reason }},
	{title: "SOURCE", content: func(event *api.Event) string { return fmt.Sprintf("%s", event.Source) }},
	{title: "MESSAGE", content: func(event *api.Event) string { return event.Message }},
}

var limitRangeColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	ageColumn,
}

var resourceQuotaColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	ageColumn,
}

var namespaceColumns = []funcColumnPrinter{
	noNamespaceColumn,
	makeNameColumn("NAME"),
	{title: "LABELS", content: func(n *api.Namespace) string { return formatLabels(n.Labels) }},
	{title: "STATUS", content: func(n *api.Namespace) string { return fmt.Sprintf("%s", n.Status.Phase) }},
	ageColumn,
}

var secretColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "TYPE", content: func(s *api.Secret) string { return fmt.Sprintf("%s", s.Type) }},
	{title: "DATA", content: func(s *api.Secret) string { return fmt.Sprintf("%d", len(s.Data)) }},
	ageColumn,
}

var serviceAccountColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "SECRETS", content: func(s *api.ServiceAccount) string { return fmt.Sprintf("%d", len(s.Secrets)) }},
	ageColumn,
}

var persistentVolumeColumns = []funcColumnPrinter{
	noNamespaceColumn,
	makeNameColumn("NAME"),
	{title: "LABELS", content: func(p *api.PersistentVolume) string { return formatLabels(p.Labels) }},
	{title: "CAPACITY", content: func(p *api.PersistentVolume) string {
		aQty := p.Spec.Capacity[api.ResourceStorage]
		aSize := aQty.String()
		return aSize
	}},
	{title: "ACCESSMODES", content: func(p *api.PersistentVolume) string { return volume.GetAccessModesAsString(p.Spec.AccessModes) }},
	{title: "STATUS", content: func(p *api.PersistentVolume) string { return fmt.Sprintf("%s", p.Status.Phase) }},
	{title: "CLAIM", content: func(p *api.PersistentVolume) string {
		claimRefUID := ""
		if p.Spec.ClaimRef != nil {
			claimRefUID += p.Spec.ClaimRef.Namespace
			claimRefUID += "/"
			claimRefUID += p.Spec.ClaimRef.Name
		}
		return claimRefUID
	}},
	{title: "REASON", content: func(p *api.PersistentVolume) string { return p.Status.Reason }},
	ageColumn,
}

var persistentVolumeClaimColumns = []funcColumnPrinter{
	namespaceColumn,
	makeNameColumn("NAME"),
	{title: "STATUS", content: func(p *api.PersistentVolumeClaim) string { return fmt.Sprintf("%s", p.Status.Phase) }},
	{title: "LABELS", content: func(p *api.PersistentVolumeClaim) string { return formatLabels(p.Labels) }},
	{title: "VOLUME", content: func(p *api.PersistentVolumeClaim) string { return p.Spec.VolumeName }},
	{title: "CAPACITY", content: func(p *api.PersistentVolumeClaim) string {
		if p.Spec.VolumeName != "" {
			storage := p.Status.Capacity[api.ResourceStorage]
			return storage.String()
		}
		return ""
	}},
	{title: "ACCESSMODES", content: func(p *api.PersistentVolumeClaim) string {
		if p.Spec.VolumeName != "" {
			return volume.GetAccessModesAsString(p.Status.AccessModes)
		}
		return ""
	}},
	ageColumn,
}

var componentStatus = []funcColumnPrinter{
	noNamespaceColumn,
	makeNameColumn("NAME"),
	{title: "STATUS", content: func(cs *api.ComponentStatus) string {
		status := "Unknown"
		for _, condition := range cs.Conditions {
			if condition.Type == api.ComponentHealthy {
				if condition.Status == api.ConditionTrue {
					status = "Healthy"
				} else {
					status = "Unhealthy"
				}
				break
			}
		}
		return status
	}},
	{title: "MESSAGE", content: func(cs *api.ComponentStatus) string {
		message := ""
		for _, condition := range cs.Conditions {
			if condition.Type == api.ComponentHealthy {
				message = condition.Message
				break
			}
		}
		return message
	}},
	{title: "ERROR", content: func(cs *api.ComponentStatus) string {
		error := ""
		for _, condition := range cs.Conditions {
			if condition.Type == api.ComponentHealthy {
				error = condition.Error
				break
			}
		}
		return error
	}},
}

var columnMap = map[reflect.Type][]funcColumnPrinter{
	reflect.TypeOf((*api.Pod)(nil)):                   podColumns,
	reflect.TypeOf((*api.PodTemplate)(nil)):           podTemplateColumns,
	reflect.TypeOf((*api.ReplicationController)(nil)): replicationControllerColumns,
	reflect.TypeOf((*api.Service)(nil)):               serviceColumns,
	reflect.TypeOf((*api.Endpoints)(nil)):             endpointsColumns,
	reflect.TypeOf((*api.Node)(nil)):                  nodeColumns,
	reflect.TypeOf((*api.Event)(nil)):                 eventColumns,
	reflect.TypeOf((*api.LimitRange)(nil)):            limitRangeColumns,
	reflect.TypeOf((*api.ResourceQuota)(nil)):         resourceQuotaColumns,
	reflect.TypeOf((*api.Namespace)(nil)):             namespaceColumns,
	reflect.TypeOf((*api.Secret)(nil)):                secretColumns,
	reflect.TypeOf((*api.ServiceAccount)(nil)):        serviceAccountColumns,
	reflect.TypeOf((*api.PersistentVolume)(nil)):      persistentVolumeColumns,
	reflect.TypeOf((*api.PersistentVolumeClaim)(nil)): persistentVolumeClaimColumns,
	reflect.TypeOf((*api.ComponentStatus)(nil)):       componentStatus,
}

func getSupportedTypeNames() []string {
	keys := make([]string, 0)

	for k := range columnMap {
		// k.String looks like "*api.PodList" and we want just "pod"
		api := strings.Split(k.String(), ".")
		resource := api[len(api)-1]
		if strings.HasSuffix(resource, "List") {
			continue
		}
		resource = strings.ToLower(resource)
		keys = append(keys, resource)
	}
	return keys
}

func getColumnPrintersFor(t reflect.Type, c printContext) ([]ColumnPrinter, error) {
	columns, ok := columnMap[t]
	if !ok {
		return nil, fmt.Errorf("Type not supported")
	}
	result := []ColumnPrinter{}
	for _, column := range columns {
		if column.shouldInclude(c) {
			result = append(result, column)
		}
	}

	for _, label := range c.columnLabels {
		result = append(result, makeLabelColumn(label))
	}
	return result, nil
}

func buildHeaders(columns []ColumnPrinter) []string {
	headers := []string{}
	for _, entry := range columns {
		headers = append(headers, entry.Title())
	}
	return headers
}

func populateColumns(obj runtime.Object, columns []ColumnPrinter) ([][]string, error) {
	var result [][]string
	for _, column := range columns {
		columnContent, err := column.Content(obj)
		if err != nil {
			return nil, err
		}
		result = append(result, columnContent)
	}
	return result, nil
}

// Pass ports=nil for all ports.
func formatEndpoints(endpoints *api.Endpoints, ports util.StringSet) string {
	if len(endpoints.Subsets) == 0 {
		return "<none>"
	}
	list := []string{}
	max := 3
	more := false
	count := 0
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if ports == nil || ports.Has(port.Name) {
				for i := range ss.Addresses {
					if len(list) == max {
						more = true
					}
					addr := &ss.Addresses[i]
					if !more {
						list = append(list, fmt.Sprintf("%s:%d", addr.IP, port.Port))
					}
					count++
				}
			}
		}
	}
	ret := strings.Join(list, ",")
	if more {
		return fmt.Sprintf("%s + %d more...", ret, count-max)
	}
	return ret
}

func shortHumanDuration(d time.Duration) string {
	if seconds := int(d.Seconds()); seconds < 60 {
		return fmt.Sprintf("%ds", seconds)
	} else if minutes := int(d.Minutes()); minutes < 60 {
		return fmt.Sprintf("%dm", minutes)
	} else if hours := int(d.Hours()); hours < 24 {
		return fmt.Sprintf("%dh", hours)
	} else if hours < 24*364 {
		return fmt.Sprintf("%dd", hours/24)
	}
	return fmt.Sprintf("%dy", int(d.Hours()/24/365))
}

// translateTimestamp returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestamp(timestamp util.Time) string {
	return shortHumanDuration(time.Now().Sub(timestamp.Time))
}
