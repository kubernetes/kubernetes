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
	"encoding/json"
	"fmt"
	"io"
	"io/ioutil"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"text/template"
	"time"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/conversion"
	"k8s.io/kubernetes/pkg/runtime"
	"k8s.io/kubernetes/pkg/util"
	"k8s.io/kubernetes/pkg/volume"
)

// GetPrinter takes a format type, an optional format argument. It will return true
// if the format is generic (untyped), otherwise it will return false. The printer
// is agnostic to schema versions, so you must send arguments to PrintObj in the
// version you wish them to be shown using a VersionedPrinter (typically when
// generic is true).
func GetPrinter(format, formatArgument string) (ResourcePrinter, bool, error) {
	var printer ResourcePrinter
	switch format {
	case "json":
		printer = &JSONPrinter{}
	case "yaml":
		printer = &YAMLPrinter{}
	case "template":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("template format specified but no template given")
		}
		var err error
		printer, err = NewTemplatePrinter([]byte(formatArgument))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", formatArgument, err)
		}
	case "templatefile":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("templatefile format specified but no template file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		printer, err = NewTemplatePrinter(data)
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
	case "wide":
		fallthrough
	case "":
		return nil, false, nil
	default:
		return nil, false, fmt.Errorf("output format %q not recognized", format)
	}
	return printer, true, nil
}

// ResourcePrinter is an interface that knows how to print runtime objects.
type ResourcePrinter interface {
	// Print receives a runtime object, formats it and prints it to a writer.
	PrintObj(runtime.Object, io.Writer) error
}

// ResourcePrinterFunc is a function that can print objects
type ResourcePrinterFunc func(runtime.Object, io.Writer) error

// PrintObj implements ResourcePrinter
func (fn ResourcePrinterFunc) PrintObj(obj runtime.Object, w io.Writer) error {
	return fn(obj, w)
}

// VersionedPrinter takes runtime objects and ensures they are converted to a given API version
// prior to being passed to a nested printer.
type VersionedPrinter struct {
	printer   ResourcePrinter
	convertor runtime.ObjectConvertor
	version   []string
}

// NewVersionedPrinter wraps a printer to convert objects to a known API version prior to printing.
func NewVersionedPrinter(printer ResourcePrinter, convertor runtime.ObjectConvertor, version ...string) ResourcePrinter {
	return &VersionedPrinter{
		printer:   printer,
		convertor: convertor,
		version:   version,
	}
}

// PrintObj implements ResourcePrinter
func (p *VersionedPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if len(p.version) == 0 {
		return fmt.Errorf("no version specified, object cannot be converted")
	}
	for _, version := range p.version {
		if len(version) == 0 {
			continue
		}
		converted, err := p.convertor.ConvertToVersion(obj, version)
		if conversion.IsNotRegisteredError(err) {
			continue
		}
		if err != nil {
			return err
		}
		return p.printer.PrintObj(converted, w)
	}
	return fmt.Errorf("the object cannot be converted to any of the versions: %v", p.version)
}

// JSONPrinter is an implementation of ResourcePrinter which outputs an object as JSON.
type JSONPrinter struct {
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which simply writes the object to the Writer.
func (p *JSONPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	dst := bytes.Buffer{}
	err = json.Indent(&dst, data, "", "    ")
	dst.WriteByte('\n')
	_, err = w.Write(dst.Bytes())
	return err
}

// YAMLPrinter is an implementation of ResourcePrinter which outputs an object as YAML.
// The input object is assumed to be in the internal version of an API and is converted
// to the given version first.
type YAMLPrinter struct {
	version   string
	convertor runtime.ObjectConvertor
}

// PrintObj prints the data as YAML.
func (p *YAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	output, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(output))
	return err
}

type handlerEntry struct {
	columns   []string
	printFunc reflect.Value
}

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	handlerMap    map[reflect.Type]*handlerEntry
	noHeaders     bool
	withNamespace bool
	wide          bool
	columnLabels  []string
	lastType      reflect.Type
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(noHeaders, withNamespace bool, wide bool, columnLabels []string) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap:    make(map[reflect.Type]*handlerEntry),
		noHeaders:     noHeaders,
		withNamespace: withNamespace,
		wide:          wide,
		columnLabels:  columnLabels,
	}
	printer.addDefaultHandlers()
	return printer
}

// Handler adds a print handler with a given set of columns to HumanReadablePrinter instance.
// See validatePrintHandlerFunc for required method signature.
func (h *HumanReadablePrinter) Handler(columns []string, printFunc interface{}) error {
	printFuncValue := reflect.ValueOf(printFunc)
	if err := h.validatePrintHandlerFunc(printFuncValue); err != nil {
		glog.Errorf("Unable to add print handler: %v", err)
		return err
	}
	objType := printFuncValue.Type().In(0)
	h.handlerMap[objType] = &handlerEntry{
		columns:   columns,
		printFunc: printFuncValue,
	}
	return nil
}

// validatePrintHandlerFunc validates print handler signature.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//  func printFunc(object ObjectType, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error
// where ObjectType is the type of the object that will be printed.
func (h *HumanReadablePrinter) validatePrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 5 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 5 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*io.Writer)(nil)).Elem() ||
		funcType.In(4) != reflect.TypeOf((*[]string)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error", funcType.In(0))
	}
	return nil
}

func (h *HumanReadablePrinter) HandledResources() []string {
	keys := make([]string, 0)

	for k := range h.handlerMap {
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

// NOTE: When adding a new resource type here, please update the list
// pkg/kubectl/cmd/get.go to reflect the new resource type.
var podColumns = []string{"NAME", "READY", "STATUS", "RESTARTS", "AGE"}
var podTemplateColumns = []string{"TEMPLATE", "CONTAINER(S)", "IMAGE(S)", "PODLABELS"}
var replicationControllerColumns = []string{"CONTROLLER", "CONTAINER(S)", "IMAGE(S)", "SELECTOR", "REPLICAS", "AGE"}
var serviceColumns = []string{"NAME", "CLUSTER_IP", "EXTERNAL_IP", "PORT(S)", "SELECTOR", "AGE"}
var endpointColumns = []string{"NAME", "ENDPOINTS", "AGE"}
var nodeColumns = []string{"NAME", "LABELS", "STATUS", "AGE"}
var eventColumns = []string{"FIRSTSEEN", "LASTSEEN", "COUNT", "NAME", "KIND", "SUBOBJECT", "REASON", "SOURCE", "MESSAGE"}
var limitRangeColumns = []string{"NAME", "AGE"}
var resourceQuotaColumns = []string{"NAME", "AGE"}
var namespaceColumns = []string{"NAME", "LABELS", "STATUS", "AGE"}
var secretColumns = []string{"NAME", "TYPE", "DATA", "AGE"}
var serviceAccountColumns = []string{"NAME", "SECRETS", "AGE"}
var persistentVolumeColumns = []string{"NAME", "LABELS", "CAPACITY", "ACCESSMODES", "STATUS", "CLAIM", "REASON", "AGE"}
var persistentVolumeClaimColumns = []string{"NAME", "LABELS", "STATUS", "VOLUME", "AGE"}
var componentStatusColumns = []string{"NAME", "STATUS", "MESSAGE", "ERROR"}
var withNamespacePrefixColumns = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.

// addDefaultHandlers adds print handlers for default Kubernetes types.
func (h *HumanReadablePrinter) addDefaultHandlers() {
	h.Handler(podColumns, printPod)
	h.Handler(podColumns, printPodList)
	h.Handler(podTemplateColumns, printPodTemplate)
	h.Handler(podTemplateColumns, printPodTemplateList)
	h.Handler(replicationControllerColumns, printReplicationController)
	h.Handler(replicationControllerColumns, printReplicationControllerList)
	h.Handler(serviceColumns, printService)
	h.Handler(serviceColumns, printServiceList)
	h.Handler(endpointColumns, printEndpoints)
	h.Handler(endpointColumns, printEndpointsList)
	h.Handler(nodeColumns, printNode)
	h.Handler(nodeColumns, printNodeList)
	h.Handler(eventColumns, printEvent)
	h.Handler(eventColumns, printEventList)
	h.Handler(limitRangeColumns, printLimitRange)
	h.Handler(limitRangeColumns, printLimitRangeList)
	h.Handler(resourceQuotaColumns, printResourceQuota)
	h.Handler(resourceQuotaColumns, printResourceQuotaList)
	h.Handler(namespaceColumns, printNamespace)
	h.Handler(namespaceColumns, printNamespaceList)
	h.Handler(secretColumns, printSecret)
	h.Handler(secretColumns, printSecretList)
	h.Handler(serviceAccountColumns, printServiceAccount)
	h.Handler(serviceAccountColumns, printServiceAccountList)
	h.Handler(persistentVolumeClaimColumns, printPersistentVolumeClaim)
	h.Handler(persistentVolumeClaimColumns, printPersistentVolumeClaimList)
	h.Handler(persistentVolumeColumns, printPersistentVolume)
	h.Handler(persistentVolumeColumns, printPersistentVolumeList)
	h.Handler(componentStatusColumns, printComponentStatus)
	h.Handler(componentStatusColumns, printComponentStatusList)
}

func (h *HumanReadablePrinter) unknown(data []byte, w io.Writer) error {
	_, err := fmt.Fprintf(w, "Unknown object: %s", string(data))
	return err
}

func (h *HumanReadablePrinter) printHeader(columnNames []string, w io.Writer) error {
	if _, err := fmt.Fprintf(w, "%s\n", strings.Join(columnNames, "\t")); err != nil {
		return err
	}
	return nil
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

func podHostString(host, ip string) string {
	if host == "" && ip == "" {
		return "<unassigned>"
	}
	return host + "/" + ip
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

func printPod(pod *api.Pod, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := pod.Name
	namespace := pod.Namespace

	restarts := 0
	totalContainers := len(pod.Spec.Containers)
	readyContainers := 0

	reason := string(pod.Status.Phase)
	if pod.Status.Reason != "" {
		reason = pod.Status.Reason
	}

	for i := len(pod.Status.ContainerStatuses) - 1; i >= 0; i-- {
		container := pod.Status.ContainerStatuses[i]

		restarts += container.RestartCount
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
		} else if container.Ready && container.State.Running != nil {
			readyContainers++
		}
	}

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%d/%d\t%s\t%d\t%s",
		name,
		readyContainers,
		totalContainers,
		reason,
		restarts,
		translateTimestamp(pod.CreationTimestamp),
	); err != nil {
		return err
	}

	if wide {
		nodeName := pod.Spec.NodeName
		if _, err := fmt.Fprintf(w, "\t%s",
			nodeName,
		); err != nil {
			return err
		}
	}

	_, err := fmt.Fprint(w, appendLabels(pod.Labels, columnLabels))
	return err
}

func printPodList(podList *api.PodList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, pod := range podList.Items {
		if err := printPod(&pod, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printPodTemplate(pod *api.PodTemplate, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := pod.Name
	namespace := pod.Namespace

	containers := pod.Template.Spec.Containers
	var firstContainer api.Container
	if len(containers) > 0 {
		firstContainer, containers = containers[0], containers[1:]
	}

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s",
		name,
		firstContainer.Name,
		firstContainer.Image,
		formatLabels(pod.Template.Labels),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, appendLabels(pod.Labels, columnLabels)); err != nil {
		return err
	}

	// Lay out all the other containers on separate lines.
	extraLinePrefix := "\t"
	if withNamespace {
		extraLinePrefix = "\t\t"
	}
	for _, container := range containers {
		_, err := fmt.Fprintf(w, "%s%s\t%s\t%s", extraLinePrefix, container.Name, container.Image, "")
		if err != nil {
			return err
		}
		if _, err := fmt.Fprint(w, appendLabelTabs(columnLabels)); err != nil {
			return err
		}
	}
	return nil
}

func printPodTemplateList(podList *api.PodTemplateList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, pod := range podList.Items {
		if err := printPodTemplate(&pod, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printReplicationController(controller *api.ReplicationController, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := controller.Name
	namespace := controller.Namespace

	containers := controller.Spec.Template.Spec.Containers
	var firstContainer api.Container
	if len(containers) > 0 {
		firstContainer, containers = containers[0], containers[1:]
	}

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\t%s",
		name,
		firstContainer.Name,
		firstContainer.Image,
		formatLabels(controller.Spec.Selector),
		controller.Spec.Replicas,
		translateTimestamp(controller.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, appendLabels(controller.Labels, columnLabels)); err != nil {
		return err
	}

	// Lay out all the other containers on separate lines.
	extraLinePrefix := "\t"
	if withNamespace {
		extraLinePrefix = "\t\t"
	}
	for _, container := range containers {
		_, err := fmt.Fprintf(w, "%s%s\t%s\t%s\t%s", extraLinePrefix, container.Name, container.Image, "", "")
		if err != nil {
			return err
		}
		if _, err := fmt.Fprint(w, appendLabelTabs(columnLabels)); err != nil {
			return err
		}
	}
	return nil
}

func printReplicationControllerList(list *api.ReplicationControllerList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, controller := range list.Items {
		if err := printReplicationController(&controller, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func getServiceExternalIP(svc *api.Service) string {
	switch svc.Spec.Type {
	case api.ServiceTypeClusterIP:
		return "<none>"
	case api.ServiceTypeNodePort:
		return "nodes"
	case api.ServiceTypeLoadBalancer:
		ingress := svc.Status.LoadBalancer.Ingress
		result := []string{}
		for i := range ingress {
			if ingress[i].IP != "" {
				result = append(result, ingress[i].IP)
			}
		}
		return strings.Join(result, ",")
	}
	return "unknown"
}

func makePortString(ports []api.ServicePort) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		port := &ports[ix]
		pieces[ix] = fmt.Sprintf("%d/%s", port.Port, port.Protocol)
	}
	return strings.Join(pieces, ",")
}

func printService(svc *api.Service, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := svc.Name
	namespace := svc.Namespace

	internalIP := svc.Spec.ClusterIP
	externalIP := getServiceExternalIP(svc)

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s",
		name,
		internalIP,
		externalIP,
		makePortString(svc.Spec.Ports),
		formatLabels(svc.Spec.Selector),
		translateTimestamp(svc.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, appendLabels(svc.Labels, columnLabels)); err != nil {
		return err
	}
	return nil
}

func printServiceList(list *api.ServiceList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, svc := range list.Items {
		if err := printService(&svc, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printEndpoints(endpoints *api.Endpoints, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := endpoints.Name
	namespace := endpoints.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, formatEndpoints(endpoints, nil), translateTimestamp(endpoints.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(endpoints.Labels, columnLabels))
	return err
}

func printEndpointsList(list *api.EndpointsList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, item := range list.Items {
		if err := printEndpoints(&item, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printNamespace(item *api.Namespace, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	if withNamespace {
		return fmt.Errorf("namespace is not namespaced")
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", item.Name, formatLabels(item.Labels), item.Status.Phase, translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(item.Labels, columnLabels))
	return err
}

func printNamespaceList(list *api.NamespaceList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, item := range list.Items {
		if err := printNamespace(&item, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printSecret(item *api.Secret, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := item.Name
	namespace := item.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%v\t%s", name, item.Type, len(item.Data), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(item.Labels, columnLabels))
	return err
}

func printSecretList(list *api.SecretList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, item := range list.Items {
		if err := printSecret(&item, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}

	return nil
}

func printServiceAccount(item *api.ServiceAccount, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := item.Name
	namespace := item.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%d\t%s", name, len(item.Secrets), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(item.Labels, columnLabels))
	return err
}

func printServiceAccountList(list *api.ServiceAccountList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, item := range list.Items {
		if err := printServiceAccount(&item, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}

	return nil
}

func printNode(node *api.Node, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	if withNamespace {
		return fmt.Errorf("node is not namespaced")
	}
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

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", node.Name, formatLabels(node.Labels), strings.Join(status, ","), translateTimestamp(node.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(node.Labels, columnLabels))
	return err
}

func printNodeList(list *api.NodeList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, node := range list.Items {
		if err := printNode(&node, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolume(pv *api.PersistentVolume, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	if withNamespace {
		return fmt.Errorf("persistentVolume is not namespaced")
	}
	name := pv.Name

	claimRefUID := ""
	if pv.Spec.ClaimRef != nil {
		claimRefUID += pv.Spec.ClaimRef.Namespace
		claimRefUID += "/"
		claimRefUID += pv.Spec.ClaimRef.Name
	}

	modesStr := volume.GetAccessModesAsString(pv.Spec.AccessModes)

	aQty := pv.Spec.Capacity[api.ResourceStorage]
	aSize := aQty.Value()

	if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s",
		name,
		formatLabels(pv.Labels),
		aSize, modesStr,
		pv.Status.Phase,
		claimRefUID,
		pv.Status.Reason,
		translateTimestamp(pv.CreationTimestamp),
	); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(pv.Labels, columnLabels))
	return err
}

func printPersistentVolumeList(list *api.PersistentVolumeList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, pv := range list.Items {
		if err := printPersistentVolume(&pv, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolumeClaim(pvc *api.PersistentVolumeClaim, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := pvc.Name
	namespace := pvc.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s", name, pvc.Labels, pvc.Status.Phase, pvc.Spec.VolumeName, translateTimestamp(pvc.CreationTimestamp)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(pvc.Labels, columnLabels))
	return err
}

func printPersistentVolumeClaimList(list *api.PersistentVolumeClaimList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, psd := range list.Items {
		if err := printPersistentVolumeClaim(&psd, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printEvent(event *api.Event, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	namespace := event.Namespace
	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(
		w, "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s",
		translateTimestamp(event.FirstTimestamp),
		translateTimestamp(event.LastTimestamp),
		event.Count,
		event.InvolvedObject.Name,
		event.InvolvedObject.Kind,
		event.InvolvedObject.FieldPath,
		event.Reason,
		event.Source,
		event.Message,
	); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(event.Labels, columnLabels))
	return err
}

// Sorts and prints the EventList in a human-friendly format.
func printEventList(list *api.EventList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	sort.Sort(SortableEvents(list.Items))
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printLimitRange(limitRange *api.LimitRange, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := limitRange.Name
	namespace := limitRange.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(limitRange.CreationTimestamp),
	); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(limitRange.Labels, columnLabels))
	return err
}

// Prints the LimitRangeList in a human-friendly format.
func printLimitRangeList(list *api.LimitRangeList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for i := range list.Items {
		if err := printLimitRange(&list.Items[i], w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printResourceQuota(resourceQuota *api.ResourceQuota, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	name := resourceQuota.Name
	namespace := resourceQuota.Namespace

	if withNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(resourceQuota.CreationTimestamp),
	); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(resourceQuota.Labels, columnLabels))
	return err
}

// Prints the ResourceQuotaList in a human-friendly format.
func printResourceQuotaList(list *api.ResourceQuotaList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for i := range list.Items {
		if err := printResourceQuota(&list.Items[i], w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}
	return nil
}

func printComponentStatus(item *api.ComponentStatus, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	if withNamespace {
		return fmt.Errorf("componentStatus is not namespaced")
	}
	status := "Unknown"
	message := ""
	error := ""
	for _, condition := range item.Conditions {
		if condition.Type == api.ComponentHealthy {
			if condition.Status == api.ConditionTrue {
				status = "Healthy"
			} else {
				status = "Unhealthy"
			}
			message = condition.Message
			error = condition.Error
			break
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", item.Name, status, message, error); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, appendLabels(item.Labels, columnLabels))
	return err
}

func printComponentStatusList(list *api.ComponentStatusList, w io.Writer, withNamespace bool, wide bool, columnLabels []string) error {
	for _, item := range list.Items {
		if err := printComponentStatus(&item, w, withNamespace, wide, columnLabels); err != nil {
			return err
		}
	}

	return nil
}

func appendLabels(itemLabels map[string]string, columnLabels []string) string {
	var buffer bytes.Buffer

	for _, cl := range columnLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		if il, ok := itemLabels[cl]; ok {
			buffer.WriteString(fmt.Sprint(il))
		} else {
			buffer.WriteString("<none>")
		}
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// Append a set of tabs for each label column.  We need this in the case where
// we have extra lines so that the tabwriter will still line things up.
func appendLabelTabs(columnLabels []string) string {
	var buffer bytes.Buffer

	for i := range columnLabels {
		// NB: This odd dance is to make the loop both compatible with go 1.3 and
		// pass `gofmt -s`
		_ = i
		buffer.WriteString("\t")
	}
	buffer.WriteString("\n")

	return buffer.String()
}

func formatLabelHeaders(columnLabels []string) []string {
	formHead := make([]string, len(columnLabels))
	for i, l := range columnLabels {
		p := strings.Split(l, "/")
		formHead[i] = strings.ToUpper((p[len(p)-1]))
	}
	return formHead
}

// headers for -o wide
func formatWideHeaders(wide bool, t reflect.Type) []string {
	if wide {
		if t.String() == "*api.Pod" || t.String() == "*api.PodList" {
			return []string{"NODE"}
		}
	}
	return nil
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	w := tabwriter.NewWriter(output, 10, 4, 3, ' ', 0)
	defer w.Flush()
	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		if !h.noHeaders && t != h.lastType {
			headers := append(handler.columns, formatWideHeaders(h.wide, t)...)
			headers = append(headers, formatLabelHeaders(h.columnLabels)...)
			if h.withNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, w)
			h.lastType = t
		}
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w), reflect.ValueOf(h.withNamespace), reflect.ValueOf(h.wide), reflect.ValueOf(h.columnLabels)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		}
		return resultValue.Interface().(error)
	}
	return fmt.Errorf("error: unknown type %#v", obj)
}

// TemplatePrinter is an implementation of ResourcePrinter which formats data with a Go Template.
type TemplatePrinter struct {
	rawTemplate string
	template    *template.Template
}

func NewTemplatePrinter(tmpl []byte) (*TemplatePrinter, error) {
	t, err := template.New("output").
		Funcs(template.FuncMap{"exists": exists}).
		Parse(string(tmpl))
	if err != nil {
		return nil, err
	}
	return &TemplatePrinter{
		rawTemplate: string(tmpl),
		template:    t,
	}, nil
}

// PrintObj formats the obj with the Go Template.
func (p *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	data, err := json.Marshal(obj)
	if err != nil {
		return err
	}
	out := map[string]interface{}{}
	if err := json.Unmarshal(data, &out); err != nil {
		return err
	}
	if err = p.safeExecute(w, out); err != nil {
		// It is way easier to debug this stuff when it shows up in
		// stdout instead of just stdin. So in addition to returning
		// a nice error, also print useful stuff with the writer.
		fmt.Fprintf(w, "Error executing template: %v\n", err)
		fmt.Fprintf(w, "template was:\n\t%v\n", p.rawTemplate)
		fmt.Fprintf(w, "raw data was:\n\t%v\n", string(data))
		fmt.Fprintf(w, "object given to template engine was:\n\t%+v\n", out)
		return fmt.Errorf("error executing template '%v': '%v'\n----data----\n%+v\n", p.rawTemplate, err, out)
	}
	return nil
}

// safeExecute tries to execute the template, but catches panics and returns an error
// should the template engine panic.
func (p *TemplatePrinter) safeExecute(w io.Writer, obj interface{}) error {
	var panicErr error
	// Sorry for the double anonymous function. There's probably a clever way
	// to do this that has the defer'd func setting the value to be returned, but
	// that would be even less obvious.
	retErr := func() error {
		defer func() {
			if x := recover(); x != nil {
				panicErr = fmt.Errorf("caught panic: %+v", x)
			}
		}()
		return p.template.Execute(w, obj)
	}()
	if panicErr != nil {
		return panicErr
	}
	return retErr
}

func tabbedString(f func(io.Writer) error) (string, error) {
	out := new(tabwriter.Writer)
	buf := &bytes.Buffer{}
	out.Init(buf, 0, 8, 1, '\t', 0)

	err := f(out)
	if err != nil {
		return "", err
	}

	out.Flush()
	str := string(buf.String())
	return str, nil
}

// exists returns true if it would be possible to call the index function
// with these arguments.
//
// TODO: how to document this for users?
//
// index returns the result of indexing its first argument by the following
// arguments.  Thus "index x 1 2 3" is, in Go syntax, x[1][2][3]. Each
// indexed item must be a map, slice, or array.
func exists(item interface{}, indices ...interface{}) bool {
	v := reflect.ValueOf(item)
	for _, i := range indices {
		index := reflect.ValueOf(i)
		var isNil bool
		if v, isNil = indirect(v); isNil {
			return false
		}
		switch v.Kind() {
		case reflect.Array, reflect.Slice, reflect.String:
			var x int64
			switch index.Kind() {
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
				x = index.Int()
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
				x = int64(index.Uint())
			default:
				return false
			}
			if x < 0 || x >= int64(v.Len()) {
				return false
			}
			v = v.Index(int(x))
		case reflect.Map:
			if !index.IsValid() {
				index = reflect.Zero(v.Type().Key())
			}
			if !index.Type().AssignableTo(v.Type().Key()) {
				return false
			}
			if x := v.MapIndex(index); x.IsValid() {
				v = x
			} else {
				v = reflect.Zero(v.Type().Elem())
			}
		default:
			return false
		}
	}
	if _, isNil := indirect(v); isNil {
		return false
	}
	return true
}

// stolen from text/template
// indirect returns the item at the end of indirection, and a bool to indicate if it's nil.
// We indirect through pointers and empty interfaces (only) because
// non-empty interfaces have methods we might need.
func indirect(v reflect.Value) (rv reflect.Value, isNil bool) {
	for ; v.Kind() == reflect.Ptr || v.Kind() == reflect.Interface; v = v.Elem() {
		if v.IsNil() {
			return v, true
		}
		if v.Kind() == reflect.Interface && v.NumMethod() > 0 {
			break
		}
	}
	return v, false
}
