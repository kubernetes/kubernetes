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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/conversion"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/runtime"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/types"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/volume"
	"github.com/docker/docker/pkg/units"
	"github.com/ghodss/yaml"
	"github.com/golang/glog"
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
// recieved from watches.
type HumanReadablePrinter struct {
	handlerMap    map[reflect.Type]*handlerEntry
	noHeaders     bool
	withNamespace bool
	lastType      reflect.Type
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(noHeaders, withNamespace bool) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap:    make(map[reflect.Type]*handlerEntry),
		noHeaders:     noHeaders,
		withNamespace: withNamespace,
	}
	printer.addDefaultHandlers()
	return printer
}

// Handler adds a print handler with a given set of columns to HumanReadablePrinter instance.
// printFunc is the function that will be called to print an object.
// It must be of the following type:
//  func printFunc(object ObjectType, w io.Writer, withNamespace bool) error
// where ObjectType is the type of the object that will be printed.
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

func (h *HumanReadablePrinter) validatePrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function.", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 3 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 3 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*io.Writer)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, w io.Writer, withNamespace bool) error", funcType.In(0))
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

var podColumns = []string{"POD", "IP", "CONTAINER(S)", "IMAGE(S)", "HOST", "LABELS", "STATUS", "CREATED", "MESSAGE"}
var podTemplateColumns = []string{"TEMPLATE", "CONTAINER(S)", "IMAGE(S)", "PODLABELS"}
var replicationControllerColumns = []string{"CONTROLLER", "CONTAINER(S)", "IMAGE(S)", "SELECTOR", "REPLICAS"}
var serviceColumns = []string{"NAME", "LABELS", "SELECTOR", "IP(S)", "PORT(S)"}
var endpointColumns = []string{"NAME", "ENDPOINTS"}
var nodeColumns = []string{"NAME", "LABELS", "STATUS"}
var eventColumns = []string{"FIRSTSEEN", "LASTSEEN", "COUNT", "NAME", "KIND", "SUBOBJECT", "REASON", "SOURCE", "MESSAGE"}
var limitRangeColumns = []string{"NAME"}
var resourceQuotaColumns = []string{"NAME"}
var namespaceColumns = []string{"NAME", "LABELS", "STATUS"}
var secretColumns = []string{"NAME", "TYPE", "DATA"}
var serviceAccountColumns = []string{"NAME", "SECRETS"}
var persistentVolumeColumns = []string{"NAME", "LABELS", "CAPACITY", "ACCESSMODES", "STATUS", "CLAIM"}
var persistentVolumeClaimColumns = []string{"NAME", "LABELS", "STATUS", "VOLUME"}
var componentStatusColumns = []string{"NAME", "STATUS", "MESSAGE", "ERROR"}

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
Loop:
	for i := range endpoints.Subsets {
		ss := &endpoints.Subsets[i]
		for i := range ss.Ports {
			port := &ss.Ports[i]
			if ports == nil || ports.Has(port.Name) {
				for i := range ss.Addresses {
					if len(list) == max {
						more = true
						break Loop
					}
					addr := &ss.Addresses[i]
					list = append(list, fmt.Sprintf("%s:%d", addr.IP, port.Port))
				}
			}
		}
	}
	ret := strings.Join(list, ",")
	if more {
		ret += "..."
	}
	return ret
}

func podHostString(host, ip string) string {
	if host == "" && ip == "" {
		return "<unassigned>"
	}
	return host + "/" + ip
}

// translateTimestamp returns the elapsed time since timestamp in
// human-readable approximation.
func translateTimestamp(timestamp util.Time) string {
	return units.HumanDuration(time.Now().Sub(timestamp.Time))
}

// interpretContainerStatus interprets the container status and returns strings
// associated with columns "STATUS", "CREATED", and "MESSAGE".
// The meaning of MESSAGE varies based on the context of STATUS:
//     STATUS: Waiting; MESSAGE: reason for waiting
//     STATUS: Running; MESSAGE: reason for the last termination
//     STATUS: Terminated; MESSAGE: reason for this termination
func interpretContainerStatus(status *api.ContainerStatus) (string, string, string, error) {
	// Helper function to compose a meaning message from terminate state.
	getTermMsg := func(state *api.ContainerStateTerminated) string {
		var message string
		if state != nil {
			message = fmt.Sprintf("exit code %d", state.ExitCode)
			if state.Reason != "" {
				message = fmt.Sprintf("%s, reason: %s", message, state.Reason)
			}
		}
		return message
	}

	state := &status.State
	if state.Waiting != nil {
		return "Waiting", "", state.Waiting.Reason, nil
	} else if state.Running != nil {
		// Get the information of the last termination state. This is useful if
		// a container is stuck in a crash loop.
		message := getTermMsg(status.LastTerminationState.Termination)
		if message != "" {
			message = "last termination: " + message
		}
		stateMsg := "Running"
		if !status.Ready {
			stateMsg = stateMsg + " *not ready*"
		}
		return stateMsg, translateTimestamp(state.Running.StartedAt), message, nil
	} else if state.Termination != nil {
		return "Terminated", translateTimestamp(state.Termination.StartedAt), getTermMsg(state.Termination), nil
	}
	return "", "", "", fmt.Errorf("unknown container state %#v", *state)
}

func printPod(pod *api.Pod, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{pod.Namespace, pod.Name}.String()
	} else {
		name = pod.Name
	}

	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
		name,
		pod.Status.PodIP,
		"", "",
		podHostString(pod.Spec.Host, pod.Status.HostIP),
		formatLabels(pod.Labels),
		pod.Status.Phase,
		translateTimestamp(pod.CreationTimestamp),
		pod.Status.Message,
	)
	if err != nil {
		return err
	}
	// Lay out all containers on separate lines.
	statuses := pod.Status.ContainerStatuses
	if len(statuses) == 0 {
		// Container status has not been reported yet. Print basic information
		// of the containers and exit the function.
		for _, container := range pod.Spec.Containers {
			_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
				"", "", container.Name, container.Image, "", "", "", "", "")
			if err != nil {
				return err
			}
		}
		return nil
	}

	// Print the actual container statuses.
	for _, status := range statuses {
		state, created, message, err := interpretContainerStatus(&status)
		if err != nil {
			return err
		}
		_, err = fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
			"", "",
			status.Name,
			status.Image,
			"", "",
			state,
			created,
			message,
		)
		if err != nil {
			return err
		}
	}
	return nil
}

func printPodList(podList *api.PodList, w io.Writer, withNamespace bool) error {
	for _, pod := range podList.Items {
		if err := printPod(&pod, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printPodTemplate(pod *api.PodTemplate, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{pod.Namespace, pod.Name}.String()
	} else {
		name = pod.Name
	}

	containers := pod.Template.Spec.Containers
	var firstContainer api.Container
	if len(containers) > 0 {
		firstContainer, containers = containers[0], containers[1:]
	}
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n",
		name,
		firstContainer.Name,
		firstContainer.Image,
		formatLabels(pod.Template.Labels),
	)
	if err != nil {
		return err
	}
	// Lay out all the other containers on separate lines.
	for _, container := range containers {
		_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", "", container.Name, container.Image, "")
		if err != nil {
			return err
		}
	}
	return nil
}

func printPodTemplateList(podList *api.PodTemplateList, w io.Writer, withNamespace bool) error {
	for _, pod := range podList.Items {
		if err := printPodTemplate(&pod, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printReplicationController(controller *api.ReplicationController, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{controller.Namespace, controller.Name}.String()
	} else {
		name = controller.Name
	}

	containers := controller.Spec.Template.Spec.Containers
	var firstContainer api.Container
	if len(containers) > 0 {
		firstContainer, containers = containers[0], containers[1:]
	}
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d\n",
		name,
		firstContainer.Name,
		firstContainer.Image,
		formatLabels(controller.Spec.Selector),
		controller.Spec.Replicas)
	if err != nil {
		return err
	}
	// Lay out all the other containers on separate lines.
	for _, container := range containers {
		_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", "", container.Name, container.Image, "", "")
		if err != nil {
			return err
		}
	}
	return nil
}

func printReplicationControllerList(list *api.ReplicationControllerList, w io.Writer, withNamespace bool) error {
	for _, controller := range list.Items {
		if err := printReplicationController(&controller, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printService(svc *api.Service, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{svc.Namespace, svc.Name}.String()
	} else {
		name = svc.Name
	}

	ips := []string{svc.Spec.PortalIP}

	ingress := svc.Status.LoadBalancer.Ingress
	for i := range ingress {
		if ingress[i].IP != "" {
			ips = append(ips, ingress[i].IP)
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%d/%s\n", name, formatLabels(svc.Labels),
		formatLabels(svc.Spec.Selector), ips[0], svc.Spec.Ports[0].Port, svc.Spec.Ports[0].Protocol); err != nil {
		return err
	}

	count := len(svc.Spec.Ports)
	if len(ips) > count {
		count = len(ips)
	}
	for i := 1; i < count; i++ {
		ip := ""
		if len(ips) > i {
			ip = ips[i]
		}
		port := ""
		if len(svc.Spec.Ports) > i {
			port = fmt.Sprintf("%d/%s", svc.Spec.Ports[i].Port, svc.Spec.Ports[i].Protocol)
		}
		// Lay out additional ports.
		if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\n", "", "", "", ip, port); err != nil {
			return err
		}
	}

	return nil
}

func printServiceList(list *api.ServiceList, w io.Writer, withNamespace bool) error {
	for _, svc := range list.Items {
		if err := printService(&svc, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printEndpoints(endpoints *api.Endpoints, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{endpoints.Namespace, endpoints.Name}.String()
	} else {
		name = endpoints.Name
	}
	_, err := fmt.Fprintf(w, "%s\t%s\n", name, formatEndpoints(endpoints, nil))
	return err
}

func printEndpointsList(list *api.EndpointsList, w io.Writer, withNamespace bool) error {
	for _, item := range list.Items {
		if err := printEndpoints(&item, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printNamespace(item *api.Namespace, w io.Writer, withNamespace bool) error {
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\n", item.Name, formatLabels(item.Labels), item.Status.Phase)
	return err
}

func printNamespaceList(list *api.NamespaceList, w io.Writer, withNamespace bool) error {
	for _, item := range list.Items {
		if err := printNamespace(&item, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printSecret(item *api.Secret, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{item.Namespace, item.Name}.String()
	} else {
		name = item.Name
	}

	_, err := fmt.Fprintf(w, "%s\t%s\t%v\n", name, item.Type, len(item.Data))
	return err
}

func printSecretList(list *api.SecretList, w io.Writer, withNamespace bool) error {
	for _, item := range list.Items {
		if err := printSecret(&item, w, withNamespace); err != nil {
			return err
		}
	}

	return nil
}

func printServiceAccount(item *api.ServiceAccount, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{item.Namespace, item.Name}.String()
	} else {
		name = item.Name
	}

	_, err := fmt.Fprintf(w, "%s\t%d\n", name, len(item.Secrets))
	return err
}

func printServiceAccountList(list *api.ServiceAccountList, w io.Writer, withNamespace bool) error {
	for _, item := range list.Items {
		if err := printServiceAccount(&item, w, withNamespace); err != nil {
			return err
		}
	}

	return nil
}

func printNode(node *api.Node, w io.Writer, withNamespace bool) error {
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
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\n", node.Name, formatLabels(node.Labels), strings.Join(status, ","))
	return err
}

func printNodeList(list *api.NodeList, w io.Writer, withNamespace bool) error {
	for _, node := range list.Items {
		if err := printNode(&node, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolume(pv *api.PersistentVolume, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{pv.Namespace, pv.Name}.String()
	} else {
		name = pv.Name
	}

	claimRefUID := ""
	if pv.Spec.ClaimRef != nil {
		claimRefUID += pv.Spec.ClaimRef.Namespace
		claimRefUID += "/"
		claimRefUID += pv.Spec.ClaimRef.Name
	}

	modesStr := volume.GetAccessModesAsString(pv.Spec.AccessModes)

	aQty := pv.Spec.Capacity[api.ResourceStorage]
	aSize := aQty.Value()

	_, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s\t%s\t%s\n", name, formatLabels(pv.Labels), aSize, modesStr, pv.Status.Phase, claimRefUID)
	return err
}

func printPersistentVolumeList(list *api.PersistentVolumeList, w io.Writer, withNamespace bool) error {
	for _, pv := range list.Items {
		if err := printPersistentVolume(&pv, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolumeClaim(pvc *api.PersistentVolumeClaim, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{pvc.Namespace, pvc.Name}.String()
	} else {
		name = pvc.Name
	}

	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", name, pvc.Labels, pvc.Status.Phase, pvc.Spec.VolumeName)
	return err
}

func printPersistentVolumeClaimList(list *api.PersistentVolumeClaimList, w io.Writer, withNamespace bool) error {
	for _, psd := range list.Items {
		if err := printPersistentVolumeClaim(&psd, w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printEvent(event *api.Event, w io.Writer, withNamespace bool) error {
	_, err := fmt.Fprintf(
		w, "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\n",
		event.FirstTimestamp.Time.Format(time.RFC1123Z),
		event.LastTimestamp.Time.Format(time.RFC1123Z),
		event.Count,
		event.InvolvedObject.Name,
		event.InvolvedObject.Kind,
		event.InvolvedObject.FieldPath,
		event.Reason,
		event.Source,
		event.Message,
	)
	return err
}

// Sorts and prints the EventList in a human-friendly format.
func printEventList(list *api.EventList, w io.Writer, withNamespace bool) error {
	sort.Sort(SortableEvents(list.Items))
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printLimitRange(limitRange *api.LimitRange, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{limitRange.Namespace, limitRange.Name}.String()
	} else {
		name = limitRange.Name
	}

	_, err := fmt.Fprintf(
		w, "%s\n",
		name,
	)
	return err
}

// Prints the LimitRangeList in a human-friendly format.
func printLimitRangeList(list *api.LimitRangeList, w io.Writer, withNamespace bool) error {
	for i := range list.Items {
		if err := printLimitRange(&list.Items[i], w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printResourceQuota(resourceQuota *api.ResourceQuota, w io.Writer, withNamespace bool) error {
	var name string
	if withNamespace {
		name = types.NamespacedName{resourceQuota.Namespace, resourceQuota.Name}.String()
	} else {
		name = resourceQuota.Name
	}

	_, err := fmt.Fprintf(
		w, "%s\n",
		name,
	)
	return err
}

// Prints the ResourceQuotaList in a human-friendly format.
func printResourceQuotaList(list *api.ResourceQuotaList, w io.Writer, withNamespace bool) error {
	for i := range list.Items {
		if err := printResourceQuota(&list.Items[i], w, withNamespace); err != nil {
			return err
		}
	}
	return nil
}

func printComponentStatus(item *api.ComponentStatus, w io.Writer, withNamespace bool) error {
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
	_, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\n", item.Name, status, message, error)
	return err
}

func printComponentStatusList(list *api.ComponentStatusList, w io.Writer, withNamespace bool) error {
	for _, item := range list.Items {
		if err := printComponentStatus(&item, w, withNamespace); err != nil {
			return err
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
			h.printHeader(handler.columns, w)
			h.lastType = t
		}
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w), reflect.ValueOf(h.withNamespace)}
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
