/*
Copyright 2014 The Kubernetes Authors.

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
	"os"
	"reflect"
	"sort"
	"strings"
	"text/tabwriter"
	"text/template"
	"time"

	"k8s.io/kubernetes/federation/apis/federation"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/events"
	"k8s.io/kubernetes/pkg/api/meta"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/apps"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
	"k8s.io/kubernetes/pkg/apis/storage"
	storageutil "k8s.io/kubernetes/pkg/apis/storage/util"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/runtime"
	utilerrors "k8s.io/kubernetes/pkg/util/errors"
	"k8s.io/kubernetes/pkg/util/jsonpath"
	"k8s.io/kubernetes/pkg/util/node"
	"k8s.io/kubernetes/pkg/util/sets"

	"github.com/ghodss/yaml"
	"github.com/golang/glog"
)

const (
	tabwriterMinWidth = 10
	tabwriterWidth    = 4
	tabwriterPadding  = 3
	tabwriterPadChar  = ' '
	tabwriterFlags    = 0
	loadBalancerWidth = 16
)

// GetPrinter takes a format type, an optional format argument. It will return true
// if the format is generic (untyped), otherwise it will return false. The printer
// is agnostic to schema versions, so you must send arguments to PrintObj in the
// version you wish them to be shown using a VersionedPrinter (typically when
// generic is true).
func GetPrinter(format, formatArgument string, noHeaders bool) (ResourcePrinter, bool, error) {
	var printer ResourcePrinter
	switch format {
	case "json":
		printer = &JSONPrinter{}
	case "yaml":
		printer = &YAMLPrinter{}
	case "name":
		printer = &NamePrinter{
			// TODO: this is wrong, these should be provided as an argument to GetPrinter
			Typer:   api.Scheme,
			Decoder: api.Codecs.UniversalDecoder(),
		}
	case "template", "go-template":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("template format specified but no template given")
		}
		var err error
		printer, err = NewTemplatePrinter([]byte(formatArgument))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", formatArgument, err)
		}
	case "templatefile", "go-template-file":
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
	case "jsonpath":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("jsonpath template format specified but no template given")
		}
		var err error
		printer, err = NewJSONPathPrinter(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error parsing jsonpath %s, %v\n", formatArgument, err)
		}
	case "jsonpath-file":
		if len(formatArgument) == 0 {
			return nil, false, fmt.Errorf("jsonpath file format specified but no template file file given")
		}
		data, err := ioutil.ReadFile(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		printer, err = NewJSONPathPrinter(string(data))
		if err != nil {
			return nil, false, fmt.Errorf("error parsing template %s, %v\n", string(data), err)
		}
	case "custom-columns":
		var err error
		if printer, err = NewCustomColumnsPrinterFromSpec(formatArgument, api.Codecs.UniversalDecoder(), noHeaders); err != nil {
			return nil, false, err
		}
	case "custom-columns-file":
		file, err := os.Open(formatArgument)
		if err != nil {
			return nil, false, fmt.Errorf("error reading template %s, %v\n", formatArgument, err)
		}
		defer file.Close()
		if printer, err = NewCustomColumnsPrinterFromTemplate(file, api.Codecs.UniversalDecoder()); err != nil {
			return nil, false, err
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
	HandledResources() []string
	//Can be used to print out warning/clarifications if needed
	//after all objects were printed
	AfterPrint(io.Writer, string) error
}

// ResourcePrinterFunc is a function that can print objects
type ResourcePrinterFunc func(runtime.Object, io.Writer) error

// PrintObj implements ResourcePrinter
func (fn ResourcePrinterFunc) PrintObj(obj runtime.Object, w io.Writer) error {
	return fn(obj, w)
}

// TODO: implement HandledResources()
func (fn ResourcePrinterFunc) HandledResources() []string {
	return []string{}
}

func (fn ResourcePrinterFunc) AfterPrint(io.Writer, string) error {
	return nil
}

// VersionedPrinter takes runtime objects and ensures they are converted to a given API version
// prior to being passed to a nested printer.
type VersionedPrinter struct {
	printer   ResourcePrinter
	converter runtime.ObjectConvertor
	versions  []unversioned.GroupVersion
}

// NewVersionedPrinter wraps a printer to convert objects to a known API version prior to printing.
func NewVersionedPrinter(printer ResourcePrinter, converter runtime.ObjectConvertor, versions ...unversioned.GroupVersion) ResourcePrinter {
	return &VersionedPrinter{
		printer:   printer,
		converter: converter,
		versions:  versions,
	}
}

func (p *VersionedPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj implements ResourcePrinter
func (p *VersionedPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if len(p.versions) == 0 {
		return fmt.Errorf("no version specified, object cannot be converted")
	}
	converted, err := p.converter.ConvertToVersion(obj, unversioned.GroupVersions(p.versions))
	if err != nil {
		return err
	}
	return p.printer.PrintObj(converted, w)
}

// TODO: implement HandledResources()
func (p *VersionedPrinter) HandledResources() []string {
	return []string{}
}

// NamePrinter is an implementation of ResourcePrinter which outputs "resource/name" pair of an object.
type NamePrinter struct {
	Decoder runtime.Decoder
	Typer   runtime.ObjectTyper
}

func (p *NamePrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which decodes the object
// and print "resource/name" pair. If the object is a List, print all items in it.
func (p *NamePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	if meta.IsListType(obj) {
		items, err := meta.ExtractList(obj)
		if err != nil {
			return err
		}
		if errs := runtime.DecodeList(items, p.Decoder, runtime.UnstructuredJSONScheme); len(errs) > 0 {
			return utilerrors.NewAggregate(errs)
		}
		for _, obj := range items {
			if err := p.PrintObj(obj, w); err != nil {
				return err
			}
		}
		return nil
	}

	name := "<unknown>"
	if acc, err := meta.Accessor(obj); err == nil {
		if n := acc.GetName(); len(n) > 0 {
			name = n
		}
	}

	if kind := obj.GetObjectKind().GroupVersionKind(); len(kind.Kind) == 0 {
		// this is the old code.  It's unnecessary on decoded external objects, but on internal objects
		// you may have to do it.  Tests are definitely calling it with internals and I'm not sure who else
		// is
		if gvks, _, err := p.Typer.ObjectKinds(obj); err == nil {
			// TODO: this is wrong, it assumes that meta knows about all Kinds - should take a RESTMapper
			_, resource := meta.KindToResource(gvks[0])
			fmt.Fprintf(w, "%s/%s\n", resource.Resource, name)
		} else {
			fmt.Fprintf(w, "<unknown>/%s\n", name)
		}

	} else {
		// TODO: this is wrong, it assumes that meta knows about all Kinds - should take a RESTMapper
		_, resource := meta.KindToResource(kind)
		fmt.Fprintf(w, "%s/%s\n", resource.Resource, name)
	}

	return nil
}

// TODO: implement HandledResources()
func (p *NamePrinter) HandledResources() []string {
	return []string{}
}

// JSONPrinter is an implementation of ResourcePrinter which outputs an object as JSON.
type JSONPrinter struct {
}

func (p *JSONPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj is an implementation of ResourcePrinter.PrintObj which simply writes the object to the Writer.
func (p *JSONPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	switch obj := obj.(type) {
	case *runtime.Unknown:
		var buf bytes.Buffer
		err := json.Indent(&buf, obj.Raw, "", "    ")
		if err != nil {
			return err
		}
		buf.WriteRune('\n')
		_, err = buf.WriteTo(w)
		return err
	}

	data, err := json.MarshalIndent(obj, "", "    ")
	if err != nil {
		return err
	}
	data = append(data, '\n')
	_, err = w.Write(data)
	return err
}

// TODO: implement HandledResources()
func (p *JSONPrinter) HandledResources() []string {
	return []string{}
}

// YAMLPrinter is an implementation of ResourcePrinter which outputs an object as YAML.
// The input object is assumed to be in the internal version of an API and is converted
// to the given version first.
type YAMLPrinter struct {
	version   string
	converter runtime.ObjectConvertor
}

func (p *YAMLPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj prints the data as YAML.
func (p *YAMLPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	switch obj := obj.(type) {
	case *runtime.Unknown:
		data, err := yaml.JSONToYAML(obj.Raw)
		if err != nil {
			return err
		}
		_, err = w.Write(data)
		return err
	}

	output, err := yaml.Marshal(obj)
	if err != nil {
		return err
	}
	_, err = fmt.Fprint(w, string(output))
	return err
}

// TODO: implement HandledResources()
func (p *YAMLPrinter) HandledResources() []string {
	return []string{}
}

type handlerEntry struct {
	columns   []string
	printFunc reflect.Value
	args      []reflect.Value
}

type PrintOptions struct {
	NoHeaders          bool
	WithNamespace      bool
	WithKind           bool
	Wide               bool
	ShowAll            bool
	ShowLabels         bool
	AbsoluteTimestamps bool
	Kind               string
	ColumnLabels       []string
}

// HumanReadablePrinter is an implementation of ResourcePrinter which attempts to provide
// more elegant output. It is not threadsafe, but you may call PrintObj repeatedly; headers
// will only be printed if the object type changes. This makes it useful for printing items
// received from watches.
type HumanReadablePrinter struct {
	handlerMap   map[reflect.Type]*handlerEntry
	options      PrintOptions
	lastType     reflect.Type
	hiddenObjNum int
}

// NewHumanReadablePrinter creates a HumanReadablePrinter.
func NewHumanReadablePrinter(options PrintOptions) *HumanReadablePrinter {
	printer := &HumanReadablePrinter{
		handlerMap: make(map[reflect.Type]*handlerEntry),
		options:    options,
	}
	printer.addDefaultHandlers()
	return printer
}

// formatResourceName receives a resource kind, name, and boolean specifying
// whether or not to update the current name to "kind/name"
func formatResourceName(kind, name string, withKind bool) string {
	if !withKind || kind == "" {
		return name
	}

	return kind + "/" + name
}

// GetResourceKind returns the type currently set for a resource
func (h *HumanReadablePrinter) GetResourceKind() string {
	return h.options.Kind
}

// EnsurePrintWithKind sets HumanReadablePrinter options "WithKind" to true
// and "Kind" to the string arg it receives, pre-pending this string
// to the "NAME" column in an output of resources.
func (h *HumanReadablePrinter) EnsurePrintWithKind(kind string) {
	h.options.WithKind = true
	h.options.Kind = kind
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
//  func printFunc(object ObjectType, w io.Writer, options PrintOptions) error
// where ObjectType is the type of the object that will be printed.
func (h *HumanReadablePrinter) validatePrintHandlerFunc(printFunc reflect.Value) error {
	if printFunc.Kind() != reflect.Func {
		return fmt.Errorf("invalid print handler. %#v is not a function", printFunc)
	}
	funcType := printFunc.Type()
	if funcType.NumIn() != 3 || funcType.NumOut() != 1 {
		return fmt.Errorf("invalid print handler." +
			"Must accept 3 parameters and return 1 value.")
	}
	if funcType.In(1) != reflect.TypeOf((*io.Writer)(nil)).Elem() ||
		funcType.In(2) != reflect.TypeOf((*PrintOptions)(nil)).Elem() ||
		funcType.Out(0) != reflect.TypeOf((*error)(nil)).Elem() {
		return fmt.Errorf("invalid print handler. The expected signature is: "+
			"func handler(obj %v, w io.Writer, options PrintOptions) error", funcType.In(0))
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

func (h *HumanReadablePrinter) AfterPrint(output io.Writer, res string) error {
	return nil
}

// NOTE: When adding a new resource type here, please update the list
// pkg/kubectl/cmd/get.go to reflect the new resource type.
var (
	podColumns                   = []string{"NAME", "READY", "STATUS", "RESTARTS", "AGE"}
	podTemplateColumns           = []string{"TEMPLATE", "CONTAINER(S)", "IMAGE(S)", "PODLABELS"}
	podDisruptionBudgetColumns   = []string{"NAME", "MIN-AVAILABLE", "ALLOWED-DISRUPTIONS", "AGE"}
	replicationControllerColumns = []string{"NAME", "DESIRED", "CURRENT", "READY", "AGE"}
	replicaSetColumns            = []string{"NAME", "DESIRED", "CURRENT", "READY", "AGE"}
	jobColumns                   = []string{"NAME", "DESIRED", "SUCCESSFUL", "AGE"}
	cronJobColumns               = []string{"NAME", "SCHEDULE", "SUSPEND", "ACTIVE", "LAST-SCHEDULE"}
	serviceColumns               = []string{"NAME", "CLUSTER-IP", "EXTERNAL-IP", "PORT(S)", "AGE"}
	ingressColumns               = []string{"NAME", "HOSTS", "ADDRESS", "PORTS", "AGE"}
	statefulSetColumns           = []string{"NAME", "DESIRED", "CURRENT", "AGE"}
	endpointColumns              = []string{"NAME", "ENDPOINTS", "AGE"}
	nodeColumns                  = []string{"NAME", "STATUS", "AGE"}
	daemonSetColumns             = []string{"NAME", "DESIRED", "CURRENT", "READY", "NODE-SELECTOR", "AGE"}
	eventColumns                 = []string{"LASTSEEN", "FIRSTSEEN", "COUNT", "NAME", "KIND", "SUBOBJECT", "TYPE", "REASON", "SOURCE", "MESSAGE"}
	limitRangeColumns            = []string{"NAME", "AGE"}
	resourceQuotaColumns         = []string{"NAME", "AGE"}
	namespaceColumns             = []string{"NAME", "STATUS", "AGE"}
	secretColumns                = []string{"NAME", "TYPE", "DATA", "AGE"}
	serviceAccountColumns        = []string{"NAME", "SECRETS", "AGE"}
	persistentVolumeColumns      = []string{"NAME", "CAPACITY", "ACCESSMODES", "RECLAIMPOLICY", "STATUS", "CLAIM", "REASON", "AGE"}
	persistentVolumeClaimColumns = []string{"NAME", "STATUS", "VOLUME", "CAPACITY", "ACCESSMODES", "AGE"}
	componentStatusColumns       = []string{"NAME", "STATUS", "MESSAGE", "ERROR"}
	thirdPartyResourceColumns    = []string{"NAME", "DESCRIPTION", "VERSION(S)"}
	roleColumns                  = []string{"NAME", "AGE"}
	roleBindingColumns           = []string{"NAME", "AGE"}
	clusterRoleColumns           = []string{"NAME", "AGE"}
	clusterRoleBindingColumns    = []string{"NAME", "AGE"}
	storageClassColumns          = []string{"NAME", "TYPE"}
	statusColumns                = []string{"STATUS", "REASON", "MESSAGE"}

	// TODO: consider having 'KIND' for third party resource data
	thirdPartyResourceDataColumns    = []string{"NAME", "LABELS", "DATA"}
	horizontalPodAutoscalerColumns   = []string{"NAME", "REFERENCE", "TARGET", "CURRENT", "MINPODS", "MAXPODS", "AGE"}
	withNamespacePrefixColumns       = []string{"NAMESPACE"} // TODO(erictune): print cluster name too.
	deploymentColumns                = []string{"NAME", "DESIRED", "CURRENT", "UP-TO-DATE", "AVAILABLE", "AGE"}
	configMapColumns                 = []string{"NAME", "DATA", "AGE"}
	podSecurityPolicyColumns         = []string{"NAME", "PRIV", "CAPS", "VOLUMEPLUGINS", "SELINUX", "RUNASUSER"}
	clusterColumns                   = []string{"NAME", "STATUS", "AGE"}
	networkPolicyColumns             = []string{"NAME", "POD-SELECTOR", "AGE"}
	certificateSigningRequestColumns = []string{"NAME", "AGE", "REQUESTOR", "CONDITION"}
)

func (h *HumanReadablePrinter) printPod(pod *api.Pod, w io.Writer, options PrintOptions) error {
	if err := printPodBase(pod, w, options); err != nil {
		return err
	}

	return nil
}

func (h *HumanReadablePrinter) printPodList(podList *api.PodList, w io.Writer, options PrintOptions) error {
	for _, pod := range podList.Items {
		if err := printPodBase(&pod, w, options); err != nil {
			return err
		}
	}
	return nil
}

// addDefaultHandlers adds print handlers for default Kubernetes types.
func (h *HumanReadablePrinter) addDefaultHandlers() {
	h.Handler(podColumns, h.printPodList)
	h.Handler(podColumns, h.printPod)
	h.Handler(podTemplateColumns, printPodTemplate)
	h.Handler(podTemplateColumns, printPodTemplateList)
	h.Handler(podDisruptionBudgetColumns, printPodDisruptionBudget)
	h.Handler(podDisruptionBudgetColumns, printPodDisruptionBudgetList)
	h.Handler(replicationControllerColumns, printReplicationController)
	h.Handler(replicationControllerColumns, printReplicationControllerList)
	h.Handler(replicaSetColumns, printReplicaSet)
	h.Handler(replicaSetColumns, printReplicaSetList)
	h.Handler(daemonSetColumns, printDaemonSet)
	h.Handler(daemonSetColumns, printDaemonSetList)
	h.Handler(jobColumns, printJob)
	h.Handler(jobColumns, printJobList)
	h.Handler(cronJobColumns, printCronJob)
	h.Handler(cronJobColumns, printCronJobList)
	h.Handler(serviceColumns, printService)
	h.Handler(serviceColumns, printServiceList)
	h.Handler(ingressColumns, printIngress)
	h.Handler(ingressColumns, printIngressList)
	h.Handler(statefulSetColumns, printStatefulSet)
	h.Handler(statefulSetColumns, printStatefulSetList)
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
	h.Handler(thirdPartyResourceColumns, printThirdPartyResource)
	h.Handler(thirdPartyResourceColumns, printThirdPartyResourceList)
	h.Handler(deploymentColumns, printDeployment)
	h.Handler(deploymentColumns, printDeploymentList)
	h.Handler(horizontalPodAutoscalerColumns, printHorizontalPodAutoscaler)
	h.Handler(horizontalPodAutoscalerColumns, printHorizontalPodAutoscalerList)
	h.Handler(configMapColumns, printConfigMap)
	h.Handler(configMapColumns, printConfigMapList)
	h.Handler(podSecurityPolicyColumns, printPodSecurityPolicy)
	h.Handler(podSecurityPolicyColumns, printPodSecurityPolicyList)
	h.Handler(thirdPartyResourceDataColumns, printThirdPartyResourceData)
	h.Handler(thirdPartyResourceDataColumns, printThirdPartyResourceDataList)
	h.Handler(clusterColumns, printCluster)
	h.Handler(clusterColumns, printClusterList)
	h.Handler(networkPolicyColumns, printNetworkPolicy)
	h.Handler(networkPolicyColumns, printNetworkPolicyList)
	h.Handler(roleColumns, printRole)
	h.Handler(roleColumns, printRoleList)
	h.Handler(roleBindingColumns, printRoleBinding)
	h.Handler(roleBindingColumns, printRoleBindingList)
	h.Handler(clusterRoleColumns, printClusterRole)
	h.Handler(clusterRoleColumns, printClusterRoleList)
	h.Handler(clusterRoleBindingColumns, printClusterRoleBinding)
	h.Handler(clusterRoleBindingColumns, printClusterRoleBindingList)
	h.Handler(certificateSigningRequestColumns, printCertificateSigningRequest)
	h.Handler(certificateSigningRequestColumns, printCertificateSigningRequestList)
	h.Handler(storageClassColumns, printStorageClass)
	h.Handler(storageClassColumns, printStorageClassList)
	h.Handler(statusColumns, printStatus)
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
func formatEndpoints(endpoints *api.Endpoints, ports sets.String) string {
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
	// Allow deviation no more than 2 seconds(excluded) to tolerate machine time
	// inconsistence, it can be considered as almost now.
	if seconds := int(d.Seconds()); seconds < -1 {
		return fmt.Sprintf("<invalid>")
	} else if seconds < 0 {
		return fmt.Sprintf("0s")
	} else if seconds < 60 {
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
func translateTimestamp(timestamp unversioned.Time) string {
	if timestamp.IsZero() {
		return "<unknown>"
	}
	return shortHumanDuration(time.Now().Sub(timestamp.Time))
}

func printPodBase(pod *api.Pod, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, pod.Name, options.WithKind)
	namespace := pod.Namespace

	restarts := 0
	totalContainers := len(pod.Spec.Containers)
	readyContainers := 0

	reason := string(pod.Status.Phase)
	if pod.Status.Reason != "" {
		reason = pod.Status.Reason
	}

	initializing := false
	for i := range pod.Status.InitContainerStatuses {
		container := pod.Status.InitContainerStatuses[i]
		restarts += int(container.RestartCount)
		switch {
		case container.State.Terminated != nil && container.State.Terminated.ExitCode == 0:
			continue
		case container.State.Terminated != nil:
			// initialization is failed
			if len(container.State.Terminated.Reason) == 0 {
				if container.State.Terminated.Signal != 0 {
					reason = fmt.Sprintf("Init:Signal:%d", container.State.Terminated.Signal)
				} else {
					reason = fmt.Sprintf("Init:ExitCode:%d", container.State.Terminated.ExitCode)
				}
			} else {
				reason = "Init:" + container.State.Terminated.Reason
			}
			initializing = true
		case container.State.Waiting != nil && len(container.State.Waiting.Reason) > 0 && container.State.Waiting.Reason != "PodInitializing":
			reason = "Init:" + container.State.Waiting.Reason
			initializing = true
		default:
			reason = fmt.Sprintf("Init:%d/%d", i, len(pod.Spec.InitContainers))
			initializing = true
		}
		break
	}
	if !initializing {
		restarts = 0
		for i := len(pod.Status.ContainerStatuses) - 1; i >= 0; i-- {
			container := pod.Status.ContainerStatuses[i]

			restarts += int(container.RestartCount)
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
	}

	if pod.DeletionTimestamp != nil && pod.Status.Reason == node.NodeUnreachablePodReason {
		reason = "Unknown"
	} else if pod.DeletionTimestamp != nil {
		reason = "Terminating"
	}

	if options.WithNamespace {
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

	if options.Wide {
		nodeName := pod.Spec.NodeName
		podIP := pod.Status.PodIP
		if podIP == "" {
			podIP = "<none>"
		}
		if _, err := fmt.Fprintf(w, "\t%s\t%s",
			podIP,
			nodeName,
		); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprint(w, AppendLabels(pod.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pod.Labels)); err != nil {
		return err
	}

	return nil
}

func printPodTemplate(pod *api.PodTemplate, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, pod.Name, options.WithKind)

	namespace := pod.Namespace

	containers := pod.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s", name); err != nil {
		return err
	}
	if err := layoutContainers(containers, w); err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(pod.Template.Labels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pod.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pod.Labels)); err != nil {
		return err
	}

	return nil
}

func printPodTemplateList(podList *api.PodTemplateList, w io.Writer, options PrintOptions) error {
	for _, pod := range podList.Items {
		if err := printPodTemplate(&pod, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPodDisruptionBudget(pdb *policy.PodDisruptionBudget, w io.Writer, options PrintOptions) error {
	// name, minavailable, selector
	name := formatResourceName(options.Kind, pdb.Name, options.WithKind)
	namespace := pdb.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s\n",
		name,
		pdb.Spec.MinAvailable.String(),
		pdb.Status.PodDisruptionsAllowed,
		translateTimestamp(pdb.CreationTimestamp),
	); err != nil {
		return err
	}

	return nil
}

func printPodDisruptionBudgetList(pdbList *policy.PodDisruptionBudgetList, w io.Writer, options PrintOptions) error {
	for _, pdb := range pdbList.Items {
		if err := printPodDisruptionBudget(&pdb, w, options); err != nil {
			return err
		}
	}
	return nil
}

// TODO(AdoHe): try to put wide output in a single method
func printReplicationController(controller *api.ReplicationController, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, controller.Name, options.WithKind)

	namespace := controller.Namespace
	containers := controller.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredReplicas := controller.Spec.Replicas
	currentReplicas := controller.Status.Replicas
	readyReplicas := controller.Status.ReadyReplicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		readyReplicas,
		translateTimestamp(controller.CreationTimestamp),
	); err != nil {
		return err
	}

	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(controller.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(controller.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, controller.Labels)); err != nil {
		return err
	}

	return nil
}

func printReplicationControllerList(list *api.ReplicationControllerList, w io.Writer, options PrintOptions) error {
	for _, controller := range list.Items {
		if err := printReplicationController(&controller, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printReplicaSet(rs *extensions.ReplicaSet, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, rs.Name, options.WithKind)

	namespace := rs.Namespace
	containers := rs.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredReplicas := rs.Spec.Replicas
	currentReplicas := rs.Status.Replicas
	readyReplicas := rs.Status.ReadyReplicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		readyReplicas,
		translateTimestamp(rs.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", unversioned.FormatLabelSelector(rs.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(rs.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, rs.Labels)); err != nil {
		return err
	}

	return nil
}

func printReplicaSetList(list *extensions.ReplicaSetList, w io.Writer, options PrintOptions) error {
	for _, rs := range list.Items {
		if err := printReplicaSet(&rs, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCluster(c *federation.Cluster, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, c.Name, options.WithKind)

	var statuses []string
	for _, condition := range c.Status.Conditions {
		if condition.Status == api.ConditionTrue {
			statuses = append(statuses, string(condition.Type))
		} else {
			statuses = append(statuses, "Not"+string(condition.Type))
		}
	}
	if len(statuses) == 0 {
		statuses = append(statuses, "Unknown")
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n",
		name,
		strings.Join(statuses, ","),
		translateTimestamp(c.CreationTimestamp),
	); err != nil {
		return err
	}
	return nil
}
func printClusterList(list *federation.ClusterList, w io.Writer, options PrintOptions) error {
	for _, rs := range list.Items {
		if err := printCluster(&rs, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printJob(job *batch.Job, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, job.Name, options.WithKind)

	namespace := job.Namespace
	containers := job.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	selector, err := unversioned.LabelSelectorAsSelector(job.Spec.Selector)
	if err != nil {
		// this shouldn't happen if LabelSelector passed validation
		return err
	}
	if job.Spec.Completions != nil {
		if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%s",
			name,
			*job.Spec.Completions,
			job.Status.Succeeded,
			translateTimestamp(job.CreationTimestamp),
		); err != nil {
			return err
		}
	} else {
		if _, err := fmt.Fprintf(w, "%s\t%s\t%d\t%s",
			name,
			"<none>",
			job.Status.Succeeded,
			translateTimestamp(job.CreationTimestamp),
		); err != nil {
			return err
		}
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", selector.String()); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(job.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, job.Labels)); err != nil {
		return err
	}

	return nil
}

func printJobList(list *batch.JobList, w io.Writer, options PrintOptions) error {
	for _, job := range list.Items {
		if err := printJob(&job, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCronJob(cronJob *batch.CronJob, w io.Writer, options PrintOptions) error {
	name := cronJob.Name
	namespace := cronJob.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	lastScheduleTime := "<none>"
	if cronJob.Status.LastScheduleTime != nil {
		lastScheduleTime = cronJob.Status.LastScheduleTime.Time.Format(time.RFC1123Z)
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%d\t%s\n",
		name,
		cronJob.Spec.Schedule,
		printBoolPtr(cronJob.Spec.Suspend),
		len(cronJob.Status.Active),
		lastScheduleTime,
	); err != nil {
		return err
	}

	return nil
}

func printCronJobList(list *batch.CronJobList, w io.Writer, options PrintOptions) error {
	for _, cronJob := range list.Items {
		if err := printCronJob(&cronJob, w, options); err != nil {
			return err
		}
	}
	return nil
}

// loadBalancerStatusStringer behaves mostly like a string interface and converts the given status to a string.
// `wide` indicates whether the returned value is meant for --o=wide output. If not, it's clipped to 16 bytes.
func loadBalancerStatusStringer(s api.LoadBalancerStatus, wide bool) string {
	ingress := s.Ingress
	result := []string{}
	for i := range ingress {
		if ingress[i].IP != "" {
			result = append(result, ingress[i].IP)
		} else if ingress[i].Hostname != "" {
			result = append(result, ingress[i].Hostname)
		}
	}
	r := strings.Join(result, ",")
	if !wide && len(r) > loadBalancerWidth {
		r = r[0:(loadBalancerWidth-3)] + "..."
	}
	return r
}

func getServiceExternalIP(svc *api.Service, wide bool) string {
	switch svc.Spec.Type {
	case api.ServiceTypeClusterIP:
		if len(svc.Spec.ExternalIPs) > 0 {
			return strings.Join(svc.Spec.ExternalIPs, ",")
		}
		return "<none>"
	case api.ServiceTypeNodePort:
		if len(svc.Spec.ExternalIPs) > 0 {
			return strings.Join(svc.Spec.ExternalIPs, ",")
		}
		return "<nodes>"
	case api.ServiceTypeLoadBalancer:
		lbIps := loadBalancerStatusStringer(svc.Status.LoadBalancer, wide)
		if len(svc.Spec.ExternalIPs) > 0 {
			result := append(strings.Split(lbIps, ","), svc.Spec.ExternalIPs...)
			return strings.Join(result, ",")
		}
		if len(lbIps) > 0 {
			return lbIps
		}
		return "<pending>"
	case api.ServiceTypeExternalName:
		return svc.Spec.ExternalName
	}
	return "<unknown>"
}

func makePortString(ports []api.ServicePort) string {
	pieces := make([]string, len(ports))
	for ix := range ports {
		port := &ports[ix]
		pieces[ix] = fmt.Sprintf("%d/%s", port.Port, port.Protocol)
		if port.NodePort > 0 {
			pieces[ix] = fmt.Sprintf("%d:%d/%s", port.Port, port.NodePort, port.Protocol)
		}
	}
	return strings.Join(pieces, ",")
}

func printService(svc *api.Service, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, svc.Name, options.WithKind)

	namespace := svc.Namespace

	internalIP := svc.Spec.ClusterIP
	externalIP := getServiceExternalIP(svc, options.Wide)

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s",
		name,
		internalIP,
		externalIP,
		makePortString(svc.Spec.Ports),
		translateTimestamp(svc.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if _, err := fmt.Fprintf(w, "\t%s", labels.FormatLabels(svc.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(svc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, svc.Labels))
	return err
}

func printServiceList(list *api.ServiceList, w io.Writer, options PrintOptions) error {
	for _, svc := range list.Items {
		if err := printService(&svc, w, options); err != nil {
			return err
		}
	}
	return nil
}

// backendStringer behaves just like a string interface and converts the given backend to a string.
func backendStringer(backend *extensions.IngressBackend) string {
	if backend == nil {
		return ""
	}
	return fmt.Sprintf("%v:%v", backend.ServiceName, backend.ServicePort.String())
}

func formatHosts(rules []extensions.IngressRule) string {
	list := []string{}
	max := 3
	more := false
	for _, rule := range rules {
		if len(list) == max {
			more = true
		}
		if !more && len(rule.Host) != 0 {
			list = append(list, rule.Host)
		}
	}
	if len(list) == 0 {
		return "*"
	}
	ret := strings.Join(list, ",")
	if more {
		return fmt.Sprintf("%s + %d more...", ret, len(rules)-max)
	}
	return ret
}

func formatPorts(tls []extensions.IngressTLS) string {
	if len(tls) != 0 {
		return "80, 443"
	}
	return "80"
}

func printIngress(ingress *extensions.Ingress, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, ingress.Name, options.WithKind)

	namespace := ingress.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%v\t%v\t%v\t%s",
		name,
		formatHosts(ingress.Spec.Rules),
		loadBalancerStatusStringer(ingress.Status.LoadBalancer, options.Wide),
		formatPorts(ingress.Spec.TLS),
		translateTimestamp(ingress.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(ingress.Labels, options.ColumnLabels)); err != nil {
		return err
	}

	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ingress.Labels)); err != nil {
		return err
	}
	return nil
}

func printIngressList(ingressList *extensions.IngressList, w io.Writer, options PrintOptions) error {
	for _, ingress := range ingressList.Items {
		if err := printIngress(&ingress, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStatefulSet(ps *apps.StatefulSet, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, ps.Name, options.WithKind)

	namespace := ps.Namespace
	containers := ps.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	desiredReplicas := ps.Spec.Replicas
	currentReplicas := ps.Status.Replicas
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%s",
		name,
		desiredReplicas,
		currentReplicas,
		translateTimestamp(ps.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", unversioned.FormatLabelSelector(ps.Spec.Selector)); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(ps.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ps.Labels)); err != nil {
		return err
	}

	return nil
}

func printStatefulSetList(statefulSetList *apps.StatefulSetList, w io.Writer, options PrintOptions) error {
	for _, ps := range statefulSetList.Items {
		if err := printStatefulSet(&ps, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printDaemonSet(ds *extensions.DaemonSet, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, ds.Name, options.WithKind)

	namespace := ds.Namespace

	containers := ds.Spec.Template.Spec.Containers

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	desiredScheduled := ds.Status.DesiredNumberScheduled
	currentScheduled := ds.Status.CurrentNumberScheduled
	numberReady := ds.Status.NumberReady
	selector, err := unversioned.LabelSelectorAsSelector(ds.Spec.Selector)
	if err != nil {
		// this shouldn't happen if LabelSelector passed validation
		return err
	}
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%s\t%s",
		name,
		desiredScheduled,
		currentScheduled,
		numberReady,
		labels.FormatLabels(ds.Spec.Template.Spec.NodeSelector),
		translateTimestamp(ds.CreationTimestamp),
	); err != nil {
		return err
	}
	if options.Wide {
		if err := layoutContainers(containers, w); err != nil {
			return err
		}
		if _, err := fmt.Fprintf(w, "\t%s", selector.String()); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprint(w, AppendLabels(ds.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, ds.Labels)); err != nil {
		return err
	}

	return nil
}

func printDaemonSetList(list *extensions.DaemonSetList, w io.Writer, options PrintOptions) error {
	for _, ds := range list.Items {
		if err := printDaemonSet(&ds, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printEndpoints(endpoints *api.Endpoints, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, endpoints.Name, options.WithKind)

	namespace := endpoints.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, formatEndpoints(endpoints, nil), translateTimestamp(endpoints.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(endpoints.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, endpoints.Labels))
	return err
}

func printEndpointsList(list *api.EndpointsList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printEndpoints(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printNamespace(item *api.Namespace, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("namespace is not namespaced")
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, item.Status.Phase, translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printNamespaceList(list *api.NamespaceList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printNamespace(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printSecret(item *api.Secret, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	namespace := item.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%v\t%s", name, item.Type, len(item.Data), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printSecretList(list *api.SecretList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printSecret(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printServiceAccount(item *api.ServiceAccount, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	namespace := item.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%d\t%s", name, len(item.Secrets), translateTimestamp(item.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printServiceAccountList(list *api.ServiceAccountList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printServiceAccount(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printNode(node *api.Node, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, node.Name, options.WithKind)

	if options.WithNamespace {
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
	role := findNodeRole(node)
	if role != "" {
		status = append(status, role)
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s", name, strings.Join(status, ","), translateTimestamp(node.CreationTimestamp)); err != nil {
		return err
	}

	if options.Wide {
		if _, err := fmt.Fprintf(w, "\t%s", getNodeExternalIP(node)); err != nil {
			return err
		}
	}
	// Display caller specify column labels first.
	if _, err := fmt.Fprint(w, AppendLabels(node.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, node.Labels))
	return err
}

// Returns first external ip of the node or "<none>" if none is found.
func getNodeExternalIP(node *api.Node) string {
	for _, address := range node.Status.Addresses {
		if address.Type == api.NodeExternalIP {
			return address.Address
		}
	}

	return "<none>"
}

// findNodeRole returns the role of a given node, or "" if none found.
// The role is determined by looking in order for:
// * a kubernetes.io/role label
// * a kubeadm.alpha.kubernetes.io/role label
// If no role is found, ("", nil) is returned
func findNodeRole(node *api.Node) string {
	if role := node.Labels[unversioned.NodeLabelRole]; role != "" {
		return role
	}
	if role := node.Labels[unversioned.NodeLabelKubeadmAlphaRole]; role != "" {
		return role
	}
	// No role found
	return ""
}

func printNodeList(list *api.NodeList, w io.Writer, options PrintOptions) error {
	for _, node := range list.Items {
		if err := printNode(&node, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolume(pv *api.PersistentVolume, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, pv.Name, options.WithKind)

	if options.WithNamespace {
		return fmt.Errorf("persistentVolume is not namespaced")
	}

	claimRefUID := ""
	if pv.Spec.ClaimRef != nil {
		claimRefUID += pv.Spec.ClaimRef.Namespace
		claimRefUID += "/"
		claimRefUID += pv.Spec.ClaimRef.Name
	}

	modesStr := api.GetAccessModesAsString(pv.Spec.AccessModes)
	reclaimPolicyStr := string(pv.Spec.PersistentVolumeReclaimPolicy)

	aQty := pv.Spec.Capacity[api.ResourceStorage]
	aSize := aQty.String()

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
		name,
		aSize, modesStr, reclaimPolicyStr,
		pv.Status.Phase,
		claimRefUID,
		pv.Status.Reason,
		translateTimestamp(pv.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pv.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pv.Labels))
	return err
}

func printPersistentVolumeList(list *api.PersistentVolumeList, w io.Writer, options PrintOptions) error {
	for _, pv := range list.Items {
		if err := printPersistentVolume(&pv, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPersistentVolumeClaim(pvc *api.PersistentVolumeClaim, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, pvc.Name, options.WithKind)

	namespace := pvc.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	phase := pvc.Status.Phase
	storage := pvc.Spec.Resources.Requests[api.ResourceStorage]
	capacity := ""
	accessModes := ""
	if pvc.Spec.VolumeName != "" {
		accessModes = api.GetAccessModesAsString(pvc.Status.AccessModes)
		storage = pvc.Status.Capacity[api.ResourceStorage]
		capacity = storage.String()
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%s", name, phase, pvc.Spec.VolumeName, capacity, accessModes, translateTimestamp(pvc.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(pvc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, pvc.Labels))
	return err
}

func printPersistentVolumeClaimList(list *api.PersistentVolumeClaimList, w io.Writer, options PrintOptions) error {
	for _, psd := range list.Items {
		if err := printPersistentVolumeClaim(&psd, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printEvent(event *api.Event, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, event.InvolvedObject.Name, options.WithKind)

	namespace := event.Namespace
	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	// While watching event, we should print absolute time.
	var FirstTimestamp, LastTimestamp string
	if options.AbsoluteTimestamps {
		FirstTimestamp = event.FirstTimestamp.String()
		LastTimestamp = event.LastTimestamp.String()
	} else {
		FirstTimestamp = translateTimestamp(event.FirstTimestamp)
		LastTimestamp = translateTimestamp(event.LastTimestamp)
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s\t%d\t%s\t%s\t%s\t%s\t%s\t%s\t%s",
		LastTimestamp,
		FirstTimestamp,
		event.Count,
		name,
		event.InvolvedObject.Kind,
		event.InvolvedObject.FieldPath,
		event.Type,
		event.Reason,
		event.Source,
		event.Message,
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(event.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, event.Labels))
	return err
}

// Sorts and prints the EventList in a human-friendly format.
func printEventList(list *api.EventList, w io.Writer, options PrintOptions) error {
	sort.Sort(events.SortableEvents(list.Items))
	for i := range list.Items {
		if err := printEvent(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printLimitRange(limitRange *api.LimitRange, w io.Writer, options PrintOptions) error {
	return printObjectMeta(limitRange.ObjectMeta, w, options, true)
}

// Prints the LimitRangeList in a human-friendly format.
func printLimitRangeList(list *api.LimitRangeList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printLimitRange(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

// printObjectMeta prints the object metadata of a given resource.
func printObjectMeta(meta api.ObjectMeta, w io.Writer, options PrintOptions, namespaced bool) error {
	name := formatResourceName(options.Kind, meta.Name, options.WithKind)

	if namespaced && options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", meta.Namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

func printResourceQuota(resourceQuota *api.ResourceQuota, w io.Writer, options PrintOptions) error {
	return printObjectMeta(resourceQuota.ObjectMeta, w, options, true)
}

// Prints the ResourceQuotaList in a human-friendly format.
func printResourceQuotaList(list *api.ResourceQuotaList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printResourceQuota(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printRole(role *rbac.Role, w io.Writer, options PrintOptions) error {
	return printObjectMeta(role.ObjectMeta, w, options, true)
}

// Prints the Role in a human-friendly format.
func printRoleList(list *rbac.RoleList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printRole(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printRoleBinding(roleBinding *rbac.RoleBinding, w io.Writer, options PrintOptions) error {
	return printObjectMeta(roleBinding.ObjectMeta, w, options, true)
}

// Prints the RoleBinding in a human-friendly format.
func printRoleBindingList(list *rbac.RoleBindingList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printRoleBinding(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printClusterRole(clusterRole *rbac.ClusterRole, w io.Writer, options PrintOptions) error {
	return printObjectMeta(clusterRole.ObjectMeta, w, options, false)
}

// Prints the ClusterRole in a human-friendly format.
func printClusterRoleList(list *rbac.ClusterRoleList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printClusterRole(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printClusterRoleBinding(clusterRoleBinding *rbac.ClusterRoleBinding, w io.Writer, options PrintOptions) error {
	return printObjectMeta(clusterRoleBinding.ObjectMeta, w, options, false)
}

// Prints the ClusterRoleBinding in a human-friendly format.
func printClusterRoleBindingList(list *rbac.ClusterRoleBindingList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printClusterRoleBinding(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printCertificateSigningRequest(csr *certificates.CertificateSigningRequest, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, csr.Name, options.WithKind)
	meta := csr.ObjectMeta

	status, err := extractCSRStatus(csr)
	if err != nil {
		return err
	}

	if _, err := fmt.Fprintf(
		w, "%s\t%s\t%s\t%s",
		name,
		translateTimestamp(meta.CreationTimestamp),
		csr.Spec.Username,
		status,
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(meta.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err = fmt.Fprint(w, AppendAllLabels(options.ShowLabels, meta.Labels))
	return err
}

func extractCSRStatus(csr *certificates.CertificateSigningRequest) (string, error) {
	var approved, denied bool
	for _, c := range csr.Status.Conditions {
		switch c.Type {
		case certificates.CertificateApproved:
			approved = true
		case certificates.CertificateDenied:
			denied = true
		default:
			return "", fmt.Errorf("unknown csr condition %q", c)
		}
	}
	var status string
	// must be in order of presidence
	if denied {
		status += "Denied"
	} else if approved {
		status += "Approved"
	} else {
		status += "Pending"
	}
	if len(csr.Status.Certificate) > 0 {
		status += ",Issued"
	}
	return status, nil
}

func printCertificateSigningRequestList(list *certificates.CertificateSigningRequestList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printCertificateSigningRequest(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printComponentStatus(item *api.ComponentStatus, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	if options.WithNamespace {
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

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s", name, status, message, error); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(item.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, item.Labels))
	return err
}

func printComponentStatusList(list *api.ComponentStatusList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printComponentStatus(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printThirdPartyResource(rsrc *extensions.ThirdPartyResource, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, rsrc.Name, options.WithKind)

	versions := make([]string, len(rsrc.Versions))
	for ix := range rsrc.Versions {
		version := &rsrc.Versions[ix]
		versions[ix] = fmt.Sprintf("%s", version.Name)
	}
	versionsString := strings.Join(versions, ",")
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", name, rsrc.Description, versionsString); err != nil {
		return err
	}
	return nil
}

func printThirdPartyResourceList(list *extensions.ThirdPartyResourceList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printThirdPartyResource(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func truncate(str string, maxLen int) string {
	if len(str) > maxLen {
		return str[0:maxLen] + "..."
	}
	return str
}

func printThirdPartyResourceData(rsrc *extensions.ThirdPartyResourceData, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, rsrc.Name, options.WithKind)

	l := labels.FormatLabels(rsrc.Labels)
	truncateCols := 50
	if options.Wide {
		truncateCols = 100
	}
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", name, l, truncate(string(rsrc.Data), truncateCols)); err != nil {
		return err
	}
	return nil
}

func printThirdPartyResourceDataList(list *extensions.ThirdPartyResourceDataList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printThirdPartyResourceData(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printDeployment(deployment *extensions.Deployment, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, deployment.Name, options.WithKind)

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", deployment.Namespace); err != nil {
			return err
		}
	}

	desiredReplicas := deployment.Spec.Replicas
	currentReplicas := deployment.Status.Replicas
	updatedReplicas := deployment.Status.UpdatedReplicas
	availableReplicas := deployment.Status.AvailableReplicas
	age := translateTimestamp(deployment.CreationTimestamp)
	if _, err := fmt.Fprintf(w, "%s\t%d\t%d\t%d\t%d\t%s", name, desiredReplicas, currentReplicas, updatedReplicas, availableReplicas, age); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(deployment.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, deployment.Labels))
	return err
}

func printDeploymentList(list *extensions.DeploymentList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printDeployment(&item, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printHorizontalPodAutoscaler(hpa *autoscaling.HorizontalPodAutoscaler, w io.Writer, options PrintOptions) error {
	namespace := hpa.Namespace
	name := formatResourceName(options.Kind, hpa.Name, options.WithKind)

	reference := fmt.Sprintf("%s/%s",
		hpa.Spec.ScaleTargetRef.Kind,
		hpa.Spec.ScaleTargetRef.Name)
	target := "<unset>"
	if hpa.Spec.TargetCPUUtilizationPercentage != nil {
		target = fmt.Sprintf("%d%%", *hpa.Spec.TargetCPUUtilizationPercentage)
	}
	current := "<waiting>"
	if hpa.Status.CurrentCPUUtilizationPercentage != nil {
		current = fmt.Sprintf("%d%%", *hpa.Status.CurrentCPUUtilizationPercentage)
	}
	minPods := "<unset>"
	if hpa.Spec.MinReplicas != nil {
		minPods = fmt.Sprintf("%d", *hpa.Spec.MinReplicas)
	}
	maxPods := hpa.Spec.MaxReplicas

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}

	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\t%s\t%s\t%d\t%s",
		name,
		reference,
		target,
		current,
		minPods,
		maxPods,
		translateTimestamp(hpa.CreationTimestamp),
	); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(hpa.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, hpa.Labels))
	return err
}

func printHorizontalPodAutoscalerList(list *autoscaling.HorizontalPodAutoscalerList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printHorizontalPodAutoscaler(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printConfigMap(configMap *api.ConfigMap, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, configMap.Name, options.WithKind)

	namespace := configMap.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%v\t%s", name, len(configMap.Data), translateTimestamp(configMap.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(configMap.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, configMap.Labels))
	return err
}

func printConfigMapList(list *api.ConfigMapList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printConfigMap(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printPodSecurityPolicy(item *extensions.PodSecurityPolicy, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, item.Name, options.WithKind)

	_, err := fmt.Fprintf(w, "%s\t%t\t%v\t%s\t%s\t%s\t%s\t%t\t%v\n", name, item.Spec.Privileged,
		item.Spec.AllowedCapabilities, item.Spec.SELinux.Rule,
		item.Spec.RunAsUser.Rule, item.Spec.FSGroup.Rule, item.Spec.SupplementalGroups.Rule, item.Spec.ReadOnlyRootFilesystem, item.Spec.Volumes)
	return err
}

func printPodSecurityPolicyList(list *extensions.PodSecurityPolicyList, w io.Writer, options PrintOptions) error {
	for _, item := range list.Items {
		if err := printPodSecurityPolicy(&item, w, options); err != nil {
			return err
		}
	}

	return nil
}

func printNetworkPolicy(networkPolicy *extensions.NetworkPolicy, w io.Writer, options PrintOptions) error {
	name := formatResourceName(options.Kind, networkPolicy.Name, options.WithKind)

	namespace := networkPolicy.Namespace

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", namespace); err != nil {
			return err
		}
	}
	if _, err := fmt.Fprintf(w, "%s\t%v\t%s", name, unversioned.FormatLabelSelector(&networkPolicy.Spec.PodSelector), translateTimestamp(networkPolicy.CreationTimestamp)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(networkPolicy.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	_, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, networkPolicy.Labels))
	return err
}

func printNetworkPolicyList(list *extensions.NetworkPolicyList, w io.Writer, options PrintOptions) error {
	for i := range list.Items {
		if err := printNetworkPolicy(&list.Items[i], w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStorageClass(sc *storage.StorageClass, w io.Writer, options PrintOptions) error {
	name := sc.Name

	if storageutil.IsDefaultAnnotation(sc.ObjectMeta) {
		name += " (default)"
	}
	provtype := sc.Provisioner

	if _, err := fmt.Fprintf(w, "%s\t%s\t", name, provtype); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(sc.Labels, options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, sc.Labels)); err != nil {
		return err
	}

	return nil
}

func printStorageClassList(scList *storage.StorageClassList, w io.Writer, options PrintOptions) error {
	for _, sc := range scList.Items {
		if err := printStorageClass(&sc, w, options); err != nil {
			return err
		}
	}
	return nil
}

func printStatus(status *unversioned.Status, w io.Writer, options PrintOptions) error {
	if _, err := fmt.Fprintf(w, "%s\t%s\t%s\n", status.Status, status.Reason, status.Message); err != nil {
		return err
	}

	return nil
}

func AppendLabels(itemLabels map[string]string, columnLabels []string) string {
	var buffer bytes.Buffer

	for _, cl := range columnLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		if il, ok := itemLabels[cl]; ok {
			buffer.WriteString(fmt.Sprint(il))
		} else {
			buffer.WriteString("<none>")
		}
	}

	return buffer.String()
}

// Append all labels to a single column. We need this even when show-labels flag* is
// false, since this adds newline delimiter to the end of each row.
func AppendAllLabels(showLabels bool, itemLabels map[string]string) string {
	var buffer bytes.Buffer

	if showLabels {
		buffer.WriteString(fmt.Sprint("\t"))
		buffer.WriteString(labels.FormatLabels(itemLabels))
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// Append a set of tabs for each label column.  We need this in the case where
// we have extra lines so that the tabwriter will still line things up.
func AppendLabelTabs(columnLabels []string) string {
	var buffer bytes.Buffer

	for range columnLabels {
		buffer.WriteString("\t")
	}
	buffer.WriteString("\n")

	return buffer.String()
}

// Lay out all the containers on one line if use wide output.
func layoutContainers(containers []api.Container, w io.Writer) error {
	var namesBuffer bytes.Buffer
	var imagesBuffer bytes.Buffer

	for i, container := range containers {
		namesBuffer.WriteString(container.Name)
		imagesBuffer.WriteString(container.Image)
		if i != len(containers)-1 {
			namesBuffer.WriteString(",")
			imagesBuffer.WriteString(",")
		}
	}
	_, err := fmt.Fprintf(w, "\t%s\t%s", namesBuffer.String(), imagesBuffer.String())
	if err != nil {
		return err
	}
	return nil
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
			return []string{"IP", "NODE"}
		}
		if t.String() == "*api.ReplicationController" || t.String() == "*api.ReplicationControllerList" {
			return []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
		}
		if t.String() == "*batch.Job" || t.String() == "*batch.JobList" {
			return []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
		}
		if t.String() == "*api.Service" || t.String() == "*api.ServiceList" {
			return []string{"SELECTOR"}
		}
		if t.String() == "*extensions.DaemonSet" || t.String() == "*extensions.DaemonSetList" {
			return []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
		}
		if t.String() == "*extensions.ReplicaSet" || t.String() == "*extensions.ReplicaSetList" {
			return []string{"CONTAINER(S)", "IMAGE(S)", "SELECTOR"}
		}
		if t.String() == "*api.Node" || t.String() == "*api.NodeList" {
			return []string{"EXTERNAL-IP"}
		}
	}
	return nil
}

// headers for --show-labels=true
func formatShowLabelsHeader(showLabels bool, t reflect.Type) []string {
	if showLabels {
		if t.String() != "*api.ThirdPartyResource" && t.String() != "*api.ThirdPartyResourceList" {
			return []string{"LABELS"}
		}
	}
	return nil
}

// GetNewTabWriter returns a tabwriter that translates tabbed columns in input into properly aligned text.
func GetNewTabWriter(output io.Writer) *tabwriter.Writer {
	return tabwriter.NewWriter(output, tabwriterMinWidth, tabwriterWidth, tabwriterPadding, tabwriterPadChar, tabwriterFlags)
}

// PrintObj prints the obj in a human-friendly format according to the type of the obj.
func (h *HumanReadablePrinter) PrintObj(obj runtime.Object, output io.Writer) error {
	// if output is a tabwriter (when it's called by kubectl get), we use it; create a new tabwriter otherwise
	w, found := output.(*tabwriter.Writer)
	if !found {
		w = GetNewTabWriter(output)
		defer w.Flush()
	}

	// check if the object is unstructured.  If so, let's attempt to convert it to a type we can understand before
	// trying to print, since the printers are keyed by type.  This is extremely expensive.
	switch obj.(type) {
	case *runtime.UnstructuredList, *runtime.Unstructured, *runtime.Unknown:
		if objBytes, err := runtime.Encode(api.Codecs.LegacyCodec(), obj); err == nil {
			if decodedObj, err := runtime.Decode(api.Codecs.UniversalDecoder(), objBytes); err == nil {
				obj = decodedObj
			}
		}
	}

	t := reflect.TypeOf(obj)
	if handler := h.handlerMap[t]; handler != nil {
		if !h.options.NoHeaders && t != h.lastType {
			headers := append(handler.columns, formatWideHeaders(h.options.Wide, t)...)
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, w)
			h.lastType = t
		}
		args := []reflect.Value{reflect.ValueOf(obj), reflect.ValueOf(w), reflect.ValueOf(h.options)}
		resultValue := handler.printFunc.Call(args)[0]
		if resultValue.IsNil() {
			return nil
		}
		return resultValue.Interface().(error)
	}

	// we don't recognize this type, but we can still attempt to print some reasonable information about.
	unstructured, ok := obj.(*runtime.Unstructured)
	if !ok {
		return fmt.Errorf("error: unknown type %#v", obj)
	}

	if _, err := meta.Accessor(obj); err == nil {
		if !h.options.NoHeaders && t != h.lastType {
			headers := []string{"NAME", "KIND"}
			headers = append(headers, formatLabelHeaders(h.options.ColumnLabels)...)
			// LABELS is always the last column.
			headers = append(headers, formatShowLabelsHeader(h.options.ShowLabels, t)...)
			if h.options.WithNamespace {
				headers = append(withNamespacePrefixColumns, headers...)
			}
			h.printHeader(headers, w)
			h.lastType = t
		}
		// if the error isn't nil, report the "I don't recognize this" error
		if err := printUnstructured(unstructured, w, h.options); err != nil {
			return err
		}
		return nil
	}

	// we failed all reasonable printing efforts, report failure
	return fmt.Errorf("error: unknown type %#v", obj)
}

func printUnstructured(unstructured *runtime.Unstructured, w io.Writer, options PrintOptions) error {
	metadata, err := meta.Accessor(unstructured)
	if err != nil {
		return err
	}

	if options.WithNamespace {
		if _, err := fmt.Fprintf(w, "%s\t", metadata.GetNamespace()); err != nil {
			return err
		}
	}

	kind := "<missing>"
	if objKind, ok := unstructured.Object["kind"]; ok {
		if str, ok := objKind.(string); ok {
			kind = str
		}
	}
	if objAPIVersion, ok := unstructured.Object["apiVersion"]; ok {
		if str, ok := objAPIVersion.(string); ok {
			version, err := unversioned.ParseGroupVersion(str)
			if err != nil {
				return err
			}
			kind = kind + "." + version.Version + "." + version.Group
		}
	}
	name := formatResourceName(options.Kind, metadata.GetName(), options.WithKind)

	if _, err := fmt.Fprintf(w, "%s\t%s", name, kind); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendLabels(metadata.GetLabels(), options.ColumnLabels)); err != nil {
		return err
	}
	if _, err := fmt.Fprint(w, AppendAllLabels(options.ShowLabels, metadata.GetLabels())); err != nil {
		return err
	}

	return nil
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

func (p *TemplatePrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj formats the obj with the Go Template.
func (p *TemplatePrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	var data []byte
	var err error
	if unstructured, ok := obj.(*runtime.Unstructured); ok {
		data, err = json.Marshal(unstructured.Object)
	} else {
		data, err = json.Marshal(obj)

	}
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
		fmt.Fprintf(w, "Error executing template: %v. Printing more information for debugging the template:\n", err)
		fmt.Fprintf(w, "\ttemplate was:\n\t\t%v\n", p.rawTemplate)
		fmt.Fprintf(w, "\traw data was:\n\t\t%v\n", string(data))
		fmt.Fprintf(w, "\tobject given to template engine was:\n\t\t%+v\n\n", out)
		return fmt.Errorf("error executing template %q: %v", p.rawTemplate, err)
	}
	return nil
}

// TODO: implement HandledResources()
func (p *TemplatePrinter) HandledResources() []string {
	return []string{}
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

// JSONPathPrinter is an implementation of ResourcePrinter which formats data with jsonpath expression.
type JSONPathPrinter struct {
	rawTemplate string
	*jsonpath.JSONPath
}

func NewJSONPathPrinter(tmpl string) (*JSONPathPrinter, error) {
	j := jsonpath.New("out")
	if err := j.Parse(tmpl); err != nil {
		return nil, err
	}
	return &JSONPathPrinter{tmpl, j}, nil
}

func (j *JSONPathPrinter) AfterPrint(w io.Writer, res string) error {
	return nil
}

// PrintObj formats the obj with the JSONPath Template.
func (j *JSONPathPrinter) PrintObj(obj runtime.Object, w io.Writer) error {
	var queryObj interface{} = obj
	if meta.IsListType(obj) {
		data, err := json.Marshal(obj)
		if err != nil {
			return err
		}
		queryObj = map[string]interface{}{}
		if err := json.Unmarshal(data, &queryObj); err != nil {
			return err
		}
	}

	if unknown, ok := obj.(*runtime.Unknown); ok {
		data, err := json.Marshal(unknown)
		if err != nil {
			return err
		}
		queryObj = map[string]interface{}{}
		if err := json.Unmarshal(data, &queryObj); err != nil {
			return err
		}
	}
	if unstructured, ok := obj.(*runtime.Unstructured); ok {
		queryObj = unstructured.Object
	}

	if err := j.JSONPath.Execute(w, queryObj); err != nil {
		fmt.Fprintf(w, "Error executing template: %v. Printing more information for debugging the template:\n", err)
		fmt.Fprintf(w, "\ttemplate was:\n\t\t%v\n", j.rawTemplate)
		fmt.Fprintf(w, "\tobject given to jsonpath engine was:\n\t\t%#v\n\n", queryObj)
		return fmt.Errorf("error executing jsonpath %q: %v\n", j.rawTemplate, err)
	}
	return nil
}

// TODO: implement HandledResources()
func (p *JSONPathPrinter) HandledResources() []string {
	return []string{}
}
