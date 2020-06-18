//events.go

package events

import (

	"fmt"
	"io"
	"net/url"
	"strings"
	"regexp"

	"github.com/spf13/cobra"

	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/apimachinery/pkg/runtime"
	utilerrors "k8s.io/apimachinery/pkg/util/errors"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/cli-runtime/pkg/genericclioptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/cli-runtime/pkg/resource"
	kubernetesscheme "k8s.io/client-go/kubernetes/scheme"
	"k8s.io/client-go/rest"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/rawhttp"
	"k8s.io/kubectl/pkg/scheme"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
)

// EventsOptions contains the input to the events command.
type EventsOptions struct {
	PrintFlags             *PrintFlags
	ToPrinter   func(*meta.RESTMapping, *bool, bool, bool) (printers.ResourcePrinterFunc, error)
	IsHumanReadablePrinter bool
	PrintWithOpenAPICols   bool


	resource.FilenameOptions

	Raw       string
	Watch     bool
	WatchOnly bool
	ChunkSize int64

	OutputWatchEvents bool

	LabelSelector     string
	FieldSelector     string
	AllNamespaces     bool
	Namespace         string
	ExplicitNamespace bool

	ServerPrint bool

	NoHeaders      bool
	Sort           bool
	IgnoreNotFound bool
	Export         bool
	SortBy       string

	genericclioptions.IOStreams
	builder *resource.Builder

}

var (
	eventsLong = templates.LongDesc(`
		Display one or many resources

		Prints a table of the most important information about the specified resources.
		You can filter the list using a label selector and the --selector flag. If the
		desired resource type is namespaced you will only see results in your current
		namespace unless you pass --all-namespaces.

		Uninitialized objects are not shown unless --include-uninitialized is passed.

		By specifying the output as 'template' and providing a Go template as the value
		of the --template flag, you can filter the attributes of the fetched resources.`)

	eventsExample = templates.Examples(i18n.T(`
		# List all pods in ps output format.
		kubectl alpha events

		# List all pods in ps output format with more information (such as node name).
		kubectl alpha events -o wide`))
)

const (
	useOpenAPIPrintColumnFlagLabel = "use-openapi-print-columns"
	useServerPrintColumns          = "server-print"
)

// NewGetOptions returns a GetOptions with default chunk size 500.
func NewEventOptions(streams genericclioptions.IOStreams) *EventsOptions {
	return &EventsOptions{
		PrintFlags:  NewGetPrintFlags(),
		IOStreams:   streams,
		//ChunkSize:   500,
		ServerPrint: true,
	}
}

// NewCmdEvents creates a command object for "events"
// which retrieves a list of events from the server.
func NewCmdEvents(f cmdutil.Factory, streams genericclioptions.IOStreams) *cobra.Command {
	o := NewEventOptions(streams)

	cmd := &cobra.Command{
		Use:                   "events [(-o|--output=)json|yaml|wide",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display one or many resources"),
		Long:                  eventsLong,
		Example:               eventsExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate(cmd))
			cmdutil.CheckErr(o.Run(f, cmd, args))
		},
		SuggestFor: []string{"list", "ps"},
	}

	o.PrintFlags.AddFlags(cmd)
	//cmd.Flags().StringVar(&o.Raw, "raw", o.Raw, "Raw URI to request from the server.  Uses the transport specified by the kubeconfig file.")
	//cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "After listing/getting the requested object, watch for changes. Uninitialized objects are excluded if no object name is provided.")
	//cmd.Flags().BoolVar(&o.WatchOnly, "watch-only", o.WatchOnly, "Watch for changes to the requested object(s), without listing/getting first.")
	//cmd.Flags().BoolVar(&o.OutputWatchEvents, "output-watch-events", o.OutputWatchEvents, "Output watch event objects when --watch or --watch-only is used. Existing objects are output as initial ADDED events.")
	//cmd.Flags().Int64Var(&o.ChunkSize, "chunk-size", o.ChunkSize, "Return large lists in chunks rather than all at once. Pass 0 to disable. This flag is beta and may change in the future.")
	cmd.Flags().BoolVar(&o.IgnoreNotFound, "ignore-not-found", o.IgnoreNotFound, "If the requested object does not exist the command will return exit code 0.")
	//cmd.Flags().StringVarP(&o.LabelSelector, "selector", "l", o.LabelSelector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	//cmd.Flags().StringVar(&o.FieldSelector, "field-selector", o.FieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&o.Export, "export", o.Export, "If true, use 'export' for the resources.  Exported resources are stripped of cluster-specific information.")
	cmd.Flags().MarkDeprecated("export", "This flag is deprecated and will be removed in future.")

	return cmd
}


func (o *EventsOptions) Run(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(o.Raw) > 0 {
		restClient, err := f.RESTClient()
		if err != nil {
			return err
		}
		return rawhttp.RawGet(restClient, o.IOStreams, o.Raw)
	}


	chunkSize := o.ChunkSize
	if o.Sort {
		// TODO(juanvallejo): in the future, we could have the client use chunking
		// to gather all results, then sort them all at the end to reduce server load.
		//chunkSize = 0
	}

	r := f.NewBuilder().
		Unstructured().
		NamespaceParam(o.Namespace).DefaultNamespace().AllNamespaces(o.AllNamespaces).
		FilenameParam(o.ExplicitNamespace, &o.FilenameOptions).
		LabelSelectorParam(o.LabelSelector).
		FieldSelectorParam(o.FieldSelector).
		ExportParam(o.Export).
		RequestChunksOf(chunkSize).
		ResourceTypeOrNameArgs(true, "events").
		ContinueOnError().
		Latest().
		Flatten().
		TransformRequests(o.transformRequests).
		Do()

	if o.IgnoreNotFound {
		r.IgnoreErrors(apierrors.IsNotFound)
	}
	if err := r.Err(); err != nil {
		return err
	}

	allErrs := []error{}
	errs := sets.NewString()
	infos, err := r.Infos()
	if err != nil {
		allErrs = append(allErrs, err)
	}

	objs := make([]runtime.Object, len(infos))
	for ix := range infos {
		objs[ix] = infos[ix].Object
	}


	sorting, err := cmd.Flags().GetString("sort-by")
	if err != nil {
		return err
	}

	var positioner OriginalPositioner
	if o.Sort {
		sorter := NewRuntimeSorter(objs, sorting)
		if err := sorter.Sort(); err != nil {
			return err
		}
	}

	var printer printers.ResourcePrinter
	var lastMapping *meta.RESTMapping

	// track if we write any output
	trackingWriter := &trackingWriterWrapper{Delegate: o.Out}
	// output an empty line separating output
	separatorWriter := &separatorWriterWrapper{Delegate: trackingWriter}

	w := printers.GetNewTabWriter(separatorWriter)
	for ix := range objs {
		var mapping *meta.RESTMapping
		var info *resource.Info
		if positioner != nil {
			info = infos[positioner.OriginalPosition(ix)]
			mapping = info.Mapping
		} else {
			info = infos[ix]
			mapping = info.Mapping
		}

		printWithNamespace := o.AllNamespaces

		if mapping != nil && mapping.Scope.Name() == meta.RESTScopeNameRoot {
			printWithNamespace = false
		}

		if shouldGetNewPrinterForMapping(printer, lastMapping, mapping) {
			w.Flush()
			w.SetRememberedWidths(nil)

			// add linebreaks between resource groups (if there is more than one)
			// when it satisfies all following 3 conditions:
			// 1) it's not the first resource group
			// 2) it has row header
			// 3) we've written output since the last time we started a new set of headers
			if lastMapping != nil && !o.NoHeaders && trackingWriter.Written > 0 {
				separatorWriter.SetReady(true)
			}

			printer, err = o.ToPrinter( mapping, nil, printWithNamespace, false)

			if err != nil {
				if !errs.Has(err.Error()) {
					errs.Insert(err.Error())
					allErrs = append(allErrs, err)
				}
				continue
			}

			lastMapping = mapping
		}

		// ensure a versioned object is passed to the custom-columns printer
		// if we are using OpenAPI columns to print
		if o.PrintWithOpenAPICols {
			printer.PrintObj(info.Object, w)
			continue
		}

		printer.PrintObj(info.Object, w)
	}
	w.Flush()
	if trackingWriter.Written == 0 && !o.IgnoreNotFound && len(allErrs) == 0 {
		// if we wrote no output, and had no errors, and are not ignoring NotFound, be sure we output something
		if !o.AllNamespaces {
			fmt.Fprintln(o.ErrOut, fmt.Sprintf("No resources found in %s namespace.", o.Namespace))
		} else {
			fmt.Fprintln(o.ErrOut, fmt.Sprintf("No resources found"))
		}
	}
	return utilerrors.NewAggregate(allErrs)
}

// Complete takes the command arguments and factory and infers any remaining options.
func (o *EventsOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	if len(o.Raw) > 0 {
		if len(args) > 0 {
			return fmt.Errorf("arguments may not be passed when --raw is specified")
		}
		return nil
	}

	var err error
	o.Namespace, o.ExplicitNamespace, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	if o.AllNamespaces {
		o.ExplicitNamespace = false
	}

	sortBy, err := cmd.Flags().GetString("sort-by")
	if err != nil {
		return err
	}
	o.Sort = len(sortBy) > 0

	o.NoHeaders = cmdutil.GetFlagBool(cmd, "no-headers")

	outputOption := cmd.Flags().Lookup("output").Value.String()
	if outputOption == "custom-columns" {
		o.ServerPrint = false
	}

	templateArg := ""
	if o.PrintFlags.TemplateFlags != nil && o.PrintFlags.TemplateFlags.TemplateArgument != nil {
		templateArg = *o.PrintFlags.TemplateFlags.TemplateArgument
	}

	if (len(*o.PrintFlags.OutputFormat) == 0 && len(templateArg) == 0) || *o.PrintFlags.OutputFormat == "wide" {
		o.IsHumanReadablePrinter = true
	}

	o.ToPrinter = func(mapping *meta.RESTMapping, outputObjects *bool, withNamespace bool, withKind bool) (printers.ResourcePrinterFunc, error) {
		// make a new copy of current flags / opts before mutating
		printFlags := o.PrintFlags.Copy()

		printer, err := printFlags.ToPrinter()
		if err != nil {
			return nil, err
		}
		printer, err = printers.NewTypeSetter(scheme.Scheme).WrapToPrinter(printer, nil)
		if err != nil {
			return nil, err
		}


		if o.ServerPrint {
			printer = &TablePrinter{Delegate: printer}
		}
		return printer.PrintObj, nil
	}



	switch {
	case o.Watch || o.WatchOnly:
		if o.Sort {
			fmt.Fprintf(o.IOStreams.ErrOut, "warning: --watch or --watch-only requested, --sort-by will be ignored\n")
		}
	default:

	}

	// openapi printing is mutually exclusive with server side printing
	if o.PrintWithOpenAPICols && o.ServerPrint {
		fmt.Fprintf(o.IOStreams.ErrOut, "warning: --%s requested, --%s will be ignored\n", useOpenAPIPrintColumnFlagLabel, useServerPrintColumns)
	}

	return nil
}

// Validate checks the set of flags provided by the user.
func (o *EventsOptions) Validate(cmd *cobra.Command) error {
	if len(o.Raw) > 0 {
		if o.Watch || o.WatchOnly || len(o.LabelSelector) > 0 || o.Export {
			return fmt.Errorf("--raw may not be specified with other flags that filter the server request or alter the output")
		}

		if _, err := url.ParseRequestURI(o.Raw); err != nil {
			return cmdutil.UsageErrorf(cmd, "--raw must be a valid URL path: %v", err)
		}
	}

	if o.OutputWatchEvents && !(o.Watch || o.WatchOnly) {
		return cmdutil.UsageErrorf(cmd, "--output-watch-events option can only be used with --watch or --watch-only")
	}
	return nil
}

func addOpenAPIPrintColumnFlags(cmd *cobra.Command, opt *EventsOptions) {
	cmd.Flags().BoolVar(&opt.PrintWithOpenAPICols, useOpenAPIPrintColumnFlagLabel, opt.PrintWithOpenAPICols, "If true, use x-kubernetes-print-column metadata (if present) from the OpenAPI schema for displaying a resource.")
	cmd.Flags().MarkDeprecated(useOpenAPIPrintColumnFlagLabel, "deprecated in favor of server-side printing")
}

func addServerPrintColumnFlags(cmd *cobra.Command, opt *EventsOptions) {
	cmd.Flags().BoolVar(&opt.ServerPrint, useServerPrintColumns, opt.ServerPrint, "If true, have the server return the appropriate table output. Supports extension APIs and CRDs.")
}


func (o *EventsOptions) transformRequests(req *rest.Request) {
	// We need full objects if printing with openapi columns
	if o.PrintWithOpenAPICols {
		return
	}
	if !o.ServerPrint || !o.IsHumanReadablePrinter {
		return
	}

	req.SetHeader("Accept", strings.Join([]string{
		fmt.Sprintf("application/json;as=Table;v=%s;g=%s", metav1.SchemeGroupVersion.Version, metav1.GroupName),
		fmt.Sprintf("application/json;as=Table;v=%s;g=%s", metav1beta1.SchemeGroupVersion.Version, metav1beta1.GroupName),
		"application/json",
	}, ","))

	// if sorting, ensure we receive the full object in order to introspect its fields via jsonpath
	if o.Sort {
		req.Param("includeObject", "Object")
	}
}


// OriginalPositioner and NopPositioner is required for swap/sort operations of data in table format
type OriginalPositioner interface {
	OriginalPosition(int) int
}

// NopPositioner and OriginalPositioner is required for swap/sort operations of data in table format
type NopPositioner struct{}

// OriginalPosition returns the original position from NopPositioner object
func (t *NopPositioner) OriginalPosition(ix int) int {
	return ix
}



// RuntimeSorter holds the required objects to perform sorting of runtime objects
type RuntimeSorter struct {
	field      string
	decoder    runtime.Decoder
	objects    []runtime.Object
	positioner OriginalPositioner
}

// OriginalPosition returns the original position of a runtime object
func (r *RuntimeSorter) OriginalPosition(ix int) int {
	if r.positioner == nil {
		return 0
	}
	return r.positioner.OriginalPosition(ix)
}

// Sort performs the sorting of runtime objects
func (r *RuntimeSorter) Sort() error {
	// a list is only considered "sorted" if there are 0 or 1 items in it
	// AND (if 1 item) the item is not a Table object
	if len(r.objects) == 0 {
		return nil
	}
	if len(r.objects) == 1 {
		_, isTable := r.objects[0].(*metav1.Table)
		if !isTable {
			return nil
		}
	}

	includesTable := false
	includesRuntimeObjs := false

	for _, obj := range r.objects {
		switch obj.(type) {
		case *metav1.Table:
			includesTable = true
		default:
			includesRuntimeObjs = true
		}
	}

	// we use a NopPositioner when dealing with Table objects
	// because the objects themselves are not swapped, but rather
	// the rows in each object are swapped / sorted.
	r.positioner = &NopPositioner{}

	if includesRuntimeObjs && includesTable {
		return fmt.Errorf("sorting is not supported on mixed Table and non-Table object lists")
	}
	if includesTable {
		return nil
	}

	return nil
}

// NewRuntimeSorter returns a new instance of RuntimeSorter
func NewRuntimeSorter(objects []runtime.Object, sortBy string) *RuntimeSorter {
	parsedField, err := RelaxedJSONPathExpression(sortBy)
	if err != nil {
		parsedField = sortBy
	}

	return &RuntimeSorter{
		field:   parsedField,
		decoder: kubernetesscheme.Codecs.UniversalDecoder(),
		objects: objects,
	}
}


type trackingWriterWrapper struct {
	Delegate io.Writer
	Written  int
}

func (t *trackingWriterWrapper) Write(p []byte) (n int, err error) {
	t.Written += len(p)
	return t.Delegate.Write(p)
}

type separatorWriterWrapper struct {
	Delegate io.Writer
	Ready    bool
}

func (s *separatorWriterWrapper) Write(p []byte) (n int, err error) {
	// If we're about to write non-empty bytes and `s` is ready,
	// we prepend an empty line to `p` and reset `s.Read`.
	if len(p) != 0 && s.Ready {
		fmt.Fprintln(s.Delegate)
		s.Ready = false
	}
	return s.Delegate.Write(p)
}

func (s *separatorWriterWrapper) SetReady(state bool) {
	s.Ready = state
}

func shouldGetNewPrinterForMapping(printer printers.ResourcePrinter, lastMapping, mapping *meta.RESTMapping) bool {
	return printer == nil || lastMapping == nil || mapping == nil || mapping.Resource != lastMapping.Resource
}

func cmdSpecifiesOutputFmt(cmd *cobra.Command) bool {
//	return cmdutil.GetFlagString(cmd, "output") != ""
return true
}



var jsonRegexp = regexp.MustCompile(`^\{\.?([^{}]+)\}$|^\.?([^{}]+)$`)

// RelaxedJSONPathExpression attempts to be flexible with JSONPath expressions, it accepts:
//   * metadata.name (no leading '.' or curly braces '{...}'
//   * {metadata.name} (no leading '.')
//   * .metadata.name (no curly braces '{...}')
//   * {.metadata.name} (complete expression)
// And transforms them all into a valid jsonpath expression:
//   {.metadata.name}
func RelaxedJSONPathExpression(pathExpression string) (string, error) {
	if len(pathExpression) == 0 {
		return pathExpression, nil
	}
	submatches := jsonRegexp.FindStringSubmatch(pathExpression)
	if submatches == nil {
		return "", fmt.Errorf("unexpected path string, expected a 'name1.name2' or '.name1.name2' or '{name1.name2}' or '{.name1.name2}'")
	}
	if len(submatches) != 3 {
		return "", fmt.Errorf("unexpected submatch list: %v", submatches)
	}
	var fieldSpec string
	if len(submatches[1]) != 0 {
		fieldSpec = submatches[1]
	} else {
		fieldSpec = submatches[2]
	}
	return fmt.Sprintf("{.%s}", fieldSpec), nil
}


