/*
Copyright 2016 The Kubernetes Authors.

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

package top

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"time"

	"github.com/spf13/cobra"
	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/resource"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/cli-runtime/pkg/genericiooptions"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/client-go/discovery"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	"k8s.io/klog/v2"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/metricsutil"
	"k8s.io/kubectl/pkg/util/completion"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"
)

type TopPodOptions struct {
	ResourceName       string
	Namespace          string
	LabelSelector      string
	FieldSelector      string
	SortBy             string
	AllNamespaces      bool
	PrintContainers    bool
	NoHeaders          bool
	UseProtocolBuffers bool
	Sum                bool
	ShowSwap           bool
	Watch              bool

	PodClient       corev1client.PodsGetter
	Printer         *metricsutil.TopCmdPrinter
	DiscoveryClient discovery.DiscoveryInterface
	MetricsClient   metricsclientset.Interface

	genericiooptions.IOStreams
}

const metricsCreationDelay = 2 * time.Minute

var (
	topPodLong = templates.LongDesc(i18n.T(`
		Display resource (CPU/memory) usage of pods.

		The 'top pod' command allows you to see the resource consumption of pods.

		Due to the metrics pipeline delay, they may be unavailable for a few minutes
		since pod creation.`))

	topPodExample = templates.Examples(i18n.T(`
		# Show metrics for all pods in the default namespace
		kubectl top pod

		# Show metrics for all pods in the given namespace
		kubectl top pod --namespace=NAMESPACE

		# Show metrics for a given pod and its containers
		kubectl top pod POD_NAME --containers

		# Show metrics for the pods defined by label name=myLabel
		kubectl top pod -l name=myLabel`))
)

func NewCmdTopPod(f cmdutil.Factory, o *TopPodOptions, streams genericiooptions.IOStreams) *cobra.Command {
	if o == nil {
		o = &TopPodOptions{
			IOStreams:          streams,
			UseProtocolBuffers: true,
		}
	}

	cmd := &cobra.Command{
		Use:                   "pod [NAME | -l label]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display resource (CPU/memory) usage of pods"),
		Long:                  topPodLong,
		Example:               topPodExample,
		ValidArgsFunction:     completion.ResourceNameCompletionFunc(f, "pod"),
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunTopPod())
		},
		Aliases: []string{"pods", "po"},
	}
	cmdutil.AddLabelSelectorFlagVar(cmd, &o.LabelSelector)
	cmd.Flags().StringVar(&o.FieldSelector, "field-selector", o.FieldSelector, "Selector (field query) to filter on, supports '=', '==', and '!='.(e.g. --field-selector key1=value1,key2=value2). The server only supports a limited number of field queries per type.")
	cmd.Flags().StringVar(&o.SortBy, "sort-by", o.SortBy, "If non-empty, sort pods list using specified field. The field can be either 'cpu' or 'memory'.")
	cmd.Flags().BoolVar(&o.PrintContainers, "containers", o.PrintContainers, "If present, print usage of containers within a pod.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "If present, print output without headers.")
	cmd.Flags().BoolVar(&o.UseProtocolBuffers, "use-protocol-buffers", o.UseProtocolBuffers, "Enables using protocol-buffers to access Metrics API.")
	cmd.Flags().BoolVar(&o.Sum, "sum", o.Sum, "Print the sum of the resource usage")
	cmd.Flags().BoolVar(&o.ShowSwap, "show-swap", o.ShowSwap, "Print pod resources related to swap memory.")
	cmd.Flags().BoolVarP(&o.Watch, "watch", "w", o.Watch, "After listing the requested pods, watch for changes by polling the metrics API.")
	return cmd
}

func (o *TopPodOptions) Complete(f cmdutil.Factory, cmd *cobra.Command, args []string) error {
	var err error
	if len(args) == 1 {
		o.ResourceName = args[0]
	} else if len(args) > 1 {
		return cmdutil.UsageErrorf(cmd, "%s", cmd.Use)
	}

	o.Namespace, _, err = f.ToRawKubeConfigLoader().Namespace()
	if err != nil {
		return err
	}
	clientset, err := f.KubernetesClientSet()
	if err != nil {
		return err
	}

	o.DiscoveryClient = clientset.DiscoveryClient
	config, err := f.ToRESTConfig()
	if err != nil {
		return err
	}
	if o.UseProtocolBuffers {
		config.ContentType = "application/vnd.kubernetes.protobuf"
	}
	o.MetricsClient, err = metricsclientset.NewForConfig(config)
	if err != nil {
		return err
	}

	o.PodClient = clientset.CoreV1()

	o.Printer = metricsutil.NewTopCmdPrinter(o.Out, o.ShowSwap)
	return nil
}

func (o *TopPodOptions) Validate() error {
	if len(o.SortBy) > 0 {
		if o.SortBy != sortByCPU && o.SortBy != sortByMemory {
			return errors.New("--sort-by accepts only cpu or memory")
		}
	}
	if len(o.ResourceName) > 0 && (len(o.LabelSelector) > 0 || len(o.FieldSelector) > 0) {
		return errors.New("only one of NAME or selector can be provided")
	}
	return nil
}

func (o TopPodOptions) RunTopPod() error {
	var err error
	labelSelector := labels.Everything()
	if len(o.LabelSelector) > 0 {
		labelSelector, err = labels.Parse(o.LabelSelector)
		if err != nil {
			return err
		}
	}
	fieldSelector := fields.Everything()
	if len(o.FieldSelector) > 0 {
		fieldSelector, err = fields.ParseSelector(o.FieldSelector)
		if err != nil {
			return err
		}
	}

	apiGroups, err := o.DiscoveryClient.ServerGroups()
	if err != nil {
		return err
	}

	metricsAPIAvailable := SupportedMetricsAPIVersionAvailable(apiGroups)

	if !metricsAPIAvailable {
		return errors.New("Metrics API not available")
	}

	if o.Watch {
		return o.watchPodMetrics(labelSelector, fieldSelector)
	}

	return o.fetchAndPrintPodMetrics(labelSelector, fieldSelector)
}

func (o TopPodOptions) fetchAndPrintPodMetrics(labelSelector labels.Selector, fieldSelector fields.Selector) error {
	metrics, err := getMetricsFromMetricsAPI(o.MetricsClient, o.Namespace, o.ResourceName, o.AllNamespaces, labelSelector, fieldSelector)
	if err != nil {
		return err
	}

	// First we check why no metrics have been received.
	if len(metrics.Items) == 0 {
		// If the API server query is successful but all the pods are newly created,
		// the metrics are probably not ready yet, so we return the error here in the first place.
		err := verifyEmptyMetrics(o, labelSelector, fieldSelector)
		if err != nil {
			return err
		}

		// if we had no errors, be sure we output something.
		if o.AllNamespaces {
			fmt.Fprintln(o.ErrOut, "No resources found")
		} else {
			fmt.Fprintf(o.ErrOut, "No resources found in %s namespace.\n", o.Namespace)
		}
	}

	return o.Printer.PrintPodMetrics(metrics.Items, o.PrintContainers, o.AllNamespaces, o.NoHeaders, o.SortBy, o.Sum)
}

func (o TopPodOptions) watchPodMetrics(labelSelector labels.Selector, fieldSelector fields.Selector) error {
	ticker := time.NewTicker(2 * time.Second)
	defer ticker.Stop()

	// Store previous metrics for delta calculation
	previousMetrics := make(map[string]corev1.ResourceList)
	var previousMetricsItems []metricsapi.PodMetrics

	// Print initial metrics !
	metrics, err := getMetricsFromMetricsAPI(o.MetricsClient, o.Namespace, o.ResourceName, o.AllNamespaces, labelSelector, fieldSelector)
	if err == nil && len(metrics.Items) > 0 {
		if err := o.printPodMetricsWithDelta(metrics.Items, previousMetrics); err != nil {
			return err
		}
		updatePreviousMetrics(metrics.Items, previousMetrics)
		previousMetricsItems = metrics.Items
	}

	for range ticker.C {
		metrics, err := getMetricsFromMetricsAPI(o.MetricsClient, o.Namespace, o.ResourceName, o.AllNamespaces, labelSelector, fieldSelector)
		if err != nil {
			fmt.Fprintf(o.ErrOut, "Error fetching metrics: %v\n", err)
			continue
		}

		if len(metrics.Items) == 0 {
			if o.AllNamespaces {
				fmt.Fprintln(o.ErrOut, "No resources found")
			} else {
				fmt.Fprintf(o.ErrOut, "No resources found in %s namespace.\n", o.Namespace)
			}
			continue
		}

		// Only update display if metrics changed
		if podMetricsChanged(metrics.Items, previousMetricsItems) {
			// Clear screen using ANSI escape codes (move cursor to home, then clear screen)
			fmt.Fprint(o.Out, "\033[H\033[2J")

			if err := o.printPodMetricsWithDelta(metrics.Items, previousMetrics); err != nil {
				fmt.Fprintf(o.ErrOut, "Error printing metrics: %v\n", err)
			}
			updatePreviousMetrics(metrics.Items, previousMetrics)
			previousMetricsItems = metrics.Items
		}
	}

	return nil
}

func (o TopPodOptions) printPodMetricsWithDelta(metrics []metricsapi.PodMetrics, previousMetrics map[string]corev1.ResourceList) error {
	if len(metrics) == 0 {
		return nil
	}

	// Sort metrics
	if len(o.SortBy) > 0 {
		sort.Sort(metricsutil.NewPodMetricsSorter(metrics, o.AllNamespaces, o.SortBy, []corev1.ResourceName{corev1.ResourceCPU, corev1.ResourceMemory}))
	}

	// Use TabWriter for proper column alignment
	w := printers.GetNewTabWriter(o.Out)
	defer w.Flush()

	// Print header
	if o.AllNamespaces {
		fmt.Fprint(w, "NAMESPACE\t")
	}
	fmt.Fprint(w, "NAME\tCPU(cores)\tCPU(Δ)\tCPU(Δ%)\tMEMORY(bytes)\tMEMORY(Δ)\tMEMORY(Δ%)\n")

	// Print each pod
	for _, m := range metrics {
		podMetrics := getPodMetrics(&m)
		podKey := m.Namespace + "/" + m.Name

		cpuQuantity := podMetrics[corev1.ResourceCPU]
		memQuantity := podMetrics[corev1.ResourceMemory]

		cpuAbsDelta := "-"
		cpuPctDelta := "-"
		memAbsDelta := "-"
		memPctDelta := "-"

		if prev, ok := previousMetrics[podKey]; ok {
			prevCPU := prev[corev1.ResourceCPU]
			prevMem := prev[corev1.ResourceMemory]

			if !prevCPU.IsZero() {
				cpuDiff := cpuQuantity.MilliValue() - prevCPU.MilliValue()
				cpuPctChange := float64(cpuDiff) / float64(prevCPU.MilliValue()) * 100
				cpuAbsDelta = formatAbsoluteDeltaCPU(cpuDiff)
				cpuPctDelta = formatPercentDelta(cpuPctChange)
			}

			if !prevMem.IsZero() {
				memDiff := memQuantity.Value() - prevMem.Value()
				memPctChange := float64(memDiff) / float64(prevMem.Value()) * 100
				memAbsDelta = formatAbsoluteDeltaMemory(memDiff)
				memPctDelta = formatPercentDelta(memPctChange)
			}
		}

		if o.AllNamespaces {
			fmt.Fprintf(w, "%s\t", m.Namespace)
		}
		fmt.Fprintf(w, "%s\t%vm\t%s\t%s\t%vMi\t%s\t%s\n",
			m.Name,
			cpuQuantity.MilliValue(),
			cpuAbsDelta,
			cpuPctDelta,
			memQuantity.Value()/(1024*1024),
			memAbsDelta,
			memPctDelta,
		)
	}

	return nil
}

func getPodMetrics(m *metricsapi.PodMetrics) corev1.ResourceList {
	podMetrics := make(corev1.ResourceList)
	podMetrics[corev1.ResourceCPU] = *resource.NewMilliQuantity(0, resource.DecimalSI)
	podMetrics[corev1.ResourceMemory] = *resource.NewQuantity(0, resource.BinarySI)

	for _, c := range m.Containers {
		cpuQuantity := podMetrics[corev1.ResourceCPU]
		cpuQuantity.Add(c.Usage[corev1.ResourceCPU])
		podMetrics[corev1.ResourceCPU] = cpuQuantity

		memQuantity := podMetrics[corev1.ResourceMemory]
		memQuantity.Add(c.Usage[corev1.ResourceMemory])
		podMetrics[corev1.ResourceMemory] = memQuantity
	}
	return podMetrics
}

func updatePreviousMetrics(metrics []metricsapi.PodMetrics, previousMetrics map[string]corev1.ResourceList) {
	for _, m := range metrics {
		podKey := m.Namespace + "/" + m.Name
		previousMetrics[podKey] = getPodMetrics(&m)
	}
}

func formatPercentDelta(change float64) string {
	if change == 0 {
		return "-"
	}
	sign := ""
	if change > 0 {
		sign = "+"
	}
	return fmt.Sprintf("%s%.1f%%", sign, change)
}

func formatAbsoluteDeltaCPU(diffMillicores int64) string {
	if diffMillicores == 0 {
		return "-"
	}
	sign := ""
	if diffMillicores > 0 {
		sign = "+"
	}
	return fmt.Sprintf("%s%vm", sign, diffMillicores)
}

func formatAbsoluteDeltaMemory(diffBytes int64) string {
	if diffBytes == 0 {
		return "-"
	}
	sign := ""
	if diffBytes > 0 {
		sign = "+"
	}
	diffMi := diffBytes / (1024 * 1024)
	return fmt.Sprintf("%s%vMi", sign, diffMi)
}

func podMetricsChanged(current, previous []metricsapi.PodMetrics) bool {
	if len(current) != len(previous) {
		return true
	}

	// Build a map of previous metrics for quick lookup
	prevMap := make(map[string]corev1.ResourceList)
	for _, p := range previous {
		key := p.Namespace + "/" + p.Name
		prevMap[key] = getPodMetrics(&p)
	}

	// Check if any metrics changed
	for _, curr := range current {
		key := curr.Namespace + "/" + curr.Name
		prev, exists := prevMap[key]
		if !exists {
			return true // New pod
		}

		currMetrics := getPodMetrics(&curr)

		if !currMetrics[corev1.ResourceCPU].Equal(prev[corev1.ResourceCPU]) ||
			!currMetrics[corev1.ResourceMemory].Equal(prev[corev1.ResourceMemory]) {
			return true
		}
	}

	return false
}

func getMetricsFromMetricsAPI(metricsClient metricsclientset.Interface, namespace, resourceName string, allNamespaces bool, labelSelector labels.Selector, fieldSelector fields.Selector) (*metricsapi.PodMetricsList, error) {
	var err error
	ns := metav1.NamespaceAll
	if !allNamespaces {
		ns = namespace
	}
	versionedMetrics := &metricsv1beta1api.PodMetricsList{}
	if resourceName != "" {
		m, err := metricsClient.MetricsV1beta1().PodMetricses(ns).Get(context.TODO(), resourceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		versionedMetrics.Items = []metricsv1beta1api.PodMetrics{*m}
	} else {
		versionedMetrics, err = metricsClient.MetricsV1beta1().PodMetricses(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: labelSelector.String(), FieldSelector: fieldSelector.String()})
		if err != nil {
			return nil, err
		}
	}
	metrics := &metricsapi.PodMetricsList{}
	err = metricsv1beta1api.Convert_v1beta1_PodMetricsList_To_metrics_PodMetricsList(versionedMetrics, metrics, nil)
	if err != nil {
		return nil, err
	}
	return metrics, nil
}

func verifyEmptyMetrics(o TopPodOptions, labelSelector labels.Selector, fieldSelector fields.Selector) error {
	if len(o.ResourceName) > 0 {
		pod, err := o.PodClient.Pods(o.Namespace).Get(context.TODO(), o.ResourceName, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if err := checkPodAge(pod); err != nil {
			return err
		}
	} else {
		pods, err := o.PodClient.Pods(o.Namespace).List(context.TODO(), metav1.ListOptions{
			LabelSelector: labelSelector.String(),
			FieldSelector: fieldSelector.String(),
		})
		if err != nil {
			return err
		}
		if len(pods.Items) == 0 {
			return nil
		}
		for _, pod := range pods.Items {
			if err := checkPodAge(&pod); err != nil {
				return err
			}
		}
	}
	return errors.New("metrics not available yet")
}

func checkPodAge(pod *corev1.Pod) error {
	age := time.Since(pod.CreationTimestamp.Time)
	if age > metricsCreationDelay {
		message := fmt.Sprintf("Metrics not available for pod %s/%s, age: %s", pod.Namespace, pod.Name, age.String())
		return errors.New(message)
	} else {
		klog.V(2).Infof("Metrics not yet available for pod %s/%s, age: %s", pod.Namespace, pod.Name, age.String())
		return nil
	}
}
