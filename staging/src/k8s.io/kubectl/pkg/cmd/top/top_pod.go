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
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/client-go/restmapper"
	"k8s.io/klog/v2"
	"k8s.io/metrics/pkg/apis/custom_metrics/v1beta2"
	"strings"
	"time"

	corev1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/client-go/discovery"
	cacheddiscovery "k8s.io/client-go/discovery/cached"
	corev1client "k8s.io/client-go/kubernetes/typed/core/v1"
	cmdutil "k8s.io/kubectl/pkg/cmd/util"
	"k8s.io/kubectl/pkg/metricsutil"
	"k8s.io/kubectl/pkg/util/i18n"
	"k8s.io/kubectl/pkg/util/templates"
	metricsapi "k8s.io/metrics/pkg/apis/metrics"
	metricsv1beta1api "k8s.io/metrics/pkg/apis/metrics/v1beta1"
	metricsclientset "k8s.io/metrics/pkg/client/clientset/versioned"
	customclientset "k8s.io/metrics/pkg/client/custom_metrics"

	"github.com/spf13/cobra"
	"k8s.io/cli-runtime/pkg/genericclioptions"
)

type TopPodOptions struct {
	ResourceName    string
	Namespace           string
	Selector            string
	SortBy              string
	CustomMetrics             string
	AllNamespaces       bool
	PrintContainers     bool
	NoHeaders           bool
	PodClient           corev1client.PodsGetter
	Printer             *metricsutil.TopCmdPrinter
	DiscoveryClient     discovery.DiscoveryInterface
	MetricsClient       metricsclientset.Interface
	CustomMetricsClient customclientset.CustomMetricsClient

	genericclioptions.IOStreams
}

const metricsCreationDelay = 2 * time.Minute

var (
	topPodLong = templates.LongDesc(i18n.T(`
		Display Resource (CPU/Memory/CustomMetric) usage of pods.

		The 'top pod' command allows you to see the resource consumption of pods.

		Due to the metrics pipeline delay, they may be unavailable for a few minutes
		since pod creation.`))

	topPodExample = templates.Examples(i18n.T(`
		# Show CPU/Memory metrics for all pods in the default namespace
		kubectl top pod

        # Show CPU/Network metrics for all pods in default namespace. Requires "network_transmit_packets" metric available in Custom Metrics API
        kubectl top pod --metrics=network_transmit_packets

		# Show metrics for all pods in the given namespace
		kubectl top pod --namespace=NAMESPACE

		# Show metrics for a given pod and its containers
		kubectl top pod POD_NAME --containers

		# Show metrics for the pods defined by label name=myLabel
		kubectl top pod -l name=myLabel`))
)

func NewCmdTopPod(f cmdutil.Factory, o *TopPodOptions, streams genericclioptions.IOStreams) *cobra.Command {
	if o == nil {
		o = &TopPodOptions{
			IOStreams: streams,
		}
	}

	cmd := &cobra.Command{
		Use:                   "pod [NAME | -l label]",
		DisableFlagsInUseLine: true,
		Short:                 i18n.T("Display Resource (CPU/Memory/Storage) usage of pods"),
		Long:                  topPodLong,
		Example:               topPodExample,
		Run: func(cmd *cobra.Command, args []string) {
			cmdutil.CheckErr(o.Complete(f, cmd, args))
			cmdutil.CheckErr(o.Validate())
			cmdutil.CheckErr(o.RunTopPod())
		},
		Aliases: []string{"pods", "po"},
	}
	cmd.Flags().StringVarP(&o.Selector, "selector", "l", o.Selector, "Selector (label query) to filter on, supports '=', '==', and '!='.(e.g. -l key1=value1,key2=value2)")
	cmd.Flags().StringVar(&o.SortBy, "sort-by", o.Selector, "If non-empty, sort pods list using specified field. The field can be either 'cpu' or 'memory'.")
	cmd.Flags().BoolVar(&o.PrintContainers, "containers", o.PrintContainers, "If present, print usage of containers within a pod.")
	cmd.Flags().BoolVarP(&o.AllNamespaces, "all-namespaces", "A", o.AllNamespaces, "If present, list the requested object(s) across all namespaces. Namespace in current context is ignored even if specified with --namespace.")
	cmd.Flags().BoolVar(&o.NoHeaders, "no-headers", o.NoHeaders, "If present, print output without headers.")
	cmd.Flags().StringVar(&o.CustomMetrics, "custom-metrics", o.CustomMetrics, "Additional custom metrics that should be displayed. List of metrics available can be checked by running `kubectl top metrics`")
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
	o.MetricsClient, err = metricsclientset.NewForConfig(config)
	if err != nil {
		return err
	}

	apiVersionsGetter := customclientset.NewAvailableAPIsGetter(clientset.Discovery())
	cachedClient := cacheddiscovery.NewMemCacheClient(clientset.Discovery())
	restMapper := restmapper.NewDeferredDiscoveryRESTMapper(cachedClient)
	o.CustomMetricsClient = customclientset.NewForConfig(config, restMapper, apiVersionsGetter)


	o.PodClient = clientset.CoreV1()

	o.Printer = metricsutil.NewTopCmdPrinter(o.Out)
	return nil
}

func (o *TopPodOptions) Validate() error {
	if len(o.SortBy) > 0 {
		if o.SortBy != sortByCPU && o.SortBy != sortByMemory {
			return errors.New("--sort-by accepts only cpu or memory")
		}
	}
	if len(o.ResourceName) > 0 && len(o.Selector) > 0 {
		return errors.New("only one of NAME or --selector can be provided")
	}
	return nil
}

func (o TopPodOptions) RunTopPod() error {
	var err error
	selector := labels.Everything()
	if len(o.Selector) > 0 {
		selector, err = labels.Parse(o.Selector)
		if err != nil {
			return err
		}
	}

	apiGroups, err := o.DiscoveryClient.ServerGroups()
	if err != nil {
		return err
	}

	metricsAPIAvailable := SupportedAPIVersionAvailable(apiGroups, metricsapi.GroupName)

	if !metricsAPIAvailable {
		return errors.New("Metrics API not available")
	}
	metrics, err := o.getMetricsFromMetricsAPI(selector)
	if err != nil {
		return err
	}

	// TODO: Refactor this once Heapster becomes the API server.
	// First we check why no metrics have been received.
	if len(metrics.Items) == 0 {
		// If the API server query is successful but all the pods are newly created,
		// the metrics are probably not ready yet, so we return the error here in the first place.
		e := verifyEmptyMetrics(o, selector)
		if e != nil {
			return e
		}

		// if we had no errors, be sure we output something.
		if o.AllNamespaces {
			fmt.Fprintln(o.ErrOut, "No resources found")
		} else {
			fmt.Fprintf(o.ErrOut, "No resources found in %s namespace.\n", o.Namespace)
		}
	}
	if err != nil {
		return err
	}

	return o.Printer.PrintPodMetrics(metrics.Items, o.PrintContainers, o.AllNamespaces, o.NoHeaders, o.SortBy, o.CustomMetrics)
}

func (o TopPodOptions) getMetricsFromMetricsAPI(selector labels.Selector) (*metricsapi.PodMetricsList, error) {
	var err error
	ns := metav1.NamespaceAll
	if !o.AllNamespaces {
		ns = o.Namespace
	}
	metricSelector := labels.NewSelector()
	versionedMetrics := &metricsv1beta1api.PodMetricsList{}

	customMetrics := strings.Split(o.CustomMetrics, ",")

	if o.ResourceName != "" {
		m, err := o.MetricsClient.MetricsV1beta1().PodMetricses(ns).Get(context.TODO(), o.ResourceName, metav1.GetOptions{})
		if err != nil {
			return nil, err
		}
		versionedMetrics.Items = []metricsv1beta1api.PodMetrics{*m}
		for _, metric := range customMetrics {
			cm, err := o.CustomMetricsClient.NamespacedMetrics(ns).GetForObject(schema.GroupKind{Kind: "Pod"}, o.ResourceName, metric, metricSelector)
			if err == nil {
				addCustomMetric(versionedMetrics, []v1beta2.MetricValue{*cm})
			}
		}

	} else {
		versionedMetrics, err = o.MetricsClient.MetricsV1beta1().PodMetricses(ns).List(context.TODO(), metav1.ListOptions{LabelSelector: selector.String()})
		if err != nil {
			return nil, err
		}
		for _, metric := range customMetrics {
			cms, err := o.CustomMetricsClient.NamespacedMetrics(ns).GetForObjects(schema.GroupKind{Kind: "Pod"}, selector, metric, metricSelector)
			if err == nil {
				addCustomMetric(versionedMetrics, cms.Items)
			}
		}
	}
	metrics := &metricsapi.PodMetricsList{}
	err = metricsv1beta1api.Convert_v1beta1_PodMetricsList_To_metrics_PodMetricsList(versionedMetrics, metrics, nil)
	if err != nil {
		return nil, err
	}
	return metrics, nil
}

func addCustomMetric(ml *metricsv1beta1api.PodMetricsList, cms []v1beta2.MetricValue) {
	metrics := map[types.NamespacedName]v1beta2.MetricValue{}
	for _, cm := range cms {
		metrics[types.NamespacedName{
			Namespace: cm.DescribedObject.Namespace,
			Name:      cm.DescribedObject.Name,
		}] = cm
	}
	for _, m := range ml.Items {
		cm, found := metrics[types.NamespacedName{
			Namespace: m.Namespace,
			Name:      m.Name,
		}]
		if !found {
			continue
		}
		found = false
		for _, c := range m.Containers {
			if c.Name == "" {
				c.Usage[corev1.ResourceName(cm.Metric.Name)] = cm.Value
				found = true
			}
		}
		if !found {
			m.Containers = append(m.Containers, metricsv1beta1api.ContainerMetrics{
				Name:  "",
				Usage: corev1.ResourceList{
					corev1.ResourceName(cm.Metric.Name): cm.Value,
				},
			})
		}
	}
}

func verifyEmptyMetrics(o TopPodOptions, selector labels.Selector) error {
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
			LabelSelector: selector.String(),
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
		klog.Warningf(message)
		return errors.New(message)
	} else {
		klog.V(2).Infof("Metrics not yet available for pod %s/%s, age: %s", pod.Namespace, pod.Name, age.String())
		return nil
	}
}
