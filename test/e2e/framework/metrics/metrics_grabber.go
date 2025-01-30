/*
Copyright 2015 The Kubernetes Authors.

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

package metrics

import (
	"context"
	"errors"
	"fmt"
	"net"
	"regexp"
	"strconv"
	"sync"
	"time"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epodoutput "k8s.io/kubernetes/test/e2e/framework/pod/output"
)

const (
	// kubeSchedulerPort is the default port for the scheduler status server.
	kubeSchedulerPort = 10259
	// kubeControllerManagerPort is the default port for the controller manager status server.
	kubeControllerManagerPort = 10257
	// snapshotControllerPort is the port for the snapshot controller
	snapshotControllerPort = 9102
	// kubeProxyPort is the default port for the kube-proxy status server.
	kubeProxyPort = 10249
)

// MetricsGrabbingDisabledError is an error that is wrapped by the
// different MetricsGrabber.Wrap functions when metrics grabbing is
// not supported. Tests that check metrics data should then skip
// the check.
var MetricsGrabbingDisabledError = errors.New("metrics grabbing disabled")

// Collection is metrics collection of components
type Collection struct {
	APIServerMetrics          APIServerMetrics
	APIServerMetricsSLIs      APIServerMetrics
	ControllerManagerMetrics  ControllerManagerMetrics
	SnapshotControllerMetrics SnapshotControllerMetrics
	KubeletMetrics            map[string]KubeletMetrics
	SchedulerMetrics          SchedulerMetrics
	ClusterAutoscalerMetrics  ClusterAutoscalerMetrics
}

// Grabber provides functions which grab metrics from components
type Grabber struct {
	client                             clientset.Interface
	externalClient                     clientset.Interface
	config                             *rest.Config
	grabFromAPIServer                  bool
	grabFromControllerManager          bool
	grabFromKubelets                   bool
	grabFromScheduler                  bool
	grabFromClusterAutoscaler          bool
	grabFromSnapshotController         bool
	kubeScheduler                      string
	waitForSchedulerReadyOnce          sync.Once
	kubeControllerManager              string
	waitForControllerManagerReadyOnce  sync.Once
	snapshotController                 string
	waitForSnapshotControllerReadyOnce sync.Once
}

// NewMetricsGrabber prepares for grabbing metrics data from several different
// components. It should be called when those components are running because
// it needs to communicate with them to determine for which components
// metrics data can be retrieved.
//
// Collecting metrics data is an optional debug feature. Not all clusters will
// support it. If disabled for a component, the corresponding Grab function
// will immediately return an error derived from MetricsGrabbingDisabledError.
func NewMetricsGrabber(ctx context.Context, c clientset.Interface, ec clientset.Interface, config *rest.Config, kubelets bool, scheduler bool, controllers bool, apiServer bool, clusterAutoscaler bool, snapshotController bool) (*Grabber, error) {

	kubeScheduler := ""
	kubeControllerManager := ""
	snapshotControllerManager := ""

	regKubeScheduler := regexp.MustCompile("kube-scheduler-.*")
	regKubeControllerManager := regexp.MustCompile("kube-controller-manager-.*")
	regSnapshotController := regexp.MustCompile("volume-snapshot-controller.*")

	if (scheduler || controllers) && config == nil {
		return nil, errors.New("a rest config is required for grabbing kube-controller and kube-controller-manager metrics")
	}

	podList, err := c.CoreV1().Pods(metav1.NamespaceSystem).List(ctx, metav1.ListOptions{})
	if err != nil {
		return nil, err
	}
	if len(podList.Items) < 1 {
		klog.Warningf("Can't find any pods in namespace %s to grab metrics from", metav1.NamespaceSystem)
	}
	for _, pod := range podList.Items {
		if regKubeScheduler.MatchString(pod.Name) {
			kubeScheduler = pod.Name
		}
		if regKubeControllerManager.MatchString(pod.Name) {
			kubeControllerManager = pod.Name
		}
		if regSnapshotController.MatchString(pod.Name) {
			snapshotControllerManager = pod.Name
		}
		if kubeScheduler != "" && kubeControllerManager != "" && snapshotControllerManager != "" {
			break
		}
	}
	if clusterAutoscaler && ec == nil {
		klog.Warningf("Did not receive an external client interface. Grabbing metrics from ClusterAutoscaler is disabled.")
	}

	return &Grabber{
		client:                     c,
		externalClient:             ec,
		config:                     config,
		grabFromAPIServer:          apiServer,
		grabFromControllerManager:  checkPodDebugHandlers(ctx, c, controllers, "kube-controller-manager", kubeControllerManager),
		grabFromKubelets:           kubelets,
		grabFromScheduler:          checkPodDebugHandlers(ctx, c, scheduler, "kube-scheduler", kubeScheduler),
		grabFromClusterAutoscaler:  clusterAutoscaler,
		grabFromSnapshotController: checkPodDebugHandlers(ctx, c, snapshotController, "snapshot-controller", snapshotControllerManager),
		kubeScheduler:              kubeScheduler,
		kubeControllerManager:      kubeControllerManager,
		snapshotController:         snapshotControllerManager,
	}, nil
}

func checkPodDebugHandlers(ctx context.Context, c clientset.Interface, requested bool, component, podName string) bool {
	if !requested {
		return false
	}
	if podName == "" {
		klog.Warningf("Can't find %s pod. Grabbing metrics from %s is disabled.", component, component)
		return false
	}

	// The debug handlers on the host where the pod runs might be disabled.
	// We can check that indirectly by trying to retrieve log output.
	limit := int64(1)
	if _, err := c.CoreV1().Pods(metav1.NamespaceSystem).GetLogs(podName, &v1.PodLogOptions{LimitBytes: &limit}).DoRaw(ctx); err != nil {
		klog.Warningf("Can't retrieve log output of %s (%q). Debug handlers might be disabled in kubelet. Grabbing metrics from %s is disabled.",
			podName, err, component)
		return false
	}

	// Metrics gathering enabled.
	return true
}

// HasControlPlanePods returns true if metrics grabber was able to find control-plane pods
func (g *Grabber) HasControlPlanePods() bool {
	return g.kubeScheduler != "" && g.kubeControllerManager != ""
}

// GrabFromKubelet returns metrics from kubelet
func (g *Grabber) GrabFromKubelet(ctx context.Context, nodeName string) (KubeletMetrics, error) {
	nodes, err := g.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{"metadata.name": nodeName}.AsSelector().String()})
	if err != nil {
		return KubeletMetrics{}, err
	}
	if len(nodes.Items) != 1 {
		return KubeletMetrics{}, fmt.Errorf("Error listing nodes with name %v, got %v", nodeName, nodes.Items)
	}
	kubeletPort := nodes.Items[0].Status.DaemonEndpoints.KubeletEndpoint.Port
	return g.grabFromKubeletInternal(ctx, nodeName, int(kubeletPort), "metrics")
}

// GrabresourceMetricsFromKubelet returns resource metrics from kubelet
func (g *Grabber) GrabResourceMetricsFromKubelet(ctx context.Context, nodeName string) (KubeletMetrics, error) {
	nodes, err := g.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{"metadata.name": nodeName}.AsSelector().String()})
	if err != nil {
		return KubeletMetrics{}, err
	}
	if len(nodes.Items) != 1 {
		return KubeletMetrics{}, fmt.Errorf("Error listing nodes with name %v, got %v", nodeName, nodes.Items)
	}
	kubeletPort := nodes.Items[0].Status.DaemonEndpoints.KubeletEndpoint.Port
	return g.grabFromKubeletInternal(ctx, nodeName, int(kubeletPort), "metrics/resource")
}

func (g *Grabber) grabFromKubeletInternal(ctx context.Context, nodeName string, kubeletPort int, pathSuffix string) (KubeletMetrics, error) {
	if kubeletPort <= 0 || kubeletPort > 65535 {
		return KubeletMetrics{}, fmt.Errorf("Invalid Kubelet port %v. Skipping Kubelet's metrics gathering", kubeletPort)
	}
	output, err := g.getMetricsFromNode(ctx, nodeName, int(kubeletPort), pathSuffix)
	if err != nil {
		return KubeletMetrics{}, err
	}
	return parseKubeletMetrics(output)
}

func (g *Grabber) getMetricsFromNode(ctx context.Context, nodeName string, kubeletPort int, pathSuffix string) (string, error) {
	// There's a problem with timing out during proxy. Wrapping this in a goroutine to prevent deadlock.
	finished := make(chan struct{}, 1)
	var err error
	var rawOutput []byte
	go func() {
		rawOutput, err = g.client.CoreV1().RESTClient().Get().
			Resource("nodes").
			SubResource("proxy").
			Name(fmt.Sprintf("%v:%v", nodeName, kubeletPort)).
			Suffix(pathSuffix).
			Do(ctx).Raw()
		finished <- struct{}{}
	}()
	select {
	case <-time.After(proxyTimeout):
		return "", fmt.Errorf("Timed out when waiting for proxy to gather metrics from %v", nodeName)
	case <-finished:
		if err != nil {
			return "", err
		}
		return string(rawOutput), nil
	}
}

// GrabFromKubeProxy returns metrics from kube-proxy
func (g *Grabber) GrabFromKubeProxy(ctx context.Context, nodeName string) (KubeProxyMetrics, error) {
	nodes, err := g.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{FieldSelector: fields.Set{"metadata.name": nodeName}.AsSelector().String()})
	if err != nil {
		return KubeProxyMetrics{}, err
	}

	if len(nodes.Items) != 1 {
		return KubeProxyMetrics{}, fmt.Errorf("error listing nodes with name %v, got %v", nodeName, nodes.Items)
	}
	output, err := g.grabFromKubeProxy(ctx, nodeName)
	if err != nil {
		return KubeProxyMetrics{}, err
	}
	return parseKubeProxyMetrics(output)
}

func (g *Grabber) grabFromKubeProxy(ctx context.Context, nodeName string) (string, error) {
	hostCmdPodName := fmt.Sprintf("grab-kube-proxy-metrics-%s", framework.RandomSuffix())
	hostCmdPod := e2epod.NewExecPodSpec(metav1.NamespaceSystem, hostCmdPodName, true)
	nodeSelection := e2epod.NodeSelection{Name: nodeName}
	e2epod.SetNodeSelection(&hostCmdPod.Spec, nodeSelection)
	if _, err := g.client.CoreV1().Pods(metav1.NamespaceSystem).Create(ctx, hostCmdPod, metav1.CreateOptions{}); err != nil {
		return "", fmt.Errorf("failed to create pod to fetch metrics: %w", err)
	}
	if err := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, g.client, hostCmdPodName, metav1.NamespaceSystem, 5*time.Minute); err != nil {
		return "", fmt.Errorf("error waiting for pod to be up: %w", err)
	}

	host := "127.0.0.1"
	if framework.TestContext.ClusterIsIPv6() {
		host = "::1"
	}

	stdout, err := e2epodoutput.RunHostCmd(metav1.NamespaceSystem, hostCmdPodName, fmt.Sprintf("curl --silent %s/metrics", net.JoinHostPort(host, strconv.Itoa(kubeProxyPort))))
	_ = g.client.CoreV1().Pods(metav1.NamespaceSystem).Delete(ctx, hostCmdPodName, metav1.DeleteOptions{})
	return stdout, err
}

// GrabFromScheduler returns metrics from scheduler
func (g *Grabber) GrabFromScheduler(ctx context.Context) (SchedulerMetrics, error) {
	if !g.grabFromScheduler {
		return SchedulerMetrics{}, fmt.Errorf("kube-scheduler: %w", MetricsGrabbingDisabledError)
	}

	var err error

	g.waitForSchedulerReadyOnce.Do(func() {
		if readyErr := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, g.client, g.kubeScheduler, metav1.NamespaceSystem, 5*time.Minute); readyErr != nil {
			err = fmt.Errorf("error waiting for kube-scheduler pod to be ready: %w", readyErr)
		}
	})
	if err != nil {
		return SchedulerMetrics{}, err
	}

	var lastMetricsFetchErr error
	var output string
	if metricsWaitErr := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, true, func(ctx context.Context) (bool, error) {
		output, lastMetricsFetchErr = g.getSecureMetricsFromPod(ctx, g.kubeScheduler, metav1.NamespaceSystem, kubeSchedulerPort)
		return lastMetricsFetchErr == nil, nil
	}); metricsWaitErr != nil {
		err := fmt.Errorf("error waiting for kube-scheduler pod to expose metrics: %v; %v", metricsWaitErr, lastMetricsFetchErr)
		return SchedulerMetrics{}, err
	}

	return parseSchedulerMetrics(output)
}

// GrabFromClusterAutoscaler returns metrics from cluster autoscaler
func (g *Grabber) GrabFromClusterAutoscaler(ctx context.Context) (ClusterAutoscalerMetrics, error) {
	if !g.HasControlPlanePods() && g.externalClient == nil {
		return ClusterAutoscalerMetrics{}, fmt.Errorf("ClusterAutoscaler: %w", MetricsGrabbingDisabledError)
	}
	var client clientset.Interface
	var namespace string
	if g.externalClient != nil {
		client = g.externalClient
		namespace = "kubemark"
	} else {
		client = g.client
		namespace = metav1.NamespaceSystem
	}
	output, err := g.getMetricsFromPod(ctx, client, "cluster-autoscaler", namespace, 8085)
	if err != nil {
		return ClusterAutoscalerMetrics{}, err
	}
	return parseClusterAutoscalerMetrics(output)
}

// GrabFromControllerManager returns metrics from controller manager
func (g *Grabber) GrabFromControllerManager(ctx context.Context) (ControllerManagerMetrics, error) {
	if !g.grabFromControllerManager {
		return ControllerManagerMetrics{}, fmt.Errorf("kube-controller-manager: %w", MetricsGrabbingDisabledError)
	}

	var err error

	g.waitForControllerManagerReadyOnce.Do(func() {
		if readyErr := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, g.client, g.kubeControllerManager, metav1.NamespaceSystem, 5*time.Minute); readyErr != nil {
			err = fmt.Errorf("error waiting for kube-controller-manager pod to be ready: %w", readyErr)
		}
	})
	if err != nil {
		return ControllerManagerMetrics{}, err
	}

	var output string
	var lastMetricsFetchErr error
	if metricsWaitErr := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, true, func(ctx context.Context) (bool, error) {
		output, lastMetricsFetchErr = g.getSecureMetricsFromPod(ctx, g.kubeControllerManager, metav1.NamespaceSystem, kubeControllerManagerPort)
		return lastMetricsFetchErr == nil, nil
	}); metricsWaitErr != nil {
		err := fmt.Errorf("error waiting for kube-controller-manager to expose metrics: %v; %v", metricsWaitErr, lastMetricsFetchErr)
		return ControllerManagerMetrics{}, err
	}

	return parseControllerManagerMetrics(output)
}

// GrabFromSnapshotController returns metrics from controller manager
func (g *Grabber) GrabFromSnapshotController(ctx context.Context, podName string, port int) (SnapshotControllerMetrics, error) {
	if !g.grabFromSnapshotController {
		return SnapshotControllerMetrics{}, fmt.Errorf("volume-snapshot-controller: %w", MetricsGrabbingDisabledError)
	}

	// Use overrides if provided via test config flags.
	// Otherwise, use the default volume-snapshot-controller pod name and port.
	if podName == "" {
		podName = g.snapshotController
	}
	if port == 0 {
		port = snapshotControllerPort
	}

	var err error

	g.waitForSnapshotControllerReadyOnce.Do(func() {
		if readyErr := e2epod.WaitTimeoutForPodReadyInNamespace(ctx, g.client, podName, metav1.NamespaceSystem, 5*time.Minute); readyErr != nil {
			err = fmt.Errorf("error waiting for volume-snapshot-controller pod to be ready: %w", readyErr)
		}
	})
	if err != nil {
		return SnapshotControllerMetrics{}, err
	}

	var output string
	var lastMetricsFetchErr error
	if metricsWaitErr := wait.PollUntilContextTimeout(ctx, time.Second, time.Minute, true, func(ctx context.Context) (bool, error) {
		output, lastMetricsFetchErr = g.getMetricsFromPod(ctx, g.client, podName, metav1.NamespaceSystem, port)
		return lastMetricsFetchErr == nil, nil
	}); metricsWaitErr != nil {
		err = fmt.Errorf("error waiting for volume-snapshot-controller pod to expose metrics: %v; %v", metricsWaitErr, lastMetricsFetchErr)
		return SnapshotControllerMetrics{}, err
	}

	return parseSnapshotControllerMetrics(output)
}

// GrabFromAPIServer returns metrics from API server
func (g *Grabber) GrabFromAPIServer(ctx context.Context) (APIServerMetrics, error) {
	output, err := g.getMetricsFromAPIServer(ctx)
	if err != nil {
		return APIServerMetrics{}, err
	}
	return parseAPIServerMetrics(output)
}

// GrabMetricsSLIsFromAPIServer returns metrics from API server
func (g *Grabber) GrabMetricsSLIsFromAPIServer(ctx context.Context) (APIServerMetrics, error) {
	output, err := g.getMetricsSLIsFromAPIServer(ctx)
	if err != nil {
		return APIServerMetrics{}, err
	}
	return parseAPIServerMetrics(output)
}

func (g *Grabber) getMetricsFromAPIServer(ctx context.Context) (string, error) {
	rawOutput, err := g.client.CoreV1().RESTClient().Get().RequestURI("/metrics").Do(ctx).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}

func (g *Grabber) getMetricsSLIsFromAPIServer(ctx context.Context) (string, error) {
	rawOutput, err := g.client.CoreV1().RESTClient().Get().RequestURI("/metrics/slis").Do(ctx).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}

// Grab returns metrics from corresponding component
func (g *Grabber) Grab(ctx context.Context) (Collection, error) {
	result := Collection{}
	var errs []error
	if g.grabFromAPIServer {
		metrics, err := g.GrabFromAPIServer(ctx)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.APIServerMetrics = metrics
		}
		metrics, err = g.GrabMetricsSLIsFromAPIServer(ctx)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.APIServerMetricsSLIs = metrics
		}
	}
	if g.grabFromScheduler {
		metrics, err := g.GrabFromScheduler(ctx)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.SchedulerMetrics = metrics
		}
	}
	if g.grabFromControllerManager {
		metrics, err := g.GrabFromControllerManager(ctx)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.ControllerManagerMetrics = metrics
		}
	}
	if g.grabFromSnapshotController {
		metrics, err := g.GrabFromSnapshotController(ctx, g.snapshotController, snapshotControllerPort)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.SnapshotControllerMetrics = metrics
		}
	}
	if g.grabFromClusterAutoscaler {
		metrics, err := g.GrabFromClusterAutoscaler(ctx)
		if err != nil {
			errs = append(errs, err)
		} else {
			result.ClusterAutoscalerMetrics = metrics
		}
	}
	if g.grabFromKubelets {
		result.KubeletMetrics = make(map[string]KubeletMetrics)
		nodes, err := g.client.CoreV1().Nodes().List(ctx, metav1.ListOptions{})
		if err != nil {
			errs = append(errs, err)
		} else {
			for _, node := range nodes.Items {
				kubeletPort := node.Status.DaemonEndpoints.KubeletEndpoint.Port
				metrics, err := g.grabFromKubeletInternal(ctx, node.Name, int(kubeletPort), "metrics")
				if err != nil {
					errs = append(errs, err)
				}
				result.KubeletMetrics[node.Name] = metrics
			}
		}
	}
	if len(errs) > 0 {
		return result, fmt.Errorf("Errors while grabbing metrics: %v", errs)
	}
	return result, nil
}

// getMetricsFromPod retrieves metrics data from an insecure port.
func (g *Grabber) getMetricsFromPod(ctx context.Context, client clientset.Interface, podName string, namespace string, port int) (string, error) {
	rawOutput, err := client.CoreV1().RESTClient().Get().
		Namespace(namespace).
		Resource("pods").
		SubResource("proxy").
		Name(fmt.Sprintf("%s:%d", podName, port)).
		Suffix("metrics").
		Do(ctx).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}

// getSecureMetricsFromPod retrieves metrics from a pod that uses TLS
// and checks client credentials. Conceptually this function is
// similar to "kubectl port-forward" + "kubectl get --raw
// https://localhost:<port>/metrics". It uses the same credentials
// as kubelet.
func (g *Grabber) getSecureMetricsFromPod(ctx context.Context, podName string, namespace string, port int) (string, error) {
	dialer := e2epod.NewDialer(g.client, g.config)
	metricConfig := rest.CopyConfig(g.config)
	addr := e2epod.Addr{
		Namespace: namespace,
		PodName:   podName,
		Port:      port,
	}
	metricConfig.Dial = func(ctx context.Context, network, address string) (net.Conn, error) {
		return dialer.DialContainerPort(ctx, addr)
	}
	// This should make it possible verify the server, but while it
	// got past the server name check, certificate validation
	// still failed.
	metricConfig.Host = addr.String()
	metricConfig.ServerName = "localhost"
	// Verifying the pod certificate with the same root CA
	// as for the API server led to an error about "unknown root
	// certificate". Disabling certificate checking on the client
	// side gets around that and should be good enough for
	// E2E testing.
	metricConfig.Insecure = true
	metricConfig.CAFile = ""
	metricConfig.CAData = nil

	// clientset.NewForConfig is used because
	// metricClient.RESTClient() is directly usable, in contrast
	// to the client constructed by rest.RESTClientFor().
	metricClient, err := clientset.NewForConfig(metricConfig)
	if err != nil {
		return "", err
	}

	rawOutput, err := metricClient.RESTClient().Get().
		AbsPath("metrics").
		Do(ctx).Raw()
	if err != nil {
		return "", err
	}
	return string(rawOutput), nil
}
