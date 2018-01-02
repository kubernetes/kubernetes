// Copyright 2015 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package integration

import (
	"encoding/json"
	"flag"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/golang/glog"
	"github.com/stretchr/testify/require"
	api_v1 "k8s.io/heapster/metrics/api/v1/types"
	metrics_api "k8s.io/heapster/metrics/apis/metrics/v1alpha1"
	"k8s.io/heapster/metrics/core"
	kube_api "k8s.io/kubernetes/pkg/api"
	apiErrors "k8s.io/kubernetes/pkg/api/errors"
	kube_api_unv "k8s.io/kubernetes/pkg/api/unversioned"
	kube_v1 "k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/labels"
	"k8s.io/kubernetes/pkg/util/sets"
)

const (
	targetTags       = "kubernetes-minion"
	heapsterBuildDir = "../deploy/docker"
)

var (
	testZone               = flag.String("test_zone", "us-central1-b", "GCE zone where the test will be executed")
	kubeVersions           = flag.String("kube_versions", "", "Comma separated list of kube versions to test against. By default will run the test against an existing cluster")
	heapsterControllerFile = flag.String("heapster_controller", "../deploy/kube-config/standalone-test/heapster-controller.yaml", "Path to heapster replication controller file.")
	heapsterServiceFile    = flag.String("heapster_service", "../deploy/kube-config/standalone-test/heapster-service.yaml", "Path to heapster service file.")
	heapsterImage          = flag.String("heapster_image", "heapster:e2e_test", "heapster docker image that needs to be tested.")
	avoidBuild             = flag.Bool("nobuild", false, "When true, a new heapster docker image will not be created and pushed to test cluster nodes.")
	namespace              = flag.String("namespace", "heapster-e2e-tests", "namespace to be used for testing, it will be deleted at the beginning of the test if exists")
	maxRetries             = flag.Int("retries", 20, "Number of attempts before failing this test.")
	runForever             = flag.Bool("run_forever", false, "If true, the tests are run in a loop forever.")
)

func deleteAll(fm kubeFramework, ns string, service *kube_api.Service, rc *kube_api.ReplicationController) error {
	glog.V(2).Infof("Deleting ns %s...", ns)
	err := fm.DeleteNs(ns)
	if err != nil {
		glog.V(2).Infof("Failed to delete %s", ns)
		return err
	}
	glog.V(2).Infof("Deleted ns %s.", ns)
	return nil
}

func createAll(fm kubeFramework, ns string, service **kube_api.Service, rc **kube_api.ReplicationController) error {
	glog.V(2).Infof("Creating ns %s...", ns)
	namespace := kube_api.Namespace{
		TypeMeta: kube_api_unv.TypeMeta{
			Kind:       "Namespace",
			APIVersion: "v1",
		},
		ObjectMeta: kube_api.ObjectMeta{
			Name: ns,
		},
	}
	if _, err := fm.CreateNs(&namespace); err != nil {
		glog.V(2).Infof("Failed to create ns: %v", err)
		return err
	}

	glog.V(2).Infof("Created ns %s.", ns)

	glog.V(2).Infof("Creating rc %s/%s...", ns, (*rc).Name)
	if newRc, err := fm.CreateRC(ns, *rc); err != nil {
		glog.V(2).Infof("Failed to create rc: %v", err)
		return err
	} else {
		*rc = newRc
	}
	glog.V(2).Infof("Created rc %s/%s.", ns, (*rc).Name)

	glog.V(2).Infof("Creating service %s/%s...", ns, (*service).Name)
	if newSvc, err := fm.CreateService(ns, *service); err != nil {
		glog.V(2).Infof("Failed to create service: %v", err)
		return err
	} else {
		*service = newSvc
	}
	glog.V(2).Infof("Created service %s/%s.", ns, (*service).Name)

	return nil
}

func removeHeapsterImage(fm kubeFramework, zone string) error {
	glog.V(2).Infof("Removing heapster image.")
	if err := removeDockerImage(*heapsterImage); err != nil {
		glog.Errorf("Failed to remove Heapster image: %v", err)
	} else {
		glog.V(2).Infof("Heapster image removed.")
	}
	if nodes, err := fm.GetNodeNames(); err == nil {
		for _, node := range nodes {
			host := strings.Split(node, ".")[0]
			cleanupRemoteHost(host, zone)
		}
	} else {
		glog.Errorf("failed to cleanup nodes - %v", err)
	}
	return nil
}

func buildAndPushHeapsterImage(hostnames []string, zone string) error {
	glog.V(2).Info("Building and pushing Heapster image...")
	curwd, err := os.Getwd()
	if err != nil {
		return err
	}
	if err := os.Chdir(heapsterBuildDir); err != nil {
		return err
	}
	if err := buildDockerImage(*heapsterImage); err != nil {
		return err
	}
	for _, host := range hostnames {
		if err := copyDockerImage(*heapsterImage, host, zone); err != nil {
			return err
		}
	}
	glog.V(2).Info("Heapster image pushed.")
	return os.Chdir(curwd)
}

func getHeapsterRcAndSvc(fm kubeFramework) (*kube_api.Service, *kube_api.ReplicationController, error) {
	// Add test docker image
	rc, err := fm.ParseRC(*heapsterControllerFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse heapster controller - %v", err)
	}
	for i := range rc.Spec.Template.Spec.Containers {
		rc.Spec.Template.Spec.Containers[i].Image = *heapsterImage
		rc.Spec.Template.Spec.Containers[i].ImagePullPolicy = kube_api.PullNever
		// increase logging level
		rc.Spec.Template.Spec.Containers[i].Env = append(rc.Spec.Template.Spec.Containers[0].Env, kube_api.EnvVar{Name: "FLAGS", Value: "--vmodule=*=3"})
	}

	svc, err := fm.ParseService(*heapsterServiceFile)
	if err != nil {
		return nil, nil, fmt.Errorf("failed to parse heapster service - %v", err)
	}

	return svc, rc, nil
}

func buildAndPushDockerImages(fm kubeFramework, zone string) error {
	if *avoidBuild {
		return nil
	}
	nodes, err := fm.GetNodeNames()
	if err != nil {
		return err
	}
	hostnames := []string{}
	for _, node := range nodes {
		hostnames = append(hostnames, strings.Split(node, ".")[0])
	}

	return buildAndPushHeapsterImage(hostnames, zone)
}

const (
	metricsEndpoint       = "/api/v1/metric-export"
	metricsSchemaEndpoint = "/api/v1/metric-export-schema"
)

func getTimeseries(fm kubeFramework, svc *kube_api.Service) ([]*api_v1.Timeseries, error) {
	body, err := fm.Client().Get().
		Namespace(svc.Namespace).
		Prefix("proxy").
		Resource("services").
		Name(svc.Name).
		Suffix(metricsEndpoint).
		Do().Raw()
	if err != nil {
		return nil, err
	}
	var timeseries []*api_v1.Timeseries
	if err := json.Unmarshal(body, &timeseries); err != nil {
		glog.V(2).Infof("Timeseries error: %v %v", err, string(body))
		return nil, err
	}
	return timeseries, nil
}

func getSchema(fm kubeFramework, svc *kube_api.Service) (*api_v1.TimeseriesSchema, error) {
	body, err := fm.Client().Get().
		Namespace(svc.Namespace).
		Prefix("proxy").
		Resource("services").
		Name(svc.Name).
		Suffix(metricsSchemaEndpoint).
		Do().Raw()
	if err != nil {
		return nil, err
	}
	var timeseriesSchema api_v1.TimeseriesSchema
	if err := json.Unmarshal(body, &timeseriesSchema); err != nil {
		glog.V(2).Infof("Metrics schema error: %v  %v", err, string(body))
		return nil, err
	}
	return &timeseriesSchema, nil
}

var expectedSystemContainers = map[string]struct{}{
	"machine":       {},
	"kubelet":       {},
	"kube-proxy":    {},
	"system":        {},
	"docker-daemon": {},
}

func isContainerBaseImageExpected(ts *api_v1.Timeseries) bool {
	_, exists := expectedSystemContainers[ts.Labels[core.LabelContainerName.Key]]
	return !exists
}

func runMetricExportTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetPodsRunningOnNodes()
	if err != nil {
		return err
	}
	expectedPods := make([]string, 0, len(podList))
	for _, pod := range podList {
		expectedPods = append(expectedPods, pod.Name)
	}
	glog.V(0).Infof("Expected pods: %v", expectedPods)

	expectedNodes, err := fm.GetNodeNames()
	if err != nil {
		return err
	}
	glog.V(0).Infof("Expected nodes: %v", expectedNodes)

	timeseries, err := getTimeseries(fm, svc)
	if err != nil {
		return err
	}
	if len(timeseries) == 0 {
		return fmt.Errorf("expected non zero timeseries")
	}
	schema, err := getSchema(fm, svc)
	if err != nil {
		return err
	}
	// Build a map of metric names to metric descriptors.
	mdMap := map[string]*api_v1.MetricDescriptor{}
	for idx := range schema.Metrics {
		mdMap[schema.Metrics[idx].Name] = &schema.Metrics[idx]
	}
	actualPods := map[string]bool{}
	actualNodes := map[string]bool{}
	actualSystemContainers := map[string]map[string]struct{}{}
	for _, ts := range timeseries {
		// Verify the relevant labels are present.
		// All common labels must be present.
		podName, podMetric := ts.Labels[core.LabelPodName.Key]

		for _, label := range core.CommonLabels() {
			_, exists := ts.Labels[label.Key]
			if !exists {
				return fmt.Errorf("timeseries: %v does not contain common label: %v", ts, label)
			}
		}
		if podMetric {
			for _, label := range core.PodLabels() {
				_, exists := ts.Labels[label.Key]
				if !exists {
					return fmt.Errorf("timeseries: %v does not contain pod label: %v", ts, label)
				}
			}
		}

		if podMetric {
			actualPods[podName] = true
			// Extra explicit check that the expecte metrics are there:
			requiredLabels := []string{
				core.LabelPodNamespaceUID.Key,
				core.LabelPodId.Key,
				core.LabelHostID.Key,
				// container name is checked later
			}
			for _, label := range requiredLabels {
				_, exists := ts.Labels[label]
				if !exists {
					return fmt.Errorf("timeseries: %v does not contain required label: %v", ts, label)
				}
			}

		} else {
			if cName, ok := ts.Labels[core.LabelContainerName.Key]; ok {
				hostname, ok := ts.Labels[core.LabelHostname.Key]
				if !ok {
					return fmt.Errorf("hostname label missing on container %+v", ts)
				}

				if cName == "machine" {
					actualNodes[hostname] = true
				} else {
					for _, label := range core.ContainerLabels() {
						if label == core.LabelContainerBaseImage && !isContainerBaseImageExpected(ts) {
							continue
						}
						_, exists := ts.Labels[label.Key]
						if !exists {
							return fmt.Errorf("timeseries: %v does not contain container label: %v", ts, label)
						}
					}
				}

				if _, exists := expectedSystemContainers[cName]; exists {
					if actualSystemContainers[cName] == nil {
						actualSystemContainers[cName] = map[string]struct{}{}
					}
					actualSystemContainers[cName][hostname] = struct{}{}
				}
			} else {
				return fmt.Errorf("container_name label missing on timeseries - %v", ts)
			}
		}

		// Explicitly check for resource id
		explicitRequirement := map[string][]string{
			core.MetricFilesystemUsage.MetricDescriptor.Name: {core.LabelResourceID.Key},
			core.MetricFilesystemLimit.MetricDescriptor.Name: {core.LabelResourceID.Key},
			core.MetricFilesystemAvailable.Name:              {core.LabelResourceID.Key}}

		for metricName, points := range ts.Metrics {
			md, exists := mdMap[metricName]
			if !exists {
				return fmt.Errorf("unexpected metric %q", metricName)
			}

			for _, point := range points {
				for _, label := range md.Labels {
					_, exists := point.Labels[label.Key]
					if !exists {
						return fmt.Errorf("metric %q point %v does not contain metric label: %v", metricName, point, label)
					}
				}
			}

			required := explicitRequirement[metricName]
			for _, label := range required {
				for _, point := range points {
					_, exists := point.Labels[label]
					if !exists {
						return fmt.Errorf("metric %q point %v does not contain metric label: %v", metricName, point, label)
					}
				}
			}
		}
	}
	// Validate that system containers are running on all the nodes.
	// This test could fail if one of the containers was down while the metrics sample was collected.
	for cName, hosts := range actualSystemContainers {
		for _, host := range expectedNodes {
			if _, ok := hosts[host]; !ok {
				return fmt.Errorf("System container %q not found on host: %q - %v", cName, host, actualSystemContainers)
			}
		}
	}

	if err := expectedItemsExist(expectedPods, actualPods); err != nil {
		return fmt.Errorf("expected pods don't exist %v.\nExpected: %v\nActual:%v", err, expectedPods, actualPods)
	}
	if err := expectedItemsExist(expectedNodes, actualNodes); err != nil {
		return fmt.Errorf("expected nodes don't exist %v.\nExpected: %v\nActual:%v", err, expectedNodes, actualNodes)
	}

	return nil
}

func expectedItemsExist(expectedItems []string, actualItems map[string]bool) error {
	for _, item := range expectedItems {
		if _, found := actualItems[item]; !found {
			return fmt.Errorf("missing %s", item)
		}
	}
	return nil
}

func getErrorCauses(err error) string {
	serr, ok := err.(*apiErrors.StatusError)
	if !ok {
		return ""
	}
	var causes []string
	for _, c := range serr.ErrStatus.Details.Causes {
		causes = append(causes, c.Message)
	}
	return strings.Join(causes, ", ")
}

var labelSelectorEverything = labels.Everything()

func getDataFromProxy(fm kubeFramework, svc *kube_api.Service, url string) ([]byte, error) {
	glog.V(2).Infof("Querying heapster: %s", url)
	return fm.Client().Get().
		Namespace(svc.Namespace).
		Prefix("proxy").
		Resource("services").
		Name(svc.Name).
		Suffix(url).
		Do().Raw()
}

func getDataFromProxyWithSelector(fm kubeFramework, svc *kube_api.Service, url string, labelSelector *labels.Selector) ([]byte, error) {
	glog.V(2).Infof("Querying heapster: %s", url)
	return fm.Client().Get().
		Namespace(svc.Namespace).
		Prefix("proxy").
		Resource("services").
		Name(svc.Name).
		Suffix(url).LabelsSelectorParam(*labelSelector).
		Do().Raw()
}

func getMetricResultList(fm kubeFramework, svc *kube_api.Service, url string) (*api_v1.MetricResultList, error) {
	body, err := getDataFromProxy(fm, svc, url)
	if err != nil {
		return nil, err
	}
	var data api_v1.MetricResultList
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return nil, err
	}
	if err := checkMetricResultListSanity(&data); err != nil {
		return nil, err
	}
	return &data, nil
}

func getMetricResult(fm kubeFramework, svc *kube_api.Service, url string) (*api_v1.MetricResult, error) {
	body, err := getDataFromProxy(fm, svc, url)
	if err != nil {
		return nil, err
	}
	var data api_v1.MetricResult
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return nil, err
	}
	if err := checkMetricResultSanity(&data); err != nil {
		return nil, err
	}
	return &data, nil
}

func getStringResult(fm kubeFramework, svc *kube_api.Service, url string) ([]string, error) {
	body, err := getDataFromProxy(fm, svc, url)
	if err != nil {
		return nil, err
	}
	var data []string
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return nil, err
	}
	if len(data) == 0 {
		return nil, fmt.Errorf("empty string array")
	}
	return data, nil
}

func checkMetricResultSanity(metrics *api_v1.MetricResult) error {
	bytes, err := json.Marshal(*metrics)
	if err != nil {
		return err
	}
	stringVersion := string(bytes)

	if len(metrics.Metrics) == 0 {
		return fmt.Errorf("empty metrics: %s", stringVersion)
	}
	// There should be recent metrics in the response.
	if time.Now().Sub(metrics.LatestTimestamp).Seconds() > 120 {
		return fmt.Errorf("corrupted last timestamp: %s", stringVersion)
	}
	// Metrics don't have to be sorted, so the oldest one can be first.
	if time.Now().Sub(metrics.Metrics[0].Timestamp).Hours() > 1 {
		return fmt.Errorf("corrupted timestamp: %s", stringVersion)
	}
	if metrics.Metrics[0].Value > 10000 {
		return fmt.Errorf("value too big: %s", stringVersion)
	}
	return nil
}

func checkMetricResultListSanity(metrics *api_v1.MetricResultList) error {
	if len(metrics.Items) == 0 {
		return fmt.Errorf("empty metrics")
	}
	for _, item := range metrics.Items {
		err := checkMetricResultSanity(&item)
		if err != nil {
			return err
		}
	}
	return nil
}

func runModelTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetPodsRunningOnNodes()
	if err != nil {
		return err
	}
	if len(podList) == 0 {
		return fmt.Errorf("empty pod list")
	}
	nodeList, err := fm.GetNodeNames()
	if err != nil {
		return err
	}
	if len(nodeList) == 0 {
		return fmt.Errorf("empty node list")
	}
	podNamesList := make([]string, 0, len(podList))
	for _, pod := range podList {
		podNamesList = append(podNamesList, fmt.Sprintf("%s/%s", pod.Namespace, pod.Name))
	}

	glog.V(0).Infof("Expected pods: %v", podNamesList)
	glog.V(0).Infof("Expected nodes: %v", nodeList)
	allkeys, err := getStringResult(fm, svc, "/api/v1/model/debug/allkeys")
	if err != nil {
		return fmt.Errorf("Failed to get debug information about keys: %v", err)
	}
	glog.V(0).Infof("Available Heapster metric sets: %v", allkeys)

	metricUrlsToCheck := []string{}
	batchMetricsUrlsToCheck := []string{}
	stringUrlsToCheck := []string{}

	/* TODO: enable once cluster aggregator is added.
	   metricUrlsToCheck = append(metricUrlsToCheck,
	   fmt.Sprintf("/api/v1/model/metrics/%s", "cpu-usage"),
	)
	*/

	/* TODO: add once Cluster metrics aggregator is added.
	   "/api/v1/model/metrics",
	   "/api/v1/model/"
	*/
	stringUrlsToCheck = append(stringUrlsToCheck)

	for _, node := range nodeList {
		metricUrlsToCheck = append(metricUrlsToCheck,
			fmt.Sprintf("/api/v1/model/nodes/%s/metrics/%s", node, "cpu/usage_rate"),
			fmt.Sprintf("/api/v1/model/nodes/%s/metrics/%s", node, "cpu-usage"),
		)

		stringUrlsToCheck = append(stringUrlsToCheck,
			fmt.Sprintf("/api/v1/model/nodes/%s/metrics", node),
		)
	}

	for _, pod := range podList {
		containerName := pod.Spec.Containers[0].Name

		metricUrlsToCheck = append(metricUrlsToCheck,
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/metrics/%s", pod.Namespace, pod.Name, "cpu/usage_rate"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/metrics/%s", pod.Namespace, pod.Name, "cpu-usage"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/containers/%s/metrics/%s", pod.Namespace, pod.Name, containerName, "cpu/usage_rate"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/containers/%s/metrics/%s", pod.Namespace, pod.Name, containerName, "cpu-usage"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/metrics/%s", pod.Namespace, "cpu/usage_rate"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/metrics/%s", pod.Namespace, "cpu-usage"),
		)

		batchMetricsUrlsToCheck = append(batchMetricsUrlsToCheck,
			fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s,%s/metrics/%s", pod.Namespace, pod.Name, pod.Name, "cpu/usage_rate"),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pod-list/%s,%s/metrics/%s", pod.Namespace, pod.Name, pod.Name, "cpu-usage"),
		)

		stringUrlsToCheck = append(stringUrlsToCheck,
			fmt.Sprintf("/api/v1/model/namespaces/%s/metrics", pod.Namespace),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/metrics", pod.Namespace, pod.Name),
			fmt.Sprintf("/api/v1/model/namespaces/%s/pods/%s/containers/%s/metrics", pod.Namespace, pod.Name, containerName),
		)
	}

	for _, url := range metricUrlsToCheck {
		_, err := getMetricResult(fm, svc, url)
		if err != nil {
			return fmt.Errorf("error while querying %s: %v", url, err)
		}
	}

	for _, url := range batchMetricsUrlsToCheck {
		_, err := getMetricResultList(fm, svc, url)
		if err != nil {
			return fmt.Errorf("error while querying %s: %v", url, err)
		}
	}

	for _, url := range stringUrlsToCheck {
		_, err := getStringResult(fm, svc, url)
		if err != nil {
			return fmt.Errorf("error while querying %s: %v", url, err)
		}
	}
	return nil
}

const (
	apiPrefix           = "apis"
	metricsApiGroupName = "metrics"
	metricsApiVersion   = "v1alpha1"
)

var baseMetricsUrl = fmt.Sprintf("%s/%s/%s", apiPrefix, metricsApiGroupName, metricsApiVersion)

func checkUsage(res kube_v1.ResourceList) error {
	if _, found := res[kube_v1.ResourceCPU]; !found {
		return fmt.Errorf("Cpu not found")
	}
	if _, found := res[kube_v1.ResourceMemory]; !found {
		return fmt.Errorf("Memory not found")
	}
	return nil
}

func getPodMetrics(fm kubeFramework, svc *kube_api.Service, pod kube_api.Pod) (*metrics_api.PodMetrics, error) {
	url := fmt.Sprintf("%s/namespaces/%s/pods/%s", baseMetricsUrl, pod.Namespace, pod.Name)
	body, err := getDataFromProxy(fm, svc, url)
	if err != nil {
		return nil, err
	}
	var data metrics_api.PodMetrics
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return nil, err
	}
	return &data, nil
}

func getAllPodsInNamespaceMetrics(fm kubeFramework, svc *kube_api.Service, namespace string) (metrics_api.PodMetricsList, error) {
	url := fmt.Sprintf("%s/namespaces/%s/pods/", baseMetricsUrl, namespace)
	return getPodMetricsList(fm, svc, url, &labelSelectorEverything)
}

func getAllPodsMetrics(fm kubeFramework, svc *kube_api.Service) (metrics_api.PodMetricsList, error) {
	url := fmt.Sprintf("%s/pods/", baseMetricsUrl)
	selector := labels.Everything()
	return getPodMetricsList(fm, svc, url, &selector)
}

func getLabelSelectedPodMetrics(fm kubeFramework, svc *kube_api.Service, namespace string, labelSelector *labels.Selector) (metrics_api.PodMetricsList, error) {
	url := fmt.Sprintf("%s/namespaces/%s/pods/", baseMetricsUrl, namespace)
	return getPodMetricsList(fm, svc, url, labelSelector)
}

func getPodMetricsList(fm kubeFramework, svc *kube_api.Service, url string, labelSelector *labels.Selector) (metrics_api.PodMetricsList, error) {
	body, err := getDataFromProxyWithSelector(fm, svc, url, labelSelector)
	if err != nil {
		return metrics_api.PodMetricsList{}, err
	}
	var data metrics_api.PodMetricsList
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return metrics_api.PodMetricsList{}, err
	}
	return data, nil
}

func checkSinglePodMetrics(metrics *metrics_api.PodMetrics, pod *kube_api.Pod) error {
	if metrics.Name != pod.Name {
		return fmt.Errorf("Wrong pod name: expected %v, got %v", pod.Name, metrics.Name)
	}
	if metrics.Namespace != pod.Namespace {
		return fmt.Errorf("Wrong pod namespace: expected %v, got %v", pod.Namespace, metrics.Namespace)
	}
	if len(pod.Spec.Containers) != len(metrics.Containers) {
		return fmt.Errorf("Wrong number of containers in returned metrics: expected %v, got %v", len(pod.Spec.Containers), len(metrics.Containers))
	}
	for _, c := range metrics.Containers {
		if err := checkUsage(c.Usage); err != nil {
			return err
		}
	}
	return nil
}

func getSingleNodeMetrics(fm kubeFramework, svc *kube_api.Service, node string) (*metrics_api.NodeMetrics, error) {
	url := fmt.Sprintf("%s/nodes/%s", baseMetricsUrl, node)
	body, err := getDataFromProxy(fm, svc, url)
	if err != nil {
		return nil, err
	}
	var data metrics_api.NodeMetrics
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return nil, err
	}
	return &data, nil
}

func getNodeMetricsList(fm kubeFramework, svc *kube_api.Service, url string, labelSelector *labels.Selector) (metrics_api.NodeMetricsList, error) {
	body, err := getDataFromProxyWithSelector(fm, svc, url, labelSelector)
	if err != nil {
		return metrics_api.NodeMetricsList{}, err
	}
	var data metrics_api.NodeMetricsList
	if err := json.Unmarshal(body, &data); err != nil {
		glog.V(2).Infof("response body: %v", string(body))
		return metrics_api.NodeMetricsList{}, err
	}
	return data, nil
}

func getLabelSelectedNodeMetrics(fm kubeFramework, svc *kube_api.Service, labelSelector *labels.Selector) (metrics_api.NodeMetricsList, error) {
	url := fmt.Sprintf("%s/nodes", baseMetricsUrl)
	return getNodeMetricsList(fm, svc, url, labelSelector)
}

func getAllNodeMetrics(fm kubeFramework, svc *kube_api.Service) (metrics_api.NodeMetricsList, error) {
	url := fmt.Sprintf("%s/nodes", baseMetricsUrl)
	selector := labels.Everything()
	return getNodeMetricsList(fm, svc, url, &selector)
}

func runSingleNodeMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	nodeList, err := fm.GetNodeNames()
	if err != nil {
		return err
	}
	if len(nodeList) == 0 {
		return fmt.Errorf("empty node list")
	}

	for _, node := range nodeList {
		metrics, err := getSingleNodeMetrics(fm, svc, node)
		if err != nil {
			return err
		}
		if metrics.Name != node {
			return fmt.Errorf("Wrong node name: expected %v, got %v", node, metrics.Name)
		}
		if err := checkUsage(metrics.Usage); err != nil {
			return err
		}
	}
	return nil
}

func runLabelSelectorNodeMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	nodeList, err := fm.GetNodes()
	if err != nil {
		return err
	}
	if len(nodeList.Items) == 0 {
		return fmt.Errorf("empty node list")
	}
	labelMap := make(map[string]map[string]kube_api.Node)
	for _, n := range nodeList.Items {
		for label, value := range n.Labels {
			selector := label + "=" + value
			if _, found := labelMap[selector]; !found {
				labelMap[selector] = make(map[string]kube_api.Node)
			}
			labelMap[selector][n.Name] = n
		}
	}

	for selector, nodesWithLabel := range labelMap {
		sel, err := labels.Parse(selector)
		if err != nil {
			return err
		}
		metrics, err := getLabelSelectedNodeMetrics(fm, svc, &sel)
		if err != nil {
			return err
		}
		if len(metrics.Items) != len(nodesWithLabel) {
			return fmt.Errorf("Wrong number of label selected node metrics: expected %v, got %v", len(nodesWithLabel), len(metrics.Items))
		}
		for _, nodeMetric := range metrics.Items {
			node := nodesWithLabel[nodeMetric.Name]
			if nodeMetric.Name != node.Name {
				return fmt.Errorf("Wrong node name: expected %v, got %v", node.Name, nodeMetric.Name)
			}
			if err := checkUsage(nodeMetric.Usage); err != nil {
				return err
			}
		}
	}
	return nil
}

func runAllNodesMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	nodeList, err := fm.GetNodeNames()
	if err != nil {
		return err
	}
	if len(nodeList) == 0 {
		return fmt.Errorf("empty node list")
	}

	nodeNames := sets.NewString(nodeList...)
	metrics, err := getAllNodeMetrics(fm, svc)
	if err != nil {
		return err
	}

	if len(metrics.Items) != len(nodeList) {
		return fmt.Errorf("Wrong number of all node metrics: expected %v, got %v", len(nodeList), len(metrics.Items))
	}
	for _, nodeMetrics := range metrics.Items {
		if !nodeNames.Has(nodeMetrics.Name) {
			return fmt.Errorf("Unexpected node name: %v, expected one of: %v", nodeMetrics.Name, nodeList)
		}
		if err := checkUsage(nodeMetrics.Usage); err != nil {
			return err
		}
	}
	return nil
}

func runSinglePodMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetAllRunningPods()
	if err != nil {
		return err
	}
	if len(podList) == 0 {
		return fmt.Errorf("empty pod list")
	}
	for _, pod := range podList {
		metrics, err := getPodMetrics(fm, svc, pod)
		if err != nil {
			return err
		}
		err = checkSinglePodMetrics(metrics, &pod)
		if err != nil {
			return err
		}
	}
	return nil
}

func runAllPodsInNamespaceMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetAllRunningPods()
	if err != nil {
		return err
	}
	if len(podList) == 0 {
		return fmt.Errorf("empty pod list")
	}
	nsToPods := make(map[string]map[string]kube_api.Pod)
	for _, pod := range podList {
		if _, found := nsToPods[pod.Namespace]; !found {
			nsToPods[pod.Namespace] = make(map[string]kube_api.Pod)
		}
		nsToPods[pod.Namespace][pod.Name] = pod
	}

	for ns, podMap := range nsToPods {
		metrics, err := getAllPodsInNamespaceMetrics(fm, svc, ns)
		if err != nil {
			return err
		}

		if len(metrics.Items) != len(nsToPods[ns]) {
			return fmt.Errorf("Wrong number of metrics of all pods in a namespace: expected %v, got %v", len(nsToPods[ns]), len(metrics.Items))
		}
		for _, podMetric := range metrics.Items {
			pod := podMap[podMetric.Name]
			err := checkSinglePodMetrics(&podMetric, &pod)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func runAllPodsMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetAllRunningPods()
	if err != nil {
		return err
	}
	if len(podList) == 0 {
		return fmt.Errorf("empty pod list")
	}
	pods := make(map[string]kube_api.Pod)
	for _, p := range podList {
		pods[p.Namespace+"/"+p.Name] = p
	}

	metrics, err := getAllPodsMetrics(fm, svc)
	if err != nil {
		return err
	}

	if len(metrics.Items) != len(podList) {
		return fmt.Errorf("Wrong number of all pod metrics: expected %v, got %v", len(podList), len(metrics.Items))
	}
	for _, podMetric := range metrics.Items {
		pod := pods[podMetric.Namespace+"/"+podMetric.Name]
		err := checkSinglePodMetrics(&podMetric, &pod)
		if err != nil {
			return err
		}
	}
	return nil
}

func runLabelSelectorPodMetricsApiTest(fm kubeFramework, svc *kube_api.Service) error {
	podList, err := fm.GetAllRunningPods()
	if err != nil {
		return err
	}
	if len(podList) == 0 {
		return fmt.Errorf("empty pod list")
	}
	nsToPods := make(map[string][]kube_api.Pod)
	for _, pod := range podList {
		nsToPods[pod.Namespace] = append(nsToPods[pod.Namespace], pod)
	}

	for ns, podsInNamespace := range nsToPods {
		labelMap := make(map[string]map[string]kube_api.Pod)
		for _, p := range podsInNamespace {
			for label, name := range p.Labels {
				selector := label + "=" + name
				if _, found := labelMap[selector]; !found {
					labelMap[selector] = make(map[string]kube_api.Pod)
				}
				labelMap[selector][p.Name] = p
			}
		}
		for selector, podsWithLabel := range labelMap {
			sel, err := labels.Parse(selector)
			if err != nil {
				return err
			}
			metrics, err := getLabelSelectedPodMetrics(fm, svc, ns, &sel)
			if err != nil {
				return err
			}
			if len(metrics.Items) != len(podsWithLabel) {
				return fmt.Errorf("Wrong number of label selected pod metrics: expected %v, got %v", len(podsWithLabel), len(metrics.Items))
			}
			for _, podMetric := range metrics.Items {
				pod := podsWithLabel[podMetric.Name]
				err := checkSinglePodMetrics(&podMetric, &pod)
				if err != nil {
					return err
				}
			}
		}
	}
	return nil
}

func apiTest(kubeVersion string, zone string) error {
	fm, err := newKubeFramework(kubeVersion)
	if err != nil {
		return err
	}
	if err := buildAndPushDockerImages(fm, zone); err != nil {
		return err
	}
	// Create heapster pod and service.
	svc, rc, err := getHeapsterRcAndSvc(fm)
	if err != nil {
		return err
	}
	ns := *namespace
	if err := deleteAll(fm, ns, svc, rc); err != nil {
		return err
	}
	if err := createAll(fm, ns, &svc, &rc); err != nil {
		return err
	}
	if err := fm.WaitUntilPodRunning(ns, rc.Spec.Template.Labels, time.Minute); err != nil {
		return err
	}
	if err := fm.WaitUntilServiceActive(svc, time.Minute); err != nil {
		return err
	}
	testFuncs := []func() error{
		func() error {
			glog.V(2).Infof("Heapster metric export test...")
			err := runMetricExportTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Heapster metric export test: OK")
			} else {
				glog.V(2).Infof("Heapster metric export test: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Model test")
			err := runModelTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Model test: OK")
			} else {
				glog.V(2).Infof("Model test: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - single pod")
			err := runSinglePodMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - single pod: OK")
			} else {
				glog.V(2).Infof("Metrics API test - single pod: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - All pods in a namespace")
			err := runAllPodsInNamespaceMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - All pods in a namespace: OK")
			} else {
				glog.V(2).Infof("Metrics API test - All pods in a namespace: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - all pods")
			err := runAllPodsMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - all pods: OK")
			} else {
				glog.V(2).Infof("Metrics API test - all pods: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - label selector for pods")
			err := runLabelSelectorPodMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - label selector for pods: OK")
			} else {
				glog.V(2).Infof("Metrics API test - label selector for pods: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - single node")
			err := runSingleNodeMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - single node: OK")
			} else {
				glog.V(2).Infof("Metrics API test - single node: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - label selector for nodes")
			err := runLabelSelectorNodeMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - label selector for nodes: OK")
			} else {
				glog.V(2).Infof("Metrics API test - label selector for nodes: error: %v", err)
			}
			return err
		},
		func() error {
			glog.V(2).Infof("Metrics API test - all nodes")
			err := runAllNodesMetricsApiTest(fm, svc)
			if err == nil {
				glog.V(2).Infof("Metrics API test - all nodes: OK")
			} else {
				glog.V(2).Infof("Metrics API test - all nodes: error: %v", err)
			}
			return err
		},
	}
	attempts := *maxRetries
	glog.Infof("Starting tests")
	for {
		var err error
		for _, testFunc := range testFuncs {
			if err = testFunc(); err != nil {
				break
			}
		}
		if *runForever {
			continue
		}
		if err == nil {
			glog.V(2).Infof("All tests passed.")
			break
		}
		if attempts == 0 {
			glog.V(2).Info("Too many attempts.")
			return err
		}
		glog.V(2).Infof("Some tests failed. Retrying.")
		attempts--
		time.Sleep(time.Second * 10)
	}
	deleteAll(fm, ns, svc, rc)
	removeHeapsterImage(fm, zone)
	return nil
}

func runApiTest() error {
	tempDir, err := ioutil.TempDir("", "deploy")
	if err != nil {
		return nil
	}
	defer os.RemoveAll(tempDir)
	if *kubeVersions == "" {
		return apiTest("", *testZone)
	}
	kubeVersionsList := strings.Split(*kubeVersions, ",")
	for _, kubeVersion := range kubeVersionsList {
		if err := apiTest(kubeVersion, *testZone); err != nil {
			return err
		}
	}
	return nil
}

func TestHeapster(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping heapster kubernetes integration test.")
	}
	require.NoError(t, runApiTest())
}
