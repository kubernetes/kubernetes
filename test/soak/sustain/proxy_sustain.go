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

package main

import (
	"encoding/json"
	"flag"
	"os"
	"path/filepath"
	"time"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client/clientcmd"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/fields"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/labels"
	"github.com/golang/glog"
)

var (
	requests = flag.Int("requests", 1000, "Number of requets to issue")
	gke      = flag.String("gke_context", "", "Target GKE cluster with context gke_{project}_{zone}_{cluster-name}")
)

const (
	esKey   = "k8s-app"
	esValue = "elasticsearch-logging"
)

func bodyToJSON(body []byte) map[string]interface{} {
	var r map[string]interface{}
	if err := json.Unmarshal(body, &r); err != nil {
		glog.Fatalf("Failed to unmarshal Elasticsearch response %s: %v", string(body), err)
	}
	return r
}

func main() {

	flag.Parse()

	glog.Infof("Sustained request test to Elasticsearch logging service via proxy with %d requests", *requests)

	var spec string
	if *gke != "" {
		spec = filepath.Join(os.Getenv("HOME"), ".config", "gcloud", "kubernetes", "kubeconfig")
	} else {
		spec = filepath.Join(os.Getenv("HOME"), ".kube", "config")
	}
	settings, err := clientcmd.LoadFromFile(spec)
	if err != nil {
		glog.Fatalf("Error loading configuration: %v", err.Error())
	}
	if *gke != "" {
		settings.CurrentContext = *gke
	}
	config, err := clientcmd.NewDefaultClientConfig(*settings, &clientcmd.ConfigOverrides{}).ClientConfig()
	if err != nil {
		glog.Fatalf("Failed to construct config: %v", err)
	}

	c, err := client.New(config)
	if err != nil {
		glog.Fatalf("Failed to make client: %v", err)
	}

	// graceTime is how long to keep retrying requests for status information.
	const graceTime = 2 * time.Minute
	// ingestionTimeout is how long to keep retrying to wait for all the
	// logs to be ingested.
	const ingestionTimeout = 3 * time.Minute

	// Check for the existence of the Elasticsearch service.
	glog.Info("Checking the Elasticsearch service exists.")
	s := c.Services(api.NamespaceSystem)
	// Make a few attempts to connect. This makes the test robust against
	// being run as the first e2e test just after the e2e cluster has been created.
	for start := time.Now(); time.Since(start) < graceTime; time.Sleep(5 * time.Second) {
		if _, err = s.Get("elasticsearch-logging"); err == nil {
			break
		}
		glog.Infof("Attempt to check for the existence of the Elasticsearch service failed after %v", time.Since(start))
	}
	if err != nil {
		glog.Fatalf("Failed to find Elasticsearch service: %v", err)
	}
	glog.Info("Found Elasticsearch service")

	// Obtain a list of the Elasticsearch pods.
	label := labels.SelectorFromSet(labels.Set(map[string]string{esKey: esValue}))
	pods, err := c.Pods(api.NamespaceSystem).List(label, fields.Everything())
	if err != nil {
		glog.Fatalf("Failed to look for Elasticsearch pods: %v", err)
	}
	if len(pods.Items) == 0 {
		glog.Fatal("Failed to find any Elasticsearch pods.")
	}
	glog.Infof("Found %d Elasticsearch pods:", len(pods.Items))
	for i, pod := range pods.Items {
		glog.Infof("%d: %s podIP=%s phase=%s condition=%s", i, pod.Name, pod.Status.PodIP, pod.Status.Phase, pod.Status.Conditions)
	}

	// Make lots of requests
	failures := 0
	for request := 0; request < *requests; request++ {
		if request%100 == 0 {
			glog.Infof("Sent %d requests so far...", request)
		}
		body, err := c.Get().
			Namespace(api.NamespaceSystem).
			Prefix("proxy").
			Resource("services").
			Name("elasticsearch-logging").
			DoRaw()
		if err != nil {
			glog.Warningf("At request %d proxy call to Elasticsearch failed: %v", request, err)
			failures++
			continue
		}
		response := bodyToJSON(body)
		statusIntf, ok := response["status"]
		if !ok {
			glog.Warningf("At request %d response has no status field: %v", request, response)
			failures++
			continue
		}
		statusCode, ok := statusIntf.(float64)
		if !ok {
			// Assume this is a string returning Failure. Retry.
			glog.Warningf("At request %d expected status to be a float64 but got %v of type %T", request, statusIntf, statusIntf)
			failures++
			continue
		}
		if int(statusCode) != 200 {
			glog.Warningf("At request %d Elasticsearch cluster has a bad status: %v", request, statusCode)
			failures++
			continue
		}
	}
	if failures > 0 {
		glog.Fatalf("Got %d failures", failures)
	}
	glog.Info("No failures detected")
}
