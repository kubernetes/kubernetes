/*
Copyright 2017 The Kubernetes Authors.

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
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/admission/v1alpha1"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

// Config contains the server (the webhook) cert and key.
type Config struct {
	CertFile string
	KeyFile  string
}

func (c *Config) addFlags() {
	flag.StringVar(&c.CertFile, "tls-cert-file", c.CertFile, ""+
		"File containing the default x509 Certificate for HTTPS. (CA cert, if any, concatenated "+
		"after server cert).")
	flag.StringVar(&c.KeyFile, "tls-private-key-file", c.KeyFile, ""+
		"File containing the default x509 private key matching --tls-cert-file.")
}

// only allow pods to pull images from specific registry.
func admit(data []byte) *v1alpha1.AdmissionReviewStatus {
	ar := v1alpha1.AdmissionReview{}
	if err := json.Unmarshal(data, &ar); err != nil {
		glog.Error(err)
		return nil
	}
	podResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	if ar.Spec.Resource != podResource {
		glog.Errorf("expect resource to be %s", podResource)
		return nil
	}

	raw := ar.Spec.Object.Raw
	pod := v1.Pod{}
	if err := json.Unmarshal(raw, &pod); err != nil {
		glog.Error(err)
		return nil
	}
	reviewStatus := v1alpha1.AdmissionReviewStatus{}
	reviewStatus.Allowed = true
	// Note: the apiserver encodes the api.Pod. Decoding it as a v1.Pod will
	// lose the metadata. So the following check on labels will not work
	// until we let the apiserver encodes the versioned object.
	for k, v := range pod.Labels {
		if k == "webhook-e2e-test" && v == "webhook-disallow" {
			reviewStatus.Allowed = false
			reviewStatus.Result = &metav1.Status{
				Reason: "the pod contains unwanted label",
			}
		}
	}
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, "webhook-disallow") {
			reviewStatus.Allowed = false
			reviewStatus.Result = &metav1.Status{
				Message: "the pod contains unwanted container name",
			}
		}
	}
	return &reviewStatus
}

func serve(w http.ResponseWriter, r *http.Request) {
	var body []byte
	if r.Body != nil {
		if data, err := ioutil.ReadAll(r.Body); err == nil {
			body = data
		}
	}

	// verify the content type is accurate
	contentType := r.Header.Get("Content-Type")
	if contentType != "application/json" {
		glog.Errorf("contentType=%s, expect application/json", contentType)
		return
	}

	reviewStatus := admit(body)
	ar := v1alpha1.AdmissionReview{
		Status: *reviewStatus,
	}

	resp, err := json.Marshal(ar)
	if err != nil {
		glog.Error(err)
	}
	if _, err := w.Write(resp); err != nil {
		glog.Error(err)
	}
}

func main() {
	var config Config
	config.addFlags()
	flag.Parse()

	http.HandleFunc("/", serve)
	clientset := getClient()
	server := &http.Server{
		Addr:      ":443",
		TLSConfig: configTLS(config, clientset),
	}
	server.ListenAndServeTLS("", "")
}
