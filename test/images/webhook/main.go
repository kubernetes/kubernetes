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
	"fmt"
	"io/ioutil"
	"net/http"
	"strings"

	"github.com/golang/glog"
	"k8s.io/api/admission/v1alpha1"
	corev1 "k8s.io/api/core/v1"
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

func toAdmissionReviewStatus(err error) *v1alpha1.AdmissionReviewStatus {
	return &v1alpha1.AdmissionReviewStatus{
		Result: &metav1.Status{
			Message: err.Error(),
		},
	}
}

// only allow pods to pull images from specific registry.
func admitPods(ar v1alpha1.AdmissionReview) *v1alpha1.AdmissionReviewStatus {
	glog.V(2).Info("admitting pods")
	podResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	if ar.Spec.Resource != podResource {
		err := fmt.Errorf("expect resource to be %s", podResource)
		glog.Error(err)
		return toAdmissionReviewStatus(err)
	}

	raw := ar.Spec.Object.Raw
	pod := corev1.Pod{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(raw, nil, &pod); err != nil {
		glog.Error(err)
		return toAdmissionReviewStatus(err)
	}
	reviewStatus := v1alpha1.AdmissionReviewStatus{}
	reviewStatus.Allowed = true

	var msg string
	for k, v := range pod.Labels {
		if k == "webhook-e2e-test" && v == "webhook-disallow" {
			reviewStatus.Allowed = false
			msg = msg + "the pod contains unwanted label; "
		}
	}
	for _, container := range pod.Spec.Containers {
		if strings.Contains(container.Name, "webhook-disallow") {
			reviewStatus.Allowed = false
			msg = msg + "the pod contains unwanted container name; "
		}
	}
	if !reviewStatus.Allowed {
		reviewStatus.Result = &metav1.Status{Message: strings.TrimSpace(msg)}
	}
	return &reviewStatus
}

// deny configmaps with specific key-value pair.
func admitConfigMaps(ar v1alpha1.AdmissionReview) *v1alpha1.AdmissionReviewStatus {
	glog.V(2).Info("admitting configmaps")
	configMapResource := metav1.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}
	if ar.Spec.Resource != configMapResource {
		glog.Errorf("expect resource to be %s", configMapResource)
		return nil
	}

	raw := ar.Spec.Object.Raw
	configmap := corev1.ConfigMap{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(raw, nil, &configmap); err != nil {
		glog.Error(err)
		return toAdmissionReviewStatus(err)
	}
	reviewStatus := v1alpha1.AdmissionReviewStatus{}
	reviewStatus.Allowed = true
	for k, v := range configmap.Data {
		if k == "webhook-e2e-test" && v == "webhook-disallow" {
			reviewStatus.Allowed = false
			reviewStatus.Result = &metav1.Status{
				Reason: "the configmap contains unwanted key and value",
			}
		}
	}
	return &reviewStatus
}

func admitCRD(ar v1alpha1.AdmissionReview) *v1alpha1.AdmissionReviewStatus {
	glog.V(2).Info("admitting crd")
	cr := struct {
		metav1.ObjectMeta
		Data map[string]string
	}{}

	raw := ar.Spec.Object.Raw
	err := json.Unmarshal(raw, &cr)
	if err != nil {
		glog.Error(err)
		return toAdmissionReviewStatus(err)
	}

	reviewStatus := v1alpha1.AdmissionReviewStatus{}
	reviewStatus.Allowed = true
	for k, v := range cr.Data {
		if k == "webhook-e2e-test" && v == "webhook-disallow" {
			reviewStatus.Allowed = false
			reviewStatus.Result = &metav1.Status{
				Reason: "the custom resource contains unwanted data",
			}
		}
	}
	return &reviewStatus
}

type admitFunc func(v1alpha1.AdmissionReview) *v1alpha1.AdmissionReviewStatus

func serve(w http.ResponseWriter, r *http.Request, admit admitFunc) {
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

	var reviewStatus *v1alpha1.AdmissionReviewStatus
	ar := v1alpha1.AdmissionReview{}
	deserializer := codecs.UniversalDeserializer()
	if _, _, err := deserializer.Decode(body, nil, &ar); err != nil {
		glog.Error(err)
		reviewStatus = toAdmissionReviewStatus(err)
	} else {
		reviewStatus = admit(ar)
	}

	if reviewStatus != nil {
		ar.Status = *reviewStatus
	}

	resp, err := json.Marshal(ar)
	if err != nil {
		glog.Error(err)
	}
	if _, err := w.Write(resp); err != nil {
		glog.Error(err)
	}
}

func servePods(w http.ResponseWriter, r *http.Request) {
	serve(w, r, admitPods)
}

func serveConfigmaps(w http.ResponseWriter, r *http.Request) {
	serve(w, r, admitConfigMaps)
}

func serveCRD(w http.ResponseWriter, r *http.Request) {
	serve(w, r, admitCRD)
}

func main() {
	var config Config
	config.addFlags()
	flag.Parse()

	http.HandleFunc("/pods", servePods)
	http.HandleFunc("/configmaps", serveConfigmaps)
	http.HandleFunc("/crd", serveCRD)
	clientset := getClient()
	server := &http.Server{
		Addr:      ":443",
		TLSConfig: configTLS(config, clientset),
	}
	server.ListenAndServeTLS("", "")
}
