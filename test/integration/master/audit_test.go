/*
Copyright 2018 The Kubernetes Authors.

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

package master

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
	"os"
	"strings"
	"testing"

	apiv1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	auditinternal "k8s.io/apiserver/pkg/apis/audit"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/client-go/kubernetes"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils"

	"github.com/evanphx/json-patch"
)

var (
	auditPolicyPattern = `
apiVersion: {version}
kind: Policy
rules:
  - level: RequestResponse
    resources:
      - group: "" # core
        resources: ["configmaps"]

`
	namespace              = "default"
	watchTestTimeout int64 = 1
	watchOptions           = metav1.ListOptions{TimeoutSeconds: &watchTestTimeout}
	patch, _               = json.Marshal(jsonpatch.Patch{})
	auditTestUser          = "system:apiserver"
	versions               = map[string]schema.GroupVersion{
		"audit.k8s.io/v1":      auditv1.SchemeGroupVersion,
		"audit.k8s.io/v1beta1": auditv1beta1.SchemeGroupVersion,
	}

	expectedEvents = []utils.AuditEvent{
		{
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
			Verb:              "create",
			Code:              201,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     true,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
			Verb:              "get",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     false,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps", namespace),
			Verb:              "list",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     false,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseStarted,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
			Verb:              "watch",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     false,
			ResponseObject:    false,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps?timeout=%ds&timeoutSeconds=%d&watch=true", namespace, watchTestTimeout, watchTestTimeout),
			Verb:              "watch",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     false,
			ResponseObject:    false,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
			Verb:              "update",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     true,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
			Verb:              "patch",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     true,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		}, {
			Level:             auditinternal.LevelRequestResponse,
			Stage:             auditinternal.StageResponseComplete,
			RequestURI:        fmt.Sprintf("/api/v1/namespaces/%s/configmaps/audit-configmap", namespace),
			Verb:              "delete",
			Code:              200,
			User:              auditTestUser,
			Resource:          "configmaps",
			Namespace:         namespace,
			RequestObject:     true,
			ResponseObject:    true,
			AuthorizeDecision: "allow",
		},
	}
)

// TestAudit ensures that both v1beta1 and v1 version audit api could work.
func TestAudit(t *testing.T) {
	for version := range versions {
		testAudit(t, version)
	}
}

func testAudit(t *testing.T, version string) {
	// prepare audit policy file
	auditPolicy := []byte(strings.Replace(auditPolicyPattern, "{version}", version, 1))
	policyFile, err := ioutil.TempFile("", "audit-policy.yaml")
	if err != nil {
		t.Fatalf("Failed to create audit policy file: %v", err)
	}
	defer os.Remove(policyFile.Name())
	if _, err := policyFile.Write(auditPolicy); err != nil {
		t.Fatalf("Failed to write audit policy file: %v", err)
	}
	if err := policyFile.Close(); err != nil {
		t.Fatalf("Failed to close audit policy file: %v", err)
	}

	// prepare audit log file
	logFile, err := ioutil.TempFile("", "audit.log")
	if err != nil {
		t.Fatalf("Failed to create audit log file: %v", err)
	}
	defer os.Remove(logFile.Name())

	// start api server
	result := kubeapiservertesting.StartTestServerOrDie(t, nil,
		[]string{
			"--audit-policy-file", policyFile.Name(),
			"--audit-log-version", version,
			"--audit-log-mode", "blocking",
			"--audit-log-path", logFile.Name()},
		framework.SharedEtcd())
	defer result.TearDownFn()

	kubeclient, err := kubernetes.NewForConfig(result.ClientConfig)
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}

	// perform configmap operations
	configMapOperations(t, kubeclient)

	// check for corresponding audit logs
	stream, err := os.Open(logFile.Name())
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	defer stream.Close()
	missingReport, err := utils.CheckAuditLines(stream, expectedEvents, versions[version])
	if err != nil {
		t.Fatalf("Unexpected error: %v", err)
	}
	if len(missingReport.MissingEvents) > 0 {
		t.Errorf(missingReport.String())
	}
}

// configMapOperations is a set of known operations performed on the configmap type
// which correspond to the expected events.
// This is shared by the dynamic test
func configMapOperations(t *testing.T, kubeclient kubernetes.Interface) {
	// create, get, watch, update, patch, list and delete configmap.
	configMap := &apiv1.ConfigMap{
		ObjectMeta: metav1.ObjectMeta{
			Name: "audit-configmap",
		},
		Data: map[string]string{
			"map-key": "map-value",
		},
	}

	_, err := kubeclient.CoreV1().ConfigMaps(namespace).Create(configMap)
	expectNoError(t, err, "failed to create audit-configmap")

	_, err = kubeclient.CoreV1().ConfigMaps(namespace).Get(configMap.Name, metav1.GetOptions{})
	expectNoError(t, err, "failed to get audit-configmap")

	configMapChan, err := kubeclient.CoreV1().ConfigMaps(namespace).Watch(watchOptions)
	expectNoError(t, err, "failed to create watch for config maps")
	for range configMapChan.ResultChan() {
		// Block until watchOptions.TimeoutSeconds expires.
		// If the test finishes before watchOptions.TimeoutSeconds expires, the watch audit
		// event at stage ResponseComplete will not be generated.
	}

	_, err = kubeclient.CoreV1().ConfigMaps(namespace).Update(configMap)
	expectNoError(t, err, "failed to update audit-configmap")

	_, err = kubeclient.CoreV1().ConfigMaps(namespace).Patch(configMap.Name, types.JSONPatchType, patch)
	expectNoError(t, err, "failed to patch configmap")

	_, err = kubeclient.CoreV1().ConfigMaps(namespace).List(metav1.ListOptions{})
	expectNoError(t, err, "failed to list config maps")

	err = kubeclient.CoreV1().ConfigMaps(namespace).Delete(configMap.Name, &metav1.DeleteOptions{})
	expectNoError(t, err, "failed to delete audit-configmap")
}

func expectNoError(t *testing.T, err error, msg string) {
	if err != nil {
		t.Fatalf("%s: %v", msg, err)
	}
}
