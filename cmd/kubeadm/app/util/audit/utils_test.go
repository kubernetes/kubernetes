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

package audit

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
)

func cleanup(t *testing.T, path string) {
	err := os.RemoveAll(path)
	if err != nil {
		t.Fatalf("Failed to clean up %v: %v", path, err)
	}
}

func TestCreateDefaultAuditLogPolicy(t *testing.T) {
	// make a tempdir
	tempDir, err := ioutil.TempDir("/tmp", "audit-test")
	if err != nil {
		t.Fatalf("could not create a tempdir: %v", err)
	}
	defer cleanup(t, tempDir)
	auditPolicyFile := filepath.Join(tempDir, "test.yaml")
	if err = CreateDefaultAuditLogPolicy(auditPolicyFile); err != nil {
		t.Fatalf("failed to create audit log policy: %v", err)
	}
	// turn the audit log back into a policy
	policyBytes, err := ioutil.ReadFile(auditPolicyFile)
	if err != nil {
		t.Fatalf("failed to read %v: %v", auditPolicyFile, err)
	}
	scheme := runtime.NewScheme()
	install.Install(scheme)
	codecs := serializer.NewCodecFactory(scheme)
	policy := auditv1.Policy{}
	err = runtime.DecodeInto(codecs.UniversalDecoder(), policyBytes, &policy)
	if err != nil {
		t.Fatalf("failed to decode written policy: %v", err)
	}
	if policy.Kind != "Policy" {
		t.Fatalf("did not decode policy properly")
	}
}
