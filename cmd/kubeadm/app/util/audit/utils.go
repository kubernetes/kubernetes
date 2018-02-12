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
	"fmt"
	"io/ioutil"
	"os"
	"path/filepath"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	auditv1beta1 "k8s.io/apiserver/pkg/apis/audit/v1beta1"
	"k8s.io/client-go/kubernetes/scheme"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// CreateDefaultAuditLogPolicy writes the default audit log policy to disk.
func CreateDefaultAuditLogPolicy(policyFile string) error {
	policy := auditv1beta1.Policy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: "audit.k8s.io/v1beta1",
			Kind:       "Policy",
		},
		Rules: []auditv1beta1.PolicyRule{
			{
				Level: auditv1beta1.LevelMetadata,
			},
		},
	}
	return writePolicyToDisk(policyFile, &policy)
}

func writePolicyToDisk(policyFile string, policy *auditv1beta1.Policy) error {
	// creates target folder if not already exists
	if err := os.MkdirAll(filepath.Dir(policyFile), 0700); err != nil {
		return fmt.Errorf("failed to create directory %q: %v", filepath.Dir(policyFile), err)
	}

	// Registers auditv1beta1 with the runtime Scheme
	auditv1beta1.AddToScheme(scheme.Scheme)

	// writes the policy to disk
	serialized, err := util.MarshalToYaml(policy, auditv1beta1.SchemeGroupVersion)
	if err != nil {
		return fmt.Errorf("failed to marshal audit policy to YAML: %v", err)
	}

	if err := ioutil.WriteFile(policyFile, serialized, 0600); err != nil {
		return fmt.Errorf("failed to write audit policy to %v: %v", policyFile, err)
	}

	return nil
}
