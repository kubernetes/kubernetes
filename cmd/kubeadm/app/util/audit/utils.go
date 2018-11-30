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

	"github.com/pkg/errors"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apiserver/pkg/apis/audit/install"
	auditv1 "k8s.io/apiserver/pkg/apis/audit/v1"
	"k8s.io/kubernetes/cmd/kubeadm/app/util"
)

// CreateDefaultAuditLogPolicy writes the default audit log policy to disk.
func CreateDefaultAuditLogPolicy(policyFile string) error {
	policy := auditv1.Policy{
		TypeMeta: metav1.TypeMeta{
			APIVersion: auditv1.SchemeGroupVersion.String(),
			Kind:       "Policy",
		},
		Rules: []auditv1.PolicyRule{
			{
				Level: auditv1.LevelMetadata,
			},
		},
	}
	return writePolicyToDisk(policyFile, &policy)
}

func writePolicyToDisk(policyFile string, policy *auditv1.Policy) error {
	// creates target folder if not already exists
	if err := os.MkdirAll(filepath.Dir(policyFile), 0700); err != nil {
		return errors.Wrapf(err, "failed to create directory %q: ", filepath.Dir(policyFile))
	}

	scheme := runtime.NewScheme()
	// Registers the API group with the scheme and adds types to a scheme
	install.Install(scheme)

	codecs := serializer.NewCodecFactory(scheme)

	// writes the policy to disk
	serialized, err := util.MarshalToYamlForCodecs(policy, auditv1.SchemeGroupVersion, codecs)

	if err != nil {
		return errors.Wrap(err, "failed to marshal audit policy to YAML")
	}

	if err := ioutil.WriteFile(policyFile, serialized, 0600); err != nil {
		return errors.Wrapf(err, "failed to write audit policy to %v: ", policyFile)
	}

	return nil
}
