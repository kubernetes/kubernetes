/*
Copyright 2019 The Kubernetes Authors.

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

package kubeadm

import (
	"context"

	"gopkg.in/yaml.v2"
	authv1 "k8s.io/api/authorization/v1"
	rbacv1 "k8s.io/api/rbac/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
)

const (
	kubeadmConfigName                             = "kubeadm-config"
	kubeadmConfigRoleName                         = "kubeadm:nodes-kubeadm-config"
	kubeadmConfigRoleBindingName                  = kubeadmConfigRoleName
	kubeadmConfigClusterConfigurationConfigMapKey = "ClusterConfiguration"
)

var (
	kubeadmConfigConfigMapResource = &authv1.ResourceAttributes{
		Namespace: kubeSystemNamespace,
		Name:      kubeadmConfigName,
		Resource:  "configmaps",
		Verb:      "get",
	}
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the cluster-info ConfigMap, that it is properly configured
// and that all the related RBAC rules are in place
var _ = Describe("kubeadm-config ConfigMap", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("kubeadm-config")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should exist and be properly configured", func(ctx context.Context) {
		cm := GetConfigMap(f.ClientSet, kubeSystemNamespace, kubeadmConfigName)

		gomega.Expect(cm.Data).To(gomega.HaveKey(kubeadmConfigClusterConfigurationConfigMapKey))
	})

	ginkgo.It("should have related Role and RoleBinding", func(ctx context.Context) {
		ExpectRole(f.ClientSet, kubeSystemNamespace, kubeadmConfigRoleName)
		ExpectRoleBinding(f.ClientSet, kubeSystemNamespace, kubeadmConfigRoleBindingName)
	})

	ginkgo.It("should be accessible for bootstrap tokens", func(ctx context.Context) {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, bootstrapTokensGroup,
			kubeadmConfigConfigMapResource,
		)
	})

	ginkgo.It("should be accessible for nodes", func(ctx context.Context) {
		ExpectSubjectHasAccessToResource(f.ClientSet,
			rbacv1.GroupKind, nodesGroup,
			kubeadmConfigConfigMapResource,
		)
	})
})

func getClusterConfiguration(c clientset.Interface) map[interface{}]interface{} {
	cm := GetConfigMap(c, kubeSystemNamespace, kubeadmConfigName)

	gomega.Expect(cm.Data).To(gomega.HaveKey(kubeadmConfigClusterConfigurationConfigMapKey))

	return unmarshalYaml(cm.Data[kubeadmConfigClusterConfigurationConfigMapKey])
}

func unmarshalYaml(data string) map[interface{}]interface{} {
	m := make(map[interface{}]interface{})
	err := yaml.Unmarshal([]byte(data), &m)
	if err != nil {
		framework.Failf("error parsing %s ConfigMap: %v", kubeadmConfigName, err)
	}
	return m
}
