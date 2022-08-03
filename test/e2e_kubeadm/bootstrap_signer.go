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
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	bootstrapTokensSignerRoleName = "system:controller:bootstrap-signer"
)

// Define container for all the test specification aimed at verifying
// that kubeadm creates the bootstrap signer
var _ = Describe("bootstrap signer", func() {

	// Get an instance of the k8s test framework
	f := framework.NewDefaultFramework("bootstrap token")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	// Tests in this container are not expected to create new objects in the cluster
	// so we are disabling the creation of a namespace in order to get a faster execution
	f.SkipNamespaceCreation = true

	ginkgo.It("should be active", func() {
		//NB. this is technically implemented a part of the control-plane phase
		//    and more specifically if the controller manager is properly configured,
		//    the bootstrapsigner controller is activated and the system:controller:bootstrap-signer
		//    group will be automatically created
		ExpectRole(f.ClientSet, kubeSystemNamespace, bootstrapTokensSignerRoleName)
	})
})
