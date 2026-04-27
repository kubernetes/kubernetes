/*
Copyright 2021 The Kubernetes Authors.

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

package auth

import (
	"context"

	"k8s.io/kubernetes/test/e2e/cloud/gcp/common"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/upgrades"
	"k8s.io/kubernetes/test/e2e/upgrades/auth"
	"k8s.io/kubernetes/test/utils/junit"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var upgradeTests = []upgrades.Test{
	&auth.ServiceAccountAdmissionControllerMigrationTest{},
}

var _ = SIGDescribe("ServiceAccount admission controller migration", feature.BoundServiceAccountTokenVolume, func() {
	f := framework.NewDefaultFramework("serviceaccount-admission-controller-migration")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged
	testFrameworks := upgrades.CreateUpgradeFrameworks(upgradeTests)

	ginkgo.Describe("master upgrade", func() {
		ginkgo.It("should maintain a functioning cluster", func(ctx context.Context) {
			upgCtx, err := common.GetUpgradeContext(f.ClientSet.Discovery())
			framework.ExpectNoError(err)

			testSuite := &junit.TestSuite{Name: "ServiceAccount admission controller migration"}
			serviceaccountAdmissionControllerMigrationTest := &junit.TestCase{
				Name:      "[sig-auth] serviceaccount-admission-controller-migration",
				Classname: "upgrade_tests",
			}
			testSuite.TestCases = append(testSuite.TestCases, serviceaccountAdmissionControllerMigrationTest)

			upgradeFunc := common.ControlPlaneUpgradeFunc(f, upgCtx, serviceaccountAdmissionControllerMigrationTest, nil)
			upgrades.RunUpgradeSuite(ctx, upgCtx, upgradeTests, testFrameworks, testSuite, upgrades.MasterUpgrade, upgradeFunc)
		})
	})
})
