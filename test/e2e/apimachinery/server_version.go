/*
Copyright 2020 The Kubernetes Authors.

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

package apimachinery

import (
	"regexp"

	"k8s.io/apimachinery/pkg/version"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("server version", func() {
	f := framework.NewDefaultFramework("server-version")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged

	/*
	   Release: v1.19
	   Testname: Confirm a server version
	   Description: Ensure that an API server version can be retrieved.
	   Both the major and minor versions MUST only be an integer.
	*/
	framework.ConformanceIt("should find the server version", func() {

		ginkgo.By("Request ServerVersion")

		var version *version.Info
		version, err := f.ClientSet.Discovery().ServerVersion()
		framework.ExpectNoError(err, "Fail to access ServerVersion")

		ginkgo.By("Confirm major version")
		re := regexp.MustCompile("[1-9]")
		framework.ExpectEqual(re.FindString(version.Major), version.Major, "unable to find major version")
		framework.Logf("Major version: %v", version.Major)

		ginkgo.By("Confirm minor version")

		re = regexp.MustCompile("[^0-9]+")
		cleanMinorVersion := re.ReplaceAllString(version.Minor, "")
		framework.Logf("cleanMinorVersion: %v", cleanMinorVersion)

		re = regexp.MustCompile("[0-9]+")
		framework.ExpectEqual(re.FindString(version.Minor), cleanMinorVersion, "unable to find minor version")
		framework.Logf("Minor version: %v", version.Minor)
	})
})
