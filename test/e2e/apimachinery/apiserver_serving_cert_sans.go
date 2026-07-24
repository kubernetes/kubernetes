/*
Copyright 2026 The Kubernetes Authors.

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
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/client-go/util/cert"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

var _ = SIGDescribe("kube-apiserver serving certificate", func() {
	f := framework.NewDefaultFramework("apiserver-serving-cert")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	/*
		Release: v1.37
		Testname: kube-apiserver serving certificate, in-cluster SANs
		Description: There are conformance tests that verify in-cluster clients can resolve
		names such as kubernetes.default via DNS. The kube-apiserver serving certificate MUST
		also list those same names as Subject Alternative Names, otherwise in-cluster clients
		that connect using them will fail TLS hostname verification even though DNS resolution
		succeeds.
	*/
	framework.ConformanceIt("should include the in-cluster kubernetes service names in its serving certificate SANs", func(ctx context.Context) {
		host := f.ClientConfig().Host

		ginkgo.By(fmt.Sprintf("Fetching the kube-apiserver serving certificate from %q", host))
		certs, _, err := cert.GetServingCertificatesForURL(host, "")
		framework.ExpectNoError(err, "fetching kube-apiserver serving certificate from %q", host)
		if len(certs) == 0 {
			framework.Failf("no serving certificate returned by kube-apiserver at %q", host)
		}
		leaf := certs[0]

		dnsNames := sets.New[string](leaf.DNSNames...)
		expectedNames := []string{
			"kubernetes",
			"kubernetes.default",
			"kubernetes.default.svc",
			// framework.TestContext.ClusterDNSDomain reflects the cluster's actual
			// --cluster-domain and defaults to "cluster.local" when unset, so this
			// isn't hardcoding the domain for clusters configured differently.
			fmt.Sprintf("kubernetes.default.svc.%s", framework.TestContext.ClusterDNSDomain),
		}

		var missing []string
		for _, name := range expectedNames {
			if !dnsNames.Has(name) {
				missing = append(missing, name)
			}
		}
		if len(missing) > 0 {
			framework.Failf("kube-apiserver serving certificate is missing expected SAN DNS name(s) %v; certificate DNSNames: %v", missing, leaf.DNSNames)
		}
	})
})
