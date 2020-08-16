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
	"context"
	"fmt"
	"strings"
	"time"

	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

var (
	requiredHealthzChecks = sets.NewString(
		"[+]ping ok",
		"[+]log ok",
		"[+]etcd ok",
		"[+]poststarthook/start-kube-apiserver-admission-initializer ok",
		"[+]poststarthook/generic-apiserver-start-informers ok",
		"[+]poststarthook/start-apiextensions-informers ok",
		"[+]poststarthook/start-apiextensions-controllers ok",
		"[+]poststarthook/crd-informer-synced ok",
		"[+]poststarthook/bootstrap-controller ok",
		"[+]poststarthook/scheduling/bootstrap-system-priority-classes ok",
		"[+]poststarthook/start-cluster-authentication-info-controller ok",
		"[+]poststarthook/start-kube-aggregator-informers ok",
		"[+]poststarthook/apiservice-registration-controller ok",
		"[+]poststarthook/apiservice-status-available-controller ok",
		"[+]poststarthook/kube-apiserver-autoregistration ok",
		"[+]autoregister-completion ok",
		"[+]poststarthook/apiservice-openapi-controller ok",
	)
	requiredLivezChecks = sets.NewString(
		"[+]ping ok",
		"[+]log ok",
		"[+]etcd ok",
		"[+]poststarthook/start-kube-apiserver-admission-initializer ok",
		"[+]poststarthook/generic-apiserver-start-informers ok",
		"[+]poststarthook/start-apiextensions-informers ok",
		"[+]poststarthook/start-apiextensions-controllers ok",
		"[+]poststarthook/crd-informer-synced ok",
		"[+]poststarthook/bootstrap-controller ok",
		"[+]poststarthook/scheduling/bootstrap-system-priority-classes ok",
		"[+]poststarthook/start-cluster-authentication-info-controller ok",
		"[+]poststarthook/start-kube-aggregator-informers ok",
		"[+]poststarthook/apiservice-registration-controller ok",
		"[+]poststarthook/apiservice-status-available-controller ok",
		"[+]poststarthook/kube-apiserver-autoregistration ok",
		"[+]autoregister-completion ok",
		"[+]poststarthook/apiservice-openapi-controller ok",
	)
	requiredReadyzChecks = sets.NewString(
		"[+]ping ok",
		"[+]log ok",
		"[+]etcd ok",
		"[+]informer-sync ok",
		"[+]poststarthook/start-kube-apiserver-admission-initializer ok",
		"[+]poststarthook/generic-apiserver-start-informers ok",
		"[+]poststarthook/start-apiextensions-informers ok",
		"[+]poststarthook/start-apiextensions-controllers ok",
		"[+]poststarthook/crd-informer-synced ok",
		"[+]poststarthook/bootstrap-controller ok",
		"[+]poststarthook/scheduling/bootstrap-system-priority-classes ok",
		"[+]poststarthook/start-cluster-authentication-info-controller ok",
		"[+]poststarthook/start-kube-aggregator-informers ok",
		"[+]poststarthook/apiservice-registration-controller ok",
		"[+]poststarthook/apiservice-status-available-controller ok",
		"[+]poststarthook/kube-apiserver-autoregistration ok",
		"[+]autoregister-completion ok",
		"[+]poststarthook/apiservice-openapi-controller ok",
	)
)

func testPath(client clientset.Interface, path string, requiredChecks sets.String) error {
	var result restclient.Result
	err := wait.Poll(100*time.Millisecond, 30*time.Second, func() (bool, error) {
		result = client.CoreV1().RESTClient().Get().RequestURI(path).Do(context.TODO())
		status := 0
		result.StatusCode(&status)
		return status == 200, nil
	})
	if err != nil {
		return err
	}
	body, err := result.Raw()
	if err != nil {
		return err
	}
	checks := sets.NewString(strings.Split(string(body), "\n")...)
	if missing := requiredChecks.Difference(checks); missing.Len() > 0 {
		return fmt.Errorf("missing required %s checks: %v in: %s", path, missing, string(body))
	}
	return nil
}

var _ = SIGDescribe("health handlers", func() {
	f := framework.NewDefaultFramework("health")

	ginkgo.It("should contain necessary checks", func() {
		ginkgo.By("/health")
		err := testPath(f.ClientSet, "/healthz?verbose=1", requiredHealthzChecks)
		framework.ExpectNoError(err)

		ginkgo.By("/livez")
		err = testPath(f.ClientSet, "/livez?verbose=1", requiredLivezChecks)
		framework.ExpectNoError(err)

		ginkgo.By("/readyz")
		err = testPath(f.ClientSet, "/readyz?verbose=1", requiredReadyzChecks)
		framework.ExpectNoError(err)
	})
})
