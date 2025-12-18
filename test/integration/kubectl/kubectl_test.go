/*
Copyright The Kubernetes Authors.

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

package kubectl

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/rogpeppe/go-internal/testscript"

	clientset "k8s.io/client-go/kubernetes"
	restclient "k8s.io/client-go/rest"
	"k8s.io/client-go/tools/clientcmd"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	kubecontrollermanagertesting "k8s.io/kubernetes/cmd/kube-controller-manager/app/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/kubernetes/test/utils/kubeconfig"
)

func TestKubectl(t *testing.T) {
	tCtx := ktesting.Init(t)
	closeFn, restConfig, client := setup(tCtx, t)
	t.Cleanup(closeFn)

	kubeconfigFile := filepath.Join(t.TempDir(), "kubeconfig")
	clientConfig := kubeconfig.CreateKubeConfig(restConfig)
	if err := clientcmd.WriteToFile(*clientConfig, kubeconfigFile); err != nil {
		t.Fatalf("Error writing kubeconfig: %v", err)
	}

	// TODO: create namespace per test file and ideally inject this into the current-context in kubeconfig
	namespace := framework.CreateNamespaceOrDie(client, "kubectl", t)
	workDir, err := os.Getwd()
	if err != nil {
		t.Fatalf("Error reading current directory: %v", err)
	}

	testscript.Run(t, testscript.Params{
		Dir: "testdata",
		Setup: func(e *testscript.Env) error {
			// workDir is always the current directory the test runs in,
			// so we need to jump 3 dirs above
			e.Setenv("K8S_ROOT", filepath.Join(workDir, "../../../"))
			e.Setenv("NAMESPACE", namespace.Name)
			e.Setenv("KUBECONFIG", kubeconfigFile)
			return nil
		},
	})
}

func setup(ctx context.Context, t *testing.T) (framework.TearDownFunc, *restclient.Config, clientset.Interface) {
	apiserver := kubeapiservertesting.StartTestServerOrDie(t, nil, framework.DefaultTestServerFlags(), framework.SharedEtcd())

	clientConfig := kubeconfig.CreateKubeConfig(apiserver.ClientConfig)
	kubeConfigFile := filepath.Join(t.TempDir(), "kubeconfig")
	if err := clientcmd.WriteToFile(*clientConfig, kubeConfigFile); err != nil {
		t.Fatalf("Error writing kubeconfig: %v", err)
	}

	controllermanager := kubecontrollermanagertesting.StartTestServerOrDie(t, ctx, []string{
		"--kubeconfig=" + kubeConfigFile,
		"--controllers=garbagecollector,resourcequota,replicaset", // we only need handful of controllers for this test
		"--leader-elect=false",                                    // KCM leader election calls os.Exit when it ends, so it is easier to just turn it off altogether
	})

	tearDownFunc := func() {
		controllermanager.TearDownFn()
		apiserver.TearDownFn()
	}

	config := restclient.CopyConfig(apiserver.ClientConfig)
	clientSet, err := clientset.NewForConfig(config)
	if err != nil {
		t.Fatalf("Error creating clientset: %v", err)
	}

	return tearDownFunc, config, clientSet
}
