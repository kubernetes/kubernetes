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

// OWNER = sig/cli

package kubectl

import (
	"context"
	"strings"

	"github.com/google/go-cmp/cmp"
	"github.com/onsi/ginkgo/v2"

	clientset "k8s.io/client-go/kubernetes"
	commonutils "k8s.io/kubernetes/test/e2e/common"
	"k8s.io/kubernetes/test/e2e/framework"
	e2ekubectl "k8s.io/kubernetes/test/e2e/framework/kubectl"
	imageutils "k8s.io/kubernetes/test/utils/image"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("kubectl apply", func() {
	defer ginkgo.GinkgoRecover()
	f := framework.NewDefaultFramework("kubectl")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	var c clientset.Interface
	var ns string
	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name
	})

	ginkgo.It("should apply a new configuration to an existing RC", func(ctx context.Context) {
		controllerJSON := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostControllerFilename)))

		ginkgo.By("creating Agnhost RC")
		e2ekubectl.RunKubectlOrDieInput(ns, controllerJSON, "create", "-f", "-")
		ginkgo.By("applying a modified configuration")
		stdin := modifyReplicationControllerConfiguration(controllerJSON)
		e2ekubectl.NewKubectlCommand(ns, "apply", "-f", "-").
			WithStdinReader(stdin).
			ExecOrDie(ns)
		ginkgo.By("checking the result")
		forEachReplicationController(ctx, c, ns, "app", "agnhost", validateReplicationControllerConfiguration)
	})
	ginkgo.It("should reuse port when apply to an existing SVC", func(ctx context.Context) {
		serviceJSON := readTestFileOrDie(agnhostServiceFilename)

		ginkgo.By("creating Agnhost SVC")
		e2ekubectl.RunKubectlOrDieInput(ns, string(serviceJSON), "create", "-f", "-")

		ginkgo.By("getting the original port")
		originalNodePort := e2ekubectl.RunKubectlOrDie(ns, "get", "service", "agnhost-primary", "-o", "jsonpath={.spec.ports[0].port}")

		ginkgo.By("applying the same configuration")
		e2ekubectl.RunKubectlOrDieInput(ns, string(serviceJSON), "apply", "-f", "-")

		ginkgo.By("getting the port after applying configuration")
		currentNodePort := e2ekubectl.RunKubectlOrDie(ns, "get", "service", "agnhost-primary", "-o", "jsonpath={.spec.ports[0].port}")

		ginkgo.By("checking the result")
		if originalNodePort != currentNodePort {
			framework.Failf("port should keep the same")
		}
	})

	ginkgo.It("apply set/view last-applied", func(ctx context.Context) {
		deployment1Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostDeployment1Filename)))
		deployment2Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostDeployment2Filename)))
		deployment3Yaml := commonutils.SubstituteImageName(string(readTestFileOrDie(agnhostDeployment3Filename)))

		ginkgo.By("deployment replicas number is 2")
		e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "-f", "-")

		ginkgo.By("check the last-applied matches expectations annotations")
		output := e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "view-last-applied", "-f", "-", "-o", "json")
		requiredString := "\"replicas\": 2"
		if !strings.Contains(output, requiredString) {
			framework.Failf("Missing %s in kubectl view-last-applied", requiredString)
		}

		ginkgo.By("apply file doesn't have replicas")
		e2ekubectl.RunKubectlOrDieInput(ns, deployment2Yaml, "apply", "set-last-applied", "-f", "-")

		ginkgo.By("check last-applied has been updated, annotations doesn't have replicas")
		output = e2ekubectl.RunKubectlOrDieInput(ns, deployment1Yaml, "apply", "view-last-applied", "-f", "-", "-o", "json")
		requiredString = "\"replicas\": 2"
		if strings.Contains(output, requiredString) {
			framework.Failf("Presenting %s in kubectl view-last-applied", requiredString)
		}

		ginkgo.By("scale set replicas to 3")
		agnhostDeploy := "agnhost-deployment"
		debugDiscovery()
		e2ekubectl.RunKubectlOrDie(ns, "scale", "deployment", agnhostDeploy, "--replicas=3")

		ginkgo.By("apply file doesn't have replicas but image changed")
		e2ekubectl.RunKubectlOrDieInput(ns, deployment3Yaml, "apply", "-f", "-")

		ginkgo.By("verify replicas still is 3 and image has been updated")
		output = e2ekubectl.RunKubectlOrDieInput(ns, deployment3Yaml, "get", "-f", "-", "-o", "json")
		requiredItems := []string{"\"replicas\": 3", imageutils.GetE2EImage(imageutils.AgnhostPrev)}
		for _, item := range requiredItems {
			if !strings.Contains(output, item) {
				framework.Failf("Missing %s in kubectl apply", item)
			}
		}
	})

	ginkgo.Describe("Kubectl prune with applyset", func() {
		ginkgo.It("should apply and prune objects", func(ctx context.Context) {
			framework.Logf("applying manifest1")
			manifest1 := `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
  namespace: {{ns}}
---
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm2
  namespace: {{ns}}
`

			manifest1 = strings.ReplaceAll(manifest1, "{{ns}}", ns)
			args := []string{"apply", "--prune", "--applyset=applyset1", "-f", "-"}
			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest1).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects := mustListObjectsInNamespace(ctx, c, ns)
			names := mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1", "cm2"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}

			framework.Logf("applying manifest2")
			manifest2 := `
apiVersion: v1
kind: ConfigMap
metadata:
  name: cm1
  namespace: {{ns}}
`
			manifest2 = strings.ReplaceAll(manifest2, "{{ns}}", ns)

			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest2).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects = mustListObjectsInNamespace(ctx, c, ns)
			names = mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}

			framework.Logf("applying manifest2 (again)")
			e2ekubectl.NewKubectlCommand(ns, args...).AppendEnv([]string{"KUBECTL_APPLYSET=true"}).WithStdinData(manifest2).ExecOrDie(ns)

			framework.Logf("checking which objects exist")
			objects = mustListObjectsInNamespace(ctx, c, ns)
			names = mustGetNames(objects)
			if diff := cmp.Diff(names, []string{"cm1"}); diff != "" {
				framework.Failf("unexpected configmap names (-want +got):\n%s", diff)
			}
		})
	})
})
