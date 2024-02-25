/*
Copyright 2017 The Kubernetes Authors.

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
	"bytes"
	"context"
	"fmt"
	"text/tabwriter"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"

	authorizationv1 "k8s.io/api/authorization/v1"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/client-go/util/workqueue"
	admissionapi "k8s.io/pod-security-admission/api"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/cli-runtime/pkg/printers"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

var serverPrintVersion = utilversion.MustParseSemantic("v1.10.0")

var _ = SIGDescribe("Servers with support for Table transformation", func() {
	f := framework.NewDefaultFramework("tables")
	f.NamespacePodSecurityLevel = admissionapi.LevelBaseline

	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessServerVersionGTE(serverPrintVersion, f.ClientSet.Discovery())
	})

	ginkgo.It("should return pod details", func(ctx context.Context) {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "pod-1"
		framework.Logf("Creating pod %s", podName)

		_, err := c.CoreV1().Pods(ns).Create(ctx, newTablePod(ns, podName), metav1.CreateOptions{})
		framework.ExpectNoError(err, "failed to create pod %s in namespace: %s", podName, ns)

		table := &metav1beta1.Table{}
		err = c.CoreV1().RESTClient().Get().Resource("pods").Namespace(ns).Name(podName).SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do(ctx).Into(table)
		framework.ExpectNoError(err, "failed to get pod %s in Table form in namespace: %s", podName, ns)
		framework.Logf("Table: %#v", table)

		gomega.Expect(len(table.ColumnDefinitions)).To(gomega.BeNumerically(">", 2))
		gomega.Expect(table.Rows).To(gomega.HaveLen(1))
		gomega.Expect(table.Rows[0].Cells).To(gomega.HaveLen(len(table.ColumnDefinitions)))
		gomega.Expect(table.ColumnDefinitions[0].Name).To(gomega.Equal("Name"))
		gomega.Expect(table.Rows[0].Cells[0]).To(gomega.Equal(podName))

		out := printTable(table)
		gomega.Expect(out).To(gomega.MatchRegexp("^NAME\\s"))
		gomega.Expect(out).To(gomega.MatchRegexp("\npod-1\\s"))
		framework.Logf("Table:\n%s", out)
	})

	ginkgo.It("should return chunks of table results for list calls", func(ctx context.Context) {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		ginkgo.By("creating a large number of resources")
		workqueue.ParallelizeUntil(ctx, 5, 20, func(i int) {
			for tries := 3; tries >= 0; tries-- {
				_, err := client.Create(ctx, &v1.PodTemplate{
					ObjectMeta: metav1.ObjectMeta{
						Name: fmt.Sprintf("template-%04d", i),
					},
					Template: v1.PodTemplateSpec{
						Spec: v1.PodSpec{
							Containers: []v1.Container{
								{Name: "test", Image: "test2"},
							},
						},
					},
				}, metav1.CreateOptions{})
				if err == nil {
					return
				}
				framework.Logf("Got an error creating template %d: %v", i, err)
			}
			framework.Failf("Unable to create template %d, exiting", i)
		})

		pagedTable := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Limit: 2}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do(ctx).Into(pagedTable)
		framework.ExpectNoError(err, "failed to get pod templates in Table form in namespace: %s", ns)
		gomega.Expect(pagedTable.Rows).To(gomega.HaveLen(2))
		gomega.Expect(pagedTable.ResourceVersion).ToNot(gomega.BeEmpty())
		gomega.Expect(pagedTable.Continue).ToNot(gomega.BeEmpty())
		gomega.Expect(pagedTable.Rows[0].Cells[0]).To(gomega.Equal("template-0000"))
		gomega.Expect(pagedTable.Rows[1].Cells[0]).To(gomega.Equal("template-0001"))

		err = c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Continue: pagedTable.Continue}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do(ctx).Into(pagedTable)
		framework.ExpectNoError(err, "failed to get pod templates in Table form in namespace: %s", ns)
		gomega.Expect(pagedTable.Rows).ToNot(gomega.BeEmpty())
		gomega.Expect(pagedTable.Rows[0].Cells[0]).To(gomega.Equal("template-0002"))
	})

	ginkgo.It("should return generic metadata details across all namespaces for nodes", func(ctx context.Context) {
		c := f.ClientSet

		table := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Resource("nodes").SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do(ctx).Into(table)
		framework.ExpectNoError(err, "failed to get nodes in Table form across all namespaces")
		framework.Logf("Table: %#v", table)

		gomega.Expect(len(table.ColumnDefinitions)).To(gomega.BeNumerically(">=", 2))
		gomega.Expect(table.Rows).ToNot(gomega.BeEmpty())
		gomega.Expect(table.Rows[0].Cells).To(gomega.HaveLen(len(table.ColumnDefinitions)))
		gomega.Expect(table.ColumnDefinitions[0].Name).To(gomega.Equal("Name"))
		gomega.Expect(table.ResourceVersion).ToNot(gomega.BeEmpty())

		out := printTable(table)
		gomega.Expect(out).To(gomega.MatchRegexp("^NAME\\s"))
		framework.Logf("Table:\n%s", out)
	})

	/*
			    Release: v1.16
				Testname: API metadata HTTP return
				Description: Issue a HTTP request to the API.
		        HTTP request MUST return a HTTP status code of 406.
	*/
	framework.ConformanceIt("should return a 406 for a backend which does not implement metadata", func(ctx context.Context) {
		c := f.ClientSet

		table := &metav1beta1.Table{}
		sar := &authorizationv1.SelfSubjectAccessReview{
			Spec: authorizationv1.SelfSubjectAccessReviewSpec{
				NonResourceAttributes: &authorizationv1.NonResourceAttributes{
					Path: "/",
					Verb: "get",
				},
			},
		}
		err := c.AuthorizationV1().RESTClient().Post().Resource("selfsubjectaccessreviews").SetHeader("Accept", "application/json;as=Table;v=v1;g=meta.k8s.io").Body(sar).Do(ctx).Into(table)
		gomega.Expect(err).To(gomega.HaveOccurred(), "failed to return error when posting self subject access review: %+v, to a backend that does not implement metadata", sar)
		gomega.Expect(err.(apierrors.APIStatus)).To(gomega.HaveField("Status().Code", gomega.Equal(int32(406))))
	})
})

func printTable(table *metav1beta1.Table) string {
	buf := &bytes.Buffer{}
	tw := tabwriter.NewWriter(buf, 5, 8, 1, ' ', 0)
	printer := printers.NewTablePrinter(printers.PrintOptions{})
	err := printer.PrintObj(table, tw)
	framework.ExpectNoError(err, "failed to print table: %+v", table)
	tw.Flush()
	return buf.String()
}

func newTablePod(ns, podName string) *v1.Pod {
	port := 8080
	pod := e2epod.NewAgnhostPod(ns, podName, nil, nil, []v1.ContainerPort{{ContainerPort: int32(port)}}, "porter")
	pod.Spec.Containers[0].Env = []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}}
	pod.Spec.RestartPolicy = v1.RestartPolicyNever
	return pod
}
