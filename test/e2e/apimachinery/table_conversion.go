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

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/client-go/util/workqueue"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var serverPrintVersion = utilversion.MustParseSemantic("v1.10.0")

var _ = SIGDescribe("Servers with support for Table transformation", func() {
	f := framework.NewDefaultFramework("tables")

	ginkgo.BeforeEach(func() {
		framework.SkipUnlessServerVersionGTE(serverPrintVersion, f.ClientSet.Discovery())
	})

	ginkgo.It("should return pod details", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "pod-1"
		e2elog.Logf("Creating pod %s", podName)

		_, err := c.CoreV1().Pods(ns).Create(newTablePod(podName))
		framework.ExpectNoError(err, "failed to create pod %s in namespace: %s", podName, ns)

		table := &metav1beta1.Table{}
		err = c.CoreV1().RESTClient().Get().Resource("pods").Namespace(ns).Name(podName).SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do().Into(table)
		framework.ExpectNoError(err, "failed to get pod %s in Table form in namespace: %s", podName, ns)
		e2elog.Logf("Table: %#v", table)

		gomega.Expect(len(table.ColumnDefinitions)).To(gomega.BeNumerically(">", 2))
		framework.ExpectEqual(len(table.Rows), 1)
		framework.ExpectEqual(len(table.Rows[0].Cells), len(table.ColumnDefinitions))
		framework.ExpectEqual(table.ColumnDefinitions[0].Name, "Name")
		framework.ExpectEqual(table.Rows[0].Cells[0], podName)

		out := printTable(table)
		gomega.Expect(out).To(gomega.MatchRegexp("^NAME\\s"))
		gomega.Expect(out).To(gomega.MatchRegexp("\npod-1\\s"))
		e2elog.Logf("Table:\n%s", out)
	})

	ginkgo.It("should return chunks of table results for list calls", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		ginkgo.By("creating a large number of resources")
		workqueue.ParallelizeUntil(context.TODO(), 5, 20, func(i int) {
			for tries := 3; tries >= 0; tries-- {
				_, err := client.Create(&v1.PodTemplate{
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
				})
				if err == nil {
					return
				}
				e2elog.Logf("Got an error creating template %d: %v", i, err)
			}
			ginkgo.Fail("Unable to create template %d, exiting", i)
		})

		pagedTable := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Limit: 2}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do().Into(pagedTable)
		framework.ExpectNoError(err, "failed to get pod templates in Table form in namespace: %s", ns)
		framework.ExpectEqual(len(pagedTable.Rows), 2)
		framework.ExpectNotEqual(pagedTable.ResourceVersion, "")
		framework.ExpectNotEqual(pagedTable.SelfLink, "")
		framework.ExpectNotEqual(pagedTable.Continue, "")
		framework.ExpectEqual(pagedTable.Rows[0].Cells[0], "template-0000")
		framework.ExpectEqual(pagedTable.Rows[1].Cells[0], "template-0001")

		err = c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Continue: pagedTable.Continue}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do().Into(pagedTable)
		framework.ExpectNoError(err, "failed to get pod templates in Table form in namespace: %s", ns)
		gomega.Expect(len(pagedTable.Rows)).To(gomega.BeNumerically(">", 0))
		framework.ExpectEqual(pagedTable.Rows[0].Cells[0], "template-0002")
	})

	ginkgo.It("should return generic metadata details across all namespaces for nodes", func() {
		c := f.ClientSet

		table := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Resource("nodes").SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do().Into(table)
		framework.ExpectNoError(err, "failed to get nodes in Table form across all namespaces")
		e2elog.Logf("Table: %#v", table)

		gomega.Expect(len(table.ColumnDefinitions)).To(gomega.BeNumerically(">=", 2))
		gomega.Expect(len(table.Rows)).To(gomega.BeNumerically(">=", 1))
		framework.ExpectEqual(len(table.Rows[0].Cells), len(table.ColumnDefinitions))
		framework.ExpectEqual(table.ColumnDefinitions[0].Name, "Name")
		framework.ExpectNotEqual(table.ResourceVersion, "")
		framework.ExpectNotEqual(table.SelfLink, "")

		out := printTable(table)
		gomega.Expect(out).To(gomega.MatchRegexp("^NAME\\s"))
		e2elog.Logf("Table:\n%s", out)
	})

	/*
			    Release : v1.16
				Testname: API metadata HTTP return
				Description: Issue a HTTP request to the API.
		        HTTP request MUST return a HTTP status code of 406.
	*/
	framework.ConformanceIt("should return a 406 for a backend which does not implement metadata", func() {
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
		err := c.AuthorizationV1().RESTClient().Post().Resource("selfsubjectaccessreviews").SetHeader("Accept", "application/json;as=Table;v=v1;g=meta.k8s.io").Body(sar).Do().Into(table)
		framework.ExpectError(err, "failed to return error when posting self subject access review: %+v, to a backend that does not implement metadata", sar)
		framework.ExpectEqual(err.(errors.APIStatus).Status().Code, int32(406))
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

func newTablePod(podName string) *v1.Pod {
	containerName := fmt.Sprintf("%s-container", podName)
	port := 8080
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:  containerName,
					Image: imageutils.GetE2EImage(imageutils.Agnhost),
					Args:  []string{"porter"},
					Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return pod
}
