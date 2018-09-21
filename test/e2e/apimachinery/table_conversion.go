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
	"fmt"
	"text/tabwriter"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	authorizationv1 "k8s.io/api/authorization/v1"
	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	metav1beta1 "k8s.io/apimachinery/pkg/apis/meta/v1beta1"
	"k8s.io/client-go/util/workqueue"

	utilversion "k8s.io/apimachinery/pkg/util/version"
	"k8s.io/kubernetes/pkg/printers"
	"k8s.io/kubernetes/test/e2e/framework"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

var serverPrintVersion = utilversion.MustParseSemantic("v1.10.0")

var _ = SIGDescribe("Servers with support for Table transformation", func() {
	f := framework.NewDefaultFramework("tables")

	BeforeEach(func() {
		framework.SkipUnlessServerVersionGTE(serverPrintVersion, f.ClientSet.Discovery())
	})

	It("should return pod details", func() {
		ns := f.Namespace.Name
		c := f.ClientSet

		podName := "pod-1"
		framework.Logf("Creating pod %s", podName)

		_, err := c.CoreV1().Pods(ns).Create(newTablePod(podName))
		Expect(err).NotTo(HaveOccurred())

		table := &metav1beta1.Table{}
		err = c.CoreV1().RESTClient().Get().Resource("pods").Namespace(ns).Name(podName).SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do().Into(table)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Table: %#v", table)

		Expect(len(table.ColumnDefinitions)).To(BeNumerically(">", 2))
		Expect(len(table.Rows)).To(Equal(1))
		Expect(len(table.Rows[0].Cells)).To(Equal(len(table.ColumnDefinitions)))
		Expect(table.ColumnDefinitions[0].Name).To(Equal("Name"))
		Expect(table.Rows[0].Cells[0]).To(Equal(podName))

		out := printTable(table)
		Expect(out).To(MatchRegexp("^NAME\\s"))
		Expect(out).To(MatchRegexp("\npod-1\\s"))
		framework.Logf("Table:\n%s", out)
	})

	It("should return chunks of table results for list calls", func() {
		ns := f.Namespace.Name
		c := f.ClientSet
		client := c.CoreV1().PodTemplates(ns)

		By("creating a large number of resources")
		workqueue.Parallelize(5, 20, func(i int) {
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
				framework.Logf("Got an error creating template %d: %v", i, err)
			}
			Fail("Unable to create template %d, exiting", i)
		})

		pagedTable := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Limit: 2}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do().Into(pagedTable)
		Expect(err).NotTo(HaveOccurred())
		// TODO: kops PR job is still using etcd2, which prevents this feature from working. Remove this check when kops is upgraded to etcd3
		if len(pagedTable.Rows) > 2 {
			framework.Skipf("ERROR: This cluster does not support chunking, which means it is running etcd2 and not supported.")
		}
		Expect(len(pagedTable.Rows)).To(Equal(2))
		Expect(pagedTable.ResourceVersion).ToNot(Equal(""))
		Expect(pagedTable.SelfLink).ToNot(Equal(""))
		Expect(pagedTable.Continue).ToNot(Equal(""))
		Expect(pagedTable.Rows[0].Cells[0]).To(Equal("template-0000"))
		Expect(pagedTable.Rows[1].Cells[0]).To(Equal("template-0001"))

		err = c.CoreV1().RESTClient().Get().Namespace(ns).Resource("podtemplates").
			VersionedParams(&metav1.ListOptions{Continue: pagedTable.Continue}, metav1.ParameterCodec).
			SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").
			Do().Into(pagedTable)
		Expect(err).NotTo(HaveOccurred())
		Expect(len(pagedTable.Rows)).To(BeNumerically(">", 0))
		Expect(pagedTable.Rows[0].Cells[0]).To(Equal("template-0002"))
	})

	It("should return generic metadata details across all namespaces for nodes", func() {
		c := f.ClientSet

		table := &metav1beta1.Table{}
		err := c.CoreV1().RESTClient().Get().Resource("nodes").SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Do().Into(table)
		Expect(err).NotTo(HaveOccurred())
		framework.Logf("Table: %#v", table)

		Expect(len(table.ColumnDefinitions)).To(BeNumerically(">=", 2))
		Expect(len(table.Rows)).To(BeNumerically(">=", 1))
		Expect(len(table.Rows[0].Cells)).To(Equal(len(table.ColumnDefinitions)))
		Expect(table.ColumnDefinitions[0].Name).To(Equal("Name"))
		Expect(table.ResourceVersion).ToNot(Equal(""))
		Expect(table.SelfLink).ToNot(Equal(""))

		out := printTable(table)
		Expect(out).To(MatchRegexp("^NAME\\s"))
		framework.Logf("Table:\n%s", out)
	})

	It("should return a 406 for a backend which does not implement metadata", func() {
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
		err := c.AuthorizationV1().RESTClient().Post().Resource("selfsubjectaccessreviews").SetHeader("Accept", "application/json;as=Table;v=v1beta1;g=meta.k8s.io").Body(sar).Do().Into(table)
		Expect(err).To(HaveOccurred())
		Expect(err.(errors.APIStatus).Status().Code).To(Equal(int32(406)))
	})
})

func printTable(table *metav1beta1.Table) string {
	buf := &bytes.Buffer{}
	tw := tabwriter.NewWriter(buf, 5, 8, 1, ' ', 0)
	err := printers.PrintTable(table, tw, printers.PrintOptions{})
	Expect(err).NotTo(HaveOccurred())
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
					Image: imageutils.GetE2EImage(imageutils.Porter),
					Env:   []v1.EnvVar{{Name: fmt.Sprintf("SERVE_PORT_%d", port), Value: "foo"}},
					Ports: []v1.ContainerPort{{ContainerPort: int32(port)}},
				},
			},
			RestartPolicy: v1.RestartPolicyNever,
		},
	}
	return pod
}
