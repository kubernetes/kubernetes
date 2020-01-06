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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("ComponentStatuses", func() {
	f := framework.NewDefaultFramework("componentstatuses")

	ginkgo.It("should read and list ComponentStatuses to ensure health", func() {
		ginkgo.By("listing all ComponentStatuses")
		csList, _ := f.ClientSet.CoreV1().ComponentStatuses().List(metav1.ListOptions{})
		healthyCount := 0
		for _, csItem := range csList.Items {
			cs, _ := f.ClientSet.CoreV1().ComponentStatuses().Get(csItem.ObjectMeta.Name, metav1.GetOptions{})

			for _, condition := range cs.Conditions {
				if condition.Type == "Healthy" && condition.Status == "True" {
					healthyCount ++
				}
			}
		}

		framework.ExpectEqual(healthyCount, len(csList.Items), "Components are not all healthy")
	})

	ginkgo.It("should ensure select Components exist", func() {
		requiredComponents := map[string]bool{
			"scheduler": true,
			"controller-manager": true,
		}
		requiredComponentFoundCount := 0

		ginkgo.By("listing all ComponentStatuses")
		csList, _ := f.ClientSet.CoreV1().ComponentStatuses().List(metav1.ListOptions{})
		for _, csItem := range csList.Items {
			if requiredComponents[csItem.ObjectMeta.Name] == true {
				requiredComponentFoundCount ++
			}
		}

		framework.ExpectEqual(requiredComponentFoundCount, len(requiredComponents), "Components are not all healthy")
	})
})

