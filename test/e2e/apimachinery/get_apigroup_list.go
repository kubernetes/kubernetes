/*
Copyright 2019 The Kubernetes Authors.

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
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("get-apigroup-list", func() {
	f := framework.NewDefaultFramework("get-apigroup-list")
	ginkgo.It("should locate PreferredVersion for each APIGroup", func() {

		// TEST BEGINS HERE
		ginkgo.By("[status] begin")

		// get list of APIGroup endpoints
		list := &metav1.APIGroupList{}
		err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/").Do(context.TODO()).Into(list)

		framework.ExpectNoError(err, "Failed to find /apis/")

		for _, group := range list.Groups {
			framework.Logf("Checking APIGroup: %v", group.Name)

			// hit APIGroup endpoint
			checkGroup := &metav1.APIGroup{}
			apiPath := "/apis/" + group.Name + "/"
			err = f.ClientSet.Discovery().RESTClient().Get().AbsPath(apiPath).Do(context.TODO()).Into(checkGroup)

			framework.ExpectNoError(err, "Fail to access: %s", apiPath)

			// get PreferredVersion for endpoint
			framework.Logf("PreferredVersion: %v", checkGroup.PreferredVersion)
		}
	})
})
