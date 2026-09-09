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
	"time"

	apiserverinternalv1alpha1 "k8s.io/api/apiserverinternal/v1alpha1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo/v2"
)

const (
	svName     = "storageversion.e2e.test.foos"
	idNonExist = "id-non-exist"
)

// This test requires that --feature-gates=APIServerIdentity=true,StorageVersionAPI=true be set on the apiserver and the controller manager
var _ = SIGDescribe("StorageVersion resources", feature.StorageVersionAPI, func() {
	f := framework.NewDefaultFramework("storage-version")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("storage version with non-existing id should be GC'ed", func(ctx context.Context) {
		client := f.ClientSet
		sv := &apiserverinternalv1alpha1.StorageVersion{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: svName,
			},
		}
		createdSV, err := client.InternalV1alpha1().StorageVersions().Create(ctx, sv, metav1.CreateOptions{})
		framework.ExpectNoError(err, "creating storage version")

		// update the created sv with server storage version
		version := "v1"
		createdSV.Status = apiserverinternalv1alpha1.StorageVersionStatus{
			StorageVersions: []apiserverinternalv1alpha1.ServerStorageVersion{
				{
					APIServerID:       idNonExist,
					EncodingVersion:   version,
					DecodableVersions: []string{version},
				},
			},
			CommonEncodingVersion: &version,
		}
		_, err = client.InternalV1alpha1().StorageVersions().UpdateStatus(
			ctx, createdSV, metav1.UpdateOptions{})
		framework.ExpectNoError(err, "updating storage version")

		// wait for sv to be GC'ed
		framework.Logf("Waiting for storage version %v to be garbage collected", createdSV.Name)
		err = wait.PollImmediate(100*time.Millisecond, wait.ForeverTestTimeout, func() (bool, error) {
			_, err := client.InternalV1alpha1().StorageVersions().Get(
				ctx, createdSV.Name, metav1.GetOptions{})
			if apierrors.IsNotFound(err) {
				return true, nil
			}
			if err != nil {
				return false, err
			}
			framework.Logf("The storage version %v hasn't been garbage collected yet. Retrying", createdSV.Name)
			return false, nil
		})
		framework.ExpectNoError(err, "garbage-collecting storage version")
	})
})
