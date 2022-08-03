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

package vsphere

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	admissionapi "k8s.io/pod-security-admission/api"

	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	invalidDatastore = "invalidDatastore"
	datastoreSCName  = "datastoresc"
)

/*
	Test to verify datastore specified in storage-class is being honored while volume creation.

	Steps
	1. Create StorageClass with invalid datastore.
	2. Create PVC which uses the StorageClass created in step 1.
	3. Expect the PVC to fail.
	4. Verify the error returned on PVC failure is the correct.
*/

var _ = utils.SIGDescribe("Volume Provisioning on Datastore [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("volume-datastore")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var (
		client                     clientset.Interface
		namespace                  string
		scParameters               map[string]string
		vSphereCSIMigrationEnabled bool
	)
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		scParameters = make(map[string]string)
		_, err := e2enode.GetRandomReadySchedulableNode(f.ClientSet)
		framework.ExpectNoError(err)
		vSphereCSIMigrationEnabled = GetAndExpectBoolEnvVar(VSphereCSIMigrationEnabled)
	})

	ginkgo.It("verify dynamically provisioned pv using storageclass fails on an invalid datastore", func() {
		ginkgo.By("Invoking Test for invalid datastore")
		scParameters[Datastore] = invalidDatastore
		scParameters[DiskFormat] = ThinDisk
		err := invokeInvalidDatastoreTestNeg(client, namespace, scParameters)
		framework.ExpectError(err)
		var errorMsg string
		if !vSphereCSIMigrationEnabled {
			errorMsg = `Failed to provision volume with StorageClass \"` + datastoreSCName + `\": Datastore '` + invalidDatastore + `' not found`
		} else {
			errorMsg = `failed to find datastoreURL for datastore name: \"` + invalidDatastore + `\"`
		}
		if !strings.Contains(err.Error(), errorMsg) {
			framework.ExpectNoError(err, errorMsg)
		}
	})
})

func invokeInvalidDatastoreTestNeg(client clientset.Interface, namespace string, scParameters map[string]string) error {
	ginkgo.By("Creating Storage Class With Invalid Datastore")
	storageclass, err := client.StorageV1().StorageClasses().Create(context.TODO(), getVSphereStorageClassSpec(datastoreSCName, scParameters, nil, ""), metav1.CreateOptions{})
	framework.ExpectNoError(err, fmt.Sprintf("Failed to create storage class with err: %v", err))
	defer client.StorageV1().StorageClasses().Delete(context.TODO(), storageclass.Name, metav1.DeleteOptions{})

	ginkgo.By("Creating PVC using the Storage Class")
	pvclaim, err := e2epv.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, "2Gi", storageclass))
	framework.ExpectNoError(err)
	defer e2epv.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

	ginkgo.By("Expect claim to fail provisioning volume")
	err = e2epv.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, pvclaim.Namespace, pvclaim.Name, framework.Poll, 2*time.Minute)
	framework.ExpectError(err)

	eventList, err := client.CoreV1().Events(pvclaim.Namespace).List(context.TODO(), metav1.ListOptions{})
	framework.ExpectNoError(err)

	var eventErrorMessages string
	for _, event := range eventList.Items {
		if event.Type != v1.EventTypeNormal {
			eventErrorMessages = eventErrorMessages + event.Message + ";"
		}
	}
	return fmt.Errorf("event messages: %+q", eventErrorMessages)
}
