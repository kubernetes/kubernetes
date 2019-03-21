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
	"fmt"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

const (
	DiskSizeSCName = "disksizesc"
)

/*
	Test to verify disk size specified in PVC is being honored while volume creation.

	Steps
	1. Create StorageClass.
	2. Create PVC with invalid disk size which uses the StorageClass created in step 1.
	3. Expect the PVC to fail.
	4. Verify the error returned on PVC failure is the correct.
*/

var _ = utils.SIGDescribe("Volume Disk Size [Feature:vsphere]", func() {
	f := framework.NewDefaultFramework("volume-disksize")
	var (
		client       clientset.Interface
		namespace    string
		scParameters map[string]string
		datastore    string
	)
	BeforeEach(func() {
		framework.SkipUnlessProviderIs("vsphere")
		Bootstrap(f)
		client = f.ClientSet
		namespace = f.Namespace.Name
		scParameters = make(map[string]string)
		datastore = GetAndExpectStringEnvVar(StorageClassDatastoreName)
	})

	It("verify dynamically provisioned pv using storageclass with an invalid disk size fails", func() {
		By("Invoking Test for invalid disk size")
		scParameters[Datastore] = datastore
		scParameters[DiskFormat] = ThinDisk
		diskSize := "1"
		err := invokeInvalidDiskSizeTestNeg(client, namespace, scParameters, diskSize)
		Expect(err).To(HaveOccurred())
		errorMsg := `Failed to provision volume with StorageClass \"` + DiskSizeSCName + `\": A specified parameter was not correct`
		if !strings.Contains(err.Error(), errorMsg) {
			Expect(err).NotTo(HaveOccurred(), errorMsg)
		}
	})
})

func invokeInvalidDiskSizeTestNeg(client clientset.Interface, namespace string, scParameters map[string]string, diskSize string) error {
	By("Creating Storage Class With invalid disk size")
	storageclass, err := client.StorageV1().StorageClasses().Create(getVSphereStorageClassSpec(DiskSizeSCName, scParameters, nil))
	Expect(err).NotTo(HaveOccurred(), fmt.Sprintf("Failed to create storage class with err: %v", err))
	defer client.StorageV1().StorageClasses().Delete(storageclass.Name, nil)

	By("Creating PVC using the Storage Class")
	pvclaim, err := framework.CreatePVC(client, namespace, getVSphereClaimSpecWithStorageClass(namespace, diskSize, storageclass))
	Expect(err).NotTo(HaveOccurred())
	defer framework.DeletePersistentVolumeClaim(client, pvclaim.Name, namespace)

	By("Expect claim to fail provisioning volume")
	err = framework.WaitForPersistentVolumeClaimPhase(v1.ClaimBound, client, pvclaim.Namespace, pvclaim.Name, framework.Poll, 2*time.Minute)
	Expect(err).To(HaveOccurred())

	eventList, err := client.CoreV1().Events(pvclaim.Namespace).List(metav1.ListOptions{})
	return fmt.Errorf("Failure message: %+q", eventList.Items[0].Message)
}
