/*
Copyright 2016 The Kubernetes Authors.

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

package e2e

import (
	"fmt"
	"time"

	. "github.com/onsi/ginkgo"
	"k8s.io/kubernetes/pkg/api"
	apierrs "k8s.io/kubernetes/pkg/api/errors"
	"k8s.io/kubernetes/pkg/api/resource"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	gcecloud "k8s.io/kubernetes/pkg/cloudprovider/providers/gce"
	"k8s.io/kubernetes/pkg/util/uuid"
	"k8s.io/kubernetes/test/e2e/framework"
)

var _ = framework.KubeDescribe("Snapshots", func() {
	// global vars for the It() tests below
	f := framework.NewDefaultFramework("snapshot")
	var c *client.Client
	var ns string
	var pv *api.PersistentVolume
	var pvc *api.PersistentVolumeClaim
	var err error
	var snapshots []string

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
	})

	AfterEach(func() {
		if c != nil && len(ns) > 0 { // still have client and namespace
			if pvc != nil && len(pvc.Name) > 0 {
				// Delete the PersistentVolumeClaim
				framework.Logf("AfterEach: PVC %v is non-nil, deleting claim", pvc.Name)
				err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
				if err != nil && !apierrs.IsNotFound(err) {
					framework.Logf("AfterEach: delete of PersistentVolumeClaim %v error: %v", pvc.Name, err)
				}
				pvc = nil
			}
			if pv != nil && len(pv.Name) > 0 {
				framework.Logf("AfterEach: PV %v is non-nil, deleting pv", pv.Name)
				err := c.PersistentVolumes().Delete(pv.Name)
				if err != nil && !apierrs.IsNotFound(err) {
					framework.Logf("AfterEach: delete of PersistentVolume %v error: %v", pv.Name, err)
				}
				pv = nil
			}
			if len(snapshots) > 0 {
				framework.Logf("AfterEach: Snapshots is not empty, deleting")
				for _, snapshotName := range snapshots {
					err := deleteSnapshot(snapshotName)
					if err != nil {
						framework.Logf("AfterEach: delete of snapshot %v error: %v", snapshotName, err)
					}
				}
				snapshots = snapshots[:0]
			}
		}
	})

	It("should request a snapshot on a snapshottable PV", func() {
		pv, pvc, err = createPVandPVC(c, ns, true /*snapshottable*/)
		if err != nil {
			framework.Failf("%v", err)
		}

		By("Validating the PV-PVC binding")
		pv, pvc, err = waitAndValidatePVandPVC(c, ns, pv, pvc)
		if err != nil {
			framework.Failf("%v", err)
		}

		//take a snapshot
		snapshotName := "snapshot-e2e-" + string(uuid.NewUUID())
		err = requestSnapshot(c, ns, pvc, snapshotName)
		if err != nil {
			framework.Failf("%v", err)
		}
		snapshots = append(snapshots, snapshotName)

		//verify the snapshot
		pvc, err = verifySnapshot(c, ns, pvc, snapshotName)
		if err != nil {
			framework.Failf("%v", err)
		}

		pv, pvc, snapshots, err = cleanUp(c, ns, pv, pvc, snapshots)
		if err != nil {
			framework.Failf("%v", err)
		}
	})
})

func cleanUp(c *client.Client, ns string, pv *api.PersistentVolume, pvc *api.PersistentVolumeClaim, snapshots []string) (*api.PersistentVolume, *api.PersistentVolumeClaim, []string, error) {
	// delete the PVC
	By("Deleting the PVC")
	pvc, err := deletePersistentVolumeClaim(c, ns, pvc)
	if err != nil {
		return pv, pvc, snapshots, err
	}

	// delete the pv
	By("Deleting the PV")
	if pv, err := deletePersistentVolume(c, pv); err != nil {
		return pv, pvc, snapshots, err
	}

	// delete snapshots
	// snapshots should be empty if the PV is not snapshottable
	By("Deleting snapshots")
	var snapshotsCopy []string
	for _, snapshotName := range snapshots {
		if err = deleteSnapshotWithRetry(snapshotName); err != nil {
			snapshotsCopy = append(snapshotsCopy, snapshotName)
		}
	}
	return pv, pvc, snapshotsCopy, nil
}

func deleteSnapshotWithRetry(snapshotName string) error {
	var err error
	for start := time.Now(); time.Since(start) < gcePDRetryTimeout; time.Sleep(gcePDRetryPollTime) {
		if err = deleteSnapshot(snapshotName); err != nil {
			framework.Logf("Couldn't delete snapshot. Sleeping 5 seconds (%v)", err)
			continue
		}
		framework.Logf("Successfully deleted snapshot: %q.", snapshotName)
		break
	}
	return err
}

func deleteSnapshot(snapshotName string) error {
	gceCloud, err := getGCECloud()
	if err != nil {
		return err
	}

	err = gceCloud.DeleteSnapshot(snapshotName)
	if err != nil {
		return err
	}
	return nil
}

func deletePersistentVolumeClaim(c *client.Client, ns string, pvc *api.PersistentVolumeClaim) (*api.PersistentVolumeClaim, error) {
	framework.Logf("Deleting PersistentVolumeClaim %v", pvc.Name)
	err := c.PersistentVolumeClaims(ns).Delete(pvc.Name)
	if err != nil {
		return pvc, fmt.Errorf("Delete of PVC %v failed: %v", pvc.Name, err)
	}

	// Check that the PVC is really deleted.
	pvc, err = c.PersistentVolumeClaims(ns).Get(pvc.Name)
	if err == nil {
		return pvc, fmt.Errorf("PVC %v deleted yet still exists", pvc.Name)
	}
	if !apierrs.IsNotFound(err) {
		return pvc, fmt.Errorf("Get on deleted PVC %v failed with error other than \"not found\": %v", pvc.Name, err)
	}

	return pvc, nil
}

func verifySnapshot(c *client.Client, ns string, pvc *api.PersistentVolumeClaim, snapshotName string) (*api.PersistentVolumeClaim, error) {
	By("Verifying the snapshot exists")

	err := framework.WaitForSnapshotCreated(c, ns, pvc.Name, snapshotName, 3*time.Second, 300*time.Second)
	if err != nil {
		return nil, fmt.Errorf("Snapshot could not be verified: %+v", err)
	}

	updatedPvc, err := c.PersistentVolumeClaims(ns).Get(pvc.Name)
	if err != nil {
		return nil, err
	}
	return updatedPvc, nil
}

func requestSnapshot(c *client.Client, ns string, pvc *api.PersistentVolumeClaim, snapshotName string) error {
	By("Adding the create snapshot annotation to the PVC")

	pvc.Annotations[api.AnnSnapshotCreate] = snapshotName
	_, err := c.PersistentVolumeClaims(ns).Update(pvc)
	if err != nil {
		return err
	}
	return nil
}

func createPVandPVC(c *client.Client, ns string, snapshottable bool) (*api.PersistentVolume, *api.PersistentVolumeClaim, error) {
	By("Creating a PV followed by a PVC")

	// make the pv and pvc definitions
	pv := makePV(snapshottable)
	pvc := makePVC(ns, pv.Name)

	// instantiate the pv
	pv, err := createPV(c, pv)
	if err != nil {
		return nil, nil, err
	}

	// instantiate the pvc
	pvc, err = createPVC(c, ns, pvc)
	if err != nil {
		return nil, nil, err
	}

	return pv, pvc, nil
}

func makePVC(ns, volumeName string) *api.PersistentVolumeClaim {
	By("Creating PVC")
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-",
			Namespace:    ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
			VolumeName: volumeName,
		},
	}
}

func makePV(snapshottable bool) *api.PersistentVolume {
	By("Creating PV")
	if snapshottable {
		gceDiskName, err := createGCEPDWithRetry()
		framework.ExpectNoError(err, "Error creating PD")
		return &api.PersistentVolume{
			ObjectMeta: api.ObjectMeta{
				GenerateName: "gcepd-",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("2Gi"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					GCEPersistentDisk: &api.GCEPersistentDiskVolumeSource{
						PDName: gceDiskName,
					},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		}
	} else {
		return &api.PersistentVolume{
			ObjectMeta: api.ObjectMeta{
				GenerateName: "hostpath-",
			},
			Spec: api.PersistentVolumeSpec{
				Capacity: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("2Gi"),
				},
				PersistentVolumeSource: api.PersistentVolumeSource{
					HostPath: &api.HostPathVolumeSource{Path: "/foo"},
				},
				AccessModes: []api.PersistentVolumeAccessMode{
					api.ReadWriteOnce,
				},
			},
		}
	}
}

func createGCEPDWithRetry() (string, error) {
	newDiskName := ""
	var err error
	for start := time.Now(); time.Since(start) < gcePDRetryTimeout; time.Sleep(gcePDRetryPollTime) {
		if newDiskName, err = createGCEPD(); err != nil {
			framework.Logf("Couldn't create a new PD. Sleeping 5 seconds (%v)", err)
			continue
		}
		framework.Logf("Successfully created a new PD: %q.", newDiskName)
		break
	}
	return newDiskName, err
}

func createGCEPD() (string, error) {
	pdName := fmt.Sprintf("%s-%s", framework.TestContext.Prefix, string(uuid.NewUUID()))

	gceCloud, err := getGCECloud()
	if err != nil {
		return "", err
	}

	tags := map[string]string{}
	err = gceCloud.CreateDisk(pdName, gcecloud.DiskTypeSSD, framework.TestContext.CloudConfig.Zone, 10 /* sizeGb */, tags)
	if err != nil {
		return "", err
	}
	return pdName, nil
}
