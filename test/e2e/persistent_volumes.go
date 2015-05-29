/*
Copyright 2015 The Kubernetes Authors All rights reserved.

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

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/resource"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
	"time"
)

var _ = Describe("persistentVolumes", func() {
	//		f := NewFramework("pv")

	var c *client.Client
	var ns string

	BeforeEach(func() {
		var err error
		c, err = loadClient()
		Expect(err).NotTo(HaveOccurred())
		ns_, err := createTestingNS("pv", c)
		ns = ns_.Name
		Expect(err).NotTo(HaveOccurred())
	})

	AfterEach(func() {
		By(fmt.Sprintf("Destroying namespace for this suite %v", ns))
		if err := c.Namespaces().Delete(ns); err != nil {
			Failf("Couldn't delete ns %s", err)
		}
	})

	It("PersistentVolume", func() {
		config := VolumeTestConfig{
			namespace:   ns,
			prefix:      "nfs",
			serverImage: "gcr.io/google_containers/volume-nfs",
			serverPorts: []int{2049},
		}

		defer func() {
			volumeTestCleanup(c, config)
		}()

		pod := startVolumeServer(c, config)
		serverIP := pod.Status.PodIP
		Logf("NFS server IP address: %v", serverIP)

		pv := makePersistentVolume(serverIP)
		pvc := makePersistentVolumeClaim(ns)

		Logf("Creating PersistentVolume using NFS")
		pv, err := c.PersistentVolumes().Create(pv)
		Expect(err).NotTo(HaveOccurred())

		Logf("Creating PersistentVolumeClaim")
		pvc, err = c.PersistentVolumeClaims(ns).Create(pvc)
		Expect(err).NotTo(HaveOccurred())

		// allow the binder a chance to catch up
		time.Sleep(20 * time.Second)

		pv, err = c.PersistentVolumes().Get(pv.Name)
		Expect(err).NotTo(HaveOccurred())
		if pv.Spec.ClaimRef == nil {
			Failf("Expected PersistentVolume to be bound, but got nil ClaimRef: %+v", pv)
		}

		Logf("Deleting PersistentVolumeClaim to trigger PV Recycling")
		err = c.PersistentVolumeClaims(ns).Delete(pvc.Name)
		Expect(err).NotTo(HaveOccurred())

		// allow the recycler a chance to catch up
		time.Sleep(120 * time.Second)

		pv, err = c.PersistentVolumes().Get(pv.Name)
		Expect(err).NotTo(HaveOccurred())
		if pv.Spec.ClaimRef != nil {
			Failf("Expected PersistentVolume to be unbound, but found non-nil ClaimRef: %+v", pv)
		}

		// Now check that index.html from the NFS server was really removed
		checkpod := makeCheckPod(ns, serverIP)
		testContainerOutputInNamespace("the volume was scrubbed", c, checkpod, []string{"index.html does not exist"}, ns)

	})
})

func makePersistentVolume(serverIP string) *api.PersistentVolume {
	return &api.PersistentVolume{
		ObjectMeta: api.ObjectMeta{
			Name: "nfs-" + string(util.NewUUID()),
		},
		Spec: api.PersistentVolumeSpec{
			PersistentVolumeReclaimPolicy: api.PersistentVolumeReclaimRecycle,
			Capacity: api.ResourceList{
				api.ResourceName(api.ResourceStorage): resource.MustParse("2Gi"),
			},
			PersistentVolumeSource: api.PersistentVolumeSource{
				NFS: &api.NFSVolumeSource{
					Server:   serverIP,
					Path:     "/",
					ReadOnly: false,
				},
			},
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
		},
	}
}

func makePersistentVolumeClaim(ns string) *api.PersistentVolumeClaim {
	return &api.PersistentVolumeClaim{
		ObjectMeta: api.ObjectMeta{
			Name:      "pvc-" + string(util.NewUUID()),
			Namespace: ns,
		},
		Spec: api.PersistentVolumeClaimSpec{
			AccessModes: []api.PersistentVolumeAccessMode{
				api.ReadWriteOnce,
				api.ReadOnlyMany,
				api.ReadWriteMany,
			},
			Resources: api.ResourceRequirements{
				Requests: api.ResourceList{
					api.ResourceName(api.ResourceStorage): resource.MustParse("1Gi"),
				},
			},
		},
	}
}

func makeCheckPod(ns string, nfsserver string) *api.Pod {
	// Prepare pod that mounts the NFS volume again and
	// checks that /mnt/index.html was scrubbed there
	return &api.Pod{
		TypeMeta: api.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1beta3",
		},
		ObjectMeta: api.ObjectMeta{
			Name: "checker-" + string(util.NewUUID()),
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "checker-" + string(util.NewUUID()),
					Image:   "busybox",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", "test -e /mnt/index.html || echo 'index.html does not exist'"},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "nfs-volume",
							MountPath: "/mnt",
						},
					},
				},
			},
			Volumes: []api.Volume{
				{
					Name: "nfs-volume",
					VolumeSource: api.VolumeSource{
						NFS: &api.NFSVolumeSource{
							Server: nfsserver,
							Path:   "/",
						},
					},
				},
			},
		},
	}

}
