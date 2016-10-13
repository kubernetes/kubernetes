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
	"strings"
	"time"

	"github.com/aws/aws-sdk-go/aws/session"
	"github.com/aws/aws-sdk-go/service/ec2"

	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/resource"
	"k8s.io/kubernetes/pkg/api/unversioned"
	"k8s.io/kubernetes/pkg/apis/storage"
	client "k8s.io/kubernetes/pkg/client/unversioned"
	"k8s.io/kubernetes/test/e2e/framework"

	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

type storageClassTest struct {
	name           string
	cloudProviders []string
	provisioner    string
	parameters     map[string]string
	claimSize      string
	expectedSize   string
	pvCheck        func(volume *api.PersistentVolume) error
}

func testDynamicProvisioning(t storageClassTest, client *client.Client, claim *api.PersistentVolumeClaim, class *storage.StorageClass) {
	if class != nil {
		By("creating a StorageClass " + class.Name)
		class, err := client.Storage().StorageClasses().Create(class)
		defer func() {
			framework.Logf("deleting storage class %s", class.Name)
			client.Storage().StorageClasses().Delete(class.Name)
		}()
		Expect(err).NotTo(HaveOccurred())
	}

	By("creating a claim")
	claim, err := client.PersistentVolumeClaims(claim.Namespace).Create(claim)
	defer func() {
		framework.Logf("deleting claim %s/%s", claim.Namespace, claim.Name)
		client.PersistentVolumeClaims(claim.Namespace).Delete(claim.Name)
	}()
	Expect(err).NotTo(HaveOccurred())
	err = framework.WaitForPersistentVolumeClaimPhase(api.ClaimBound, client, claim.Namespace, claim.Name, framework.Poll, framework.ClaimProvisionTimeout)
	Expect(err).NotTo(HaveOccurred())

	By("checking the claim")
	// Get new copy of the claim
	claim, err = client.PersistentVolumeClaims(claim.Namespace).Get(claim.Name)
	Expect(err).NotTo(HaveOccurred())

	// Get the bound PV
	pv, err := client.PersistentVolumes().Get(claim.Spec.VolumeName)
	Expect(err).NotTo(HaveOccurred())

	// Check sizes
	expectedCapacity := resource.MustParse(t.expectedSize)
	pvCapacity := pv.Spec.Capacity[api.ResourceName(api.ResourceStorage)]
	Expect(pvCapacity.Value()).To(Equal(expectedCapacity.Value()))

	requestedCapacity := resource.MustParse(t.claimSize)
	claimCapacity := claim.Spec.Resources.Requests[api.ResourceName(api.ResourceStorage)]
	Expect(claimCapacity.Value()).To(Equal(requestedCapacity.Value()))

	// Check PV properties
	By("checking the PV")
	Expect(pv.Spec.PersistentVolumeReclaimPolicy).To(Equal(api.PersistentVolumeReclaimDelete))
	expectedAccessModes := []api.PersistentVolumeAccessMode{api.ReadWriteOnce}
	Expect(pv.Spec.AccessModes).To(Equal(expectedAccessModes))
	Expect(pv.Spec.ClaimRef.Name).To(Equal(claim.ObjectMeta.Name))
	Expect(pv.Spec.ClaimRef.Namespace).To(Equal(claim.ObjectMeta.Namespace))

	// Run the checker
	if t.pvCheck != nil {
		err = t.pvCheck(pv)
		Expect(err).NotTo(HaveOccurred())
	}

	// We start two pods:
	// - The first writes 'hello word' to the /mnt/test (= the volume).
	// - The second one runs grep 'hello world' on /mnt/test.
	// If both succeed, Kubernetes actually allocated something that is
	// persistent across pods.
	By("checking the created volume is writable")
	runInPodWithVolume(client, claim.Namespace, claim.Name, "echo 'hello world' > /mnt/test/data")

	By("checking the created volume is readable and retains data")
	runInPodWithVolume(client, claim.Namespace, claim.Name, "grep 'hello world' /mnt/test/data")

	// Ugly hack: if we delete the AWS/GCE/OpenStack volume here, it will
	// probably collide with destruction of the pods above - the pods
	// still have the volume attached (kubelet is slow...) and deletion
	// of attached volume is not allowed by AWS/GCE/OpenStack.
	// Kubernetes *will* retry deletion several times in
	// pvclaimbinder-sync-period.
	// So, technically, this sleep is not needed. On the other hand,
	// the sync perion is 10 minutes and we really don't want to wait
	// 10 minutes here. There is no way how to see if kubelet is
	// finished with cleaning volumes. A small sleep here actually
	// speeds up the test!
	// Three minutes should be enough to clean up the pods properly.
	// We've seen GCE PD detach to take more than 1 minute.
	By("Sleeping to let kubelet destroy all pods")
	time.Sleep(3 * time.Minute)

	By("deleting the claim")
	framework.ExpectNoError(client.PersistentVolumeClaims(claim.Namespace).Delete(claim.Name))

	// Wait for the PV to get deleted too.
	framework.ExpectNoError(framework.WaitForPersistentVolumeDeleted(client, pv.Name, 5*time.Second, 20*time.Minute))
}

// checkAWSEBS checks properties of an AWS EBS. Test framework does not
// instantiate full AWS provider, therefore we need use ec2 API directly.
func checkAWSEBS(volume *api.PersistentVolume, volumeType string, encrypted bool) error {
	diskName := volume.Spec.AWSElasticBlockStore.VolumeID

	client := ec2.New(session.New())
	tokens := strings.Split(diskName, "/")
	volumeID := tokens[len(tokens)-1]

	request := &ec2.DescribeVolumesInput{
		VolumeIds: []*string{&volumeID},
	}
	info, err := client.DescribeVolumes(request)
	if err != nil {
		return fmt.Errorf("error querying ec2 for volume %q: %v", volumeID, err)
	}
	if len(info.Volumes) == 0 {
		return fmt.Errorf("no volumes found for volume %q", volumeID)
	}
	if len(info.Volumes) > 1 {
		return fmt.Errorf("multiple volumes found for volume %q", volumeID)
	}

	awsVolume := info.Volumes[0]
	if awsVolume.VolumeType == nil {
		return fmt.Errorf("expected volume type %q, got nil", volumeType)
	}
	if *awsVolume.VolumeType != volumeType {
		return fmt.Errorf("expected volume type %q, got %q", volumeType, *awsVolume.VolumeType)
	}
	if encrypted && awsVolume.Encrypted == nil {
		return fmt.Errorf("expected encrypted volume, got no encryption")
	}
	if encrypted && !*awsVolume.Encrypted {
		return fmt.Errorf("expected encrypted volume, got %v", *awsVolume.Encrypted)
	}
	return nil
}

func checkGCEPD(volume *api.PersistentVolume, volumeType string) error {
	cloud, err := getGCECloud()
	if err != nil {
		return err
	}
	diskName := volume.Spec.GCEPersistentDisk.PDName
	return cloud.TestDisk(diskName, volumeType)
}

var _ = framework.KubeDescribe("Dynamic provisioning", func() {
	f := framework.NewDefaultFramework("volume-provisioning")

	// filled in BeforeEach
	var c *client.Client
	var ns string

	BeforeEach(func() {
		c = f.Client
		ns = f.Namespace.Name
	})

	framework.KubeDescribe("DynamicProvisioner", func() {
		// This test checks that dynamic provisioning can provision a volume
		// that can be used to persist data among pods.

		tests := []storageClassTest{
			{
				"should provision SSD PD on GCE/GKE",
				[]string{"gce", "gke"},
				"kubernetes.io/gce-pd",
				map[string]string{
					"type": "pd-ssd",
					// Check that GCE can parse "zone" parameter, however
					// we can't create PDs in different than default zone
					// as we don't know if we're running with Multizone=true
					"zone": framework.TestContext.CloudConfig.Zone,
				},
				"1.5Gi",
				"2Gi",
				func(volume *api.PersistentVolume) error {
					return checkGCEPD(volume, "pd-ssd")
				},
			},
			{
				"should provision HDD PD on GCE/GKE",
				[]string{"gce", "gke"},
				"kubernetes.io/gce-pd",
				map[string]string{
					"type": "pd-standard",
				},
				"1.5Gi",
				"2Gi",
				func(volume *api.PersistentVolume) error {
					return checkGCEPD(volume, "pd-standard")
				},
			},
			// AWS
			{
				"should provision gp2 EBS on AWS",
				[]string{"aws"},
				"kubernetes.io/aws-ebs",
				map[string]string{
					"type": "gp2",
					// Check that AWS can parse "zone" parameter, however
					// we can't create PDs in different than default zone
					// as we don't know zone names
					"zone": framework.TestContext.CloudConfig.Zone,
				},
				"1.5Gi",
				"2Gi",
				func(volume *api.PersistentVolume) error {
					return checkAWSEBS(volume, "gp2", false)
				},
			},
			{
				"should provision io1 EBS on AWS",
				[]string{"aws"},
				"kubernetes.io/aws-ebs",
				map[string]string{
					"type":      "io1",
					"iopsPerGB": "50",
				},
				"3.5Gi",
				"4Gi", // 4 GiB is minimum for io1
				func(volume *api.PersistentVolume) error {
					return checkAWSEBS(volume, "io1", false)
				},
			},
			{
				"should provision sc1 EBS on AWS",
				[]string{"aws"},
				"kubernetes.io/aws-ebs",
				map[string]string{
					"type": "sc1",
				},
				"500Gi", // minimum for sc1
				"500Gi",
				func(volume *api.PersistentVolume) error {
					return checkAWSEBS(volume, "sc1", false)
				},
			},
			{
				"should provision st1 EBS on AWS",
				[]string{"aws"},
				"kubernetes.io/aws-ebs",
				map[string]string{
					"type": "st1",
				},
				"500Gi", // minimum for st1
				"500Gi",
				func(volume *api.PersistentVolume) error {
					return checkAWSEBS(volume, "st1", false)
				},
			},
			{
				"should provision encrypted EBS on AWS",
				[]string{"aws"},
				"kubernetes.io/aws-ebs",
				map[string]string{
					"encrypted": "true",
				},
				"1Gi",
				"1Gi",
				func(volume *api.PersistentVolume) error {
					return checkAWSEBS(volume, "gp2", true)
				},
			},
			// OpenStack generic tests (works on all OpenStack deployments)
			{
				"should provision generic Cinder volume on OpenStack",
				[]string{"openstack"},
				"kubernetes.io/cinder",
				map[string]string{},
				"1.5Gi",
				"2Gi",
				nil, // there is currently nothing to check on OpenStack
			},
			{
				"should provision Cinder volume with empty volume type and zone on OpenStack",
				[]string{"openstack"},
				"kubernetes.io/cinder",
				map[string]string{
					"type":         "",
					"availability": "",
				},
				"1.5Gi",
				"2Gi",
				nil, // there is currently nothing to check on OpenStack
			},
		}

		for i, t := range tests {
			// Beware of clojure, use local variables instead of those from
			// outer scope
			test := t
			suffix := fmt.Sprintf("%d", i)
			It(test.name, func() {
				if len(t.cloudProviders) > 0 {
					framework.SkipUnlessProviderIs(test.cloudProviders...)
				}

				class := newStorageClass(test, suffix)
				claim := newClaim(test, ns, suffix, false)
				testDynamicProvisioning(test, c, claim, class)
			})
		}
	})

	framework.KubeDescribe("DynamicProvisioner Alpha", func() {
		It("should provision alpha volumes [Slow]", func() {
			framework.SkipUnlessProviderIs("openstack", "gce", "aws", "gke")

			By("creating a claim with an alpha dynamic provisioning annotation")
			test := storageClassTest{
				name:         "alpha test",
				claimSize:    "1500Mi",
				expectedSize: "2Gi",
			}

			claim := newClaim(test, ns, "", true)
			testDynamicProvisioning(test, c, claim, nil)
		})
	})
})

func newClaim(t storageClassTest, ns, suffix string, alpha bool) *api.PersistentVolumeClaim {
	claim := api.PersistentVolumeClaim{
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
					api.ResourceName(api.ResourceStorage): resource.MustParse(t.claimSize),
				},
			},
		},
	}

	if alpha {
		claim.Annotations = map[string]string{
			"volume.alpha.kubernetes.io/storage-class": "",
		}
	} else {
		claim.Annotations = map[string]string{
			"volume.beta.kubernetes.io/storage-class": "myclass-" + suffix,
		}
	}

	return &claim
}

// runInPodWithVolume runs a command in a pod with given claim mounted to /mnt directory.
func runInPodWithVolume(c *client.Client, ns, claimName, command string) {
	pod := &api.Pod{
		TypeMeta: unversioned.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: api.ObjectMeta{
			GenerateName: "pvc-volume-tester-",
		},
		Spec: api.PodSpec{
			Containers: []api.Container{
				{
					Name:    "volume-tester",
					Image:   "gcr.io/google_containers/busybox:1.24",
					Command: []string{"/bin/sh"},
					Args:    []string{"-c", command},
					VolumeMounts: []api.VolumeMount{
						{
							Name:      "my-volume",
							MountPath: "/mnt/test",
						},
					},
				},
			},
			RestartPolicy: api.RestartPolicyNever,
			Volumes: []api.Volume{
				{
					Name: "my-volume",
					VolumeSource: api.VolumeSource{
						PersistentVolumeClaim: &api.PersistentVolumeClaimVolumeSource{
							ClaimName: claimName,
							ReadOnly:  false,
						},
					},
				},
			},
		},
	}
	pod, err := c.Pods(ns).Create(pod)
	framework.ExpectNoError(err, "Failed to create pod: %v", err)
	defer func() {
		framework.ExpectNoError(c.Pods(ns).Delete(pod.Name, nil))
	}()
	framework.ExpectNoError(framework.WaitForPodSuccessInNamespaceSlow(c, pod.Name, pod.Namespace))
}

func newStorageClass(t storageClassTest, suffix string) *storage.StorageClass {
	return &storage.StorageClass{
		TypeMeta: unversioned.TypeMeta{
			Kind: "StorageClass",
		},
		ObjectMeta: api.ObjectMeta{
			Name: "myclass-" + suffix,
		},
		Provisioner: t.provisioner,
		Parameters:  t.parameters,
	}
}
