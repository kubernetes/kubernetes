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

package storage

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = utils.SIGDescribe("Multi-AZ Cluster Volumes", func() {
	f := framework.NewDefaultFramework("multi-az")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelPrivileged
	var zoneCount int
	var err error
	image := framework.ServeHostnameImage
	ginkgo.BeforeEach(func() {
		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		if zoneCount <= 0 {
			zoneCount, err = getZoneCount(f.ClientSet)
			framework.ExpectNoError(err)
		}
		ginkgo.By(fmt.Sprintf("Checking for multi-zone cluster.  Zone count = %d", zoneCount))
		msg := fmt.Sprintf("Zone count is %d, only run for multi-zone clusters, skipping test", zoneCount)
		e2eskipper.SkipUnlessAtLeast(zoneCount, 2, msg)
		// TODO: SkipUnlessDefaultScheduler() // Non-default schedulers might not spread
	})
	ginkgo.It("should schedule pods in the same zones as statically provisioned PVs", func() {
		PodsUseStaticPVsOrFail(f, (2*zoneCount)+1, image)
	})
})

// Return the number of zones in which we have nodes in this cluster.
func getZoneCount(c clientset.Interface) (int, error) {
	zoneNames, err := e2enode.GetSchedulableClusterZones(c)
	if err != nil {
		return -1, err
	}
	return len(zoneNames), nil
}

type staticPVTestConfig struct {
	pvSource *v1.PersistentVolumeSource
	pv       *v1.PersistentVolume
	pvc      *v1.PersistentVolumeClaim
	pod      *v1.Pod
}

// PodsUseStaticPVsOrFail Check that the pods using statically
// created PVs get scheduled to the same zone that the PV is in.
func PodsUseStaticPVsOrFail(f *framework.Framework, podCount int, image string) {
	var err error
	c := f.ClientSet
	ns := f.Namespace.Name

	zones, err := e2enode.GetSchedulableClusterZones(c)
	framework.ExpectNoError(err)
	zonelist := zones.List()
	ginkgo.By("Creating static PVs across zones")
	configs := make([]*staticPVTestConfig, podCount)
	for i := range configs {
		configs[i] = &staticPVTestConfig{}
	}

	defer func() {
		ginkgo.By("Cleaning up pods and PVs")
		for _, config := range configs {
			e2epod.DeletePodOrFail(c, ns, config.pod.Name)
		}
		for _, config := range configs {
			e2epod.WaitForPodNoLongerRunningInNamespace(c, config.pod.Name, ns)
			e2epv.PVPVCCleanup(c, ns, config.pv, config.pvc)
			err = e2epv.DeletePVSource(config.pvSource)
			framework.ExpectNoError(err)
		}
	}()

	for i, config := range configs {
		zone := zonelist[i%len(zones)]
		config.pvSource, err = e2epv.CreatePVSource(zone)
		framework.ExpectNoError(err)

		pvConfig := e2epv.PersistentVolumeConfig{
			NamePrefix: "multizone-pv",
			PVSource:   *config.pvSource,
			Prebind:    nil,
		}
		className := ""
		pvcConfig := e2epv.PersistentVolumeClaimConfig{StorageClassName: &className}

		config.pv, config.pvc, err = e2epv.CreatePVPVC(c, f.Timeouts, pvConfig, pvcConfig, ns, true)
		framework.ExpectNoError(err)
	}

	ginkgo.By("Waiting for all PVCs to be bound")
	for _, config := range configs {
		e2epv.WaitOnPVandPVC(c, f.Timeouts, ns, config.pv, config.pvc)
	}

	ginkgo.By("Creating pods for each static PV")
	for _, config := range configs {
		podConfig := e2epod.MakePod(ns, nil, []*v1.PersistentVolumeClaim{config.pvc}, false, "")
		config.pod, err = c.CoreV1().Pods(ns).Create(context.TODO(), podConfig, metav1.CreateOptions{})
		framework.ExpectNoError(err)
	}

	ginkgo.By("Waiting for all pods to be running")
	for _, config := range configs {
		err = e2epod.WaitForPodRunningInNamespace(c, config.pod)
		framework.ExpectNoError(err)
	}
}
