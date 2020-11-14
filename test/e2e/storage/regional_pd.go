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

package storage

import (
	"context"
	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"

	"fmt"
	"strings"
	"time"

	"encoding/json"

	appsv1 "k8s.io/api/apps/v1"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	volumehelpers "k8s.io/cloud-provider/volume/helpers"
	podutil "k8s.io/kubernetes/pkg/api/v1/pod"
	"k8s.io/kubernetes/test/e2e/framework"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	pvDeletionTimeout       = 3 * time.Minute
	statefulSetReadyTimeout = 3 * time.Minute
	taintKeyPrefix          = "zoneTaint_"
	repdMinSize             = "200Gi"
	pvcName                 = "regional-pd-vol"
)

var _ = utils.SIGDescribe("Regional PD", func() {
	f := framework.NewDefaultFramework("regional-pd")

	// filled in BeforeEach
	var c clientset.Interface
	var ns string

	ginkgo.BeforeEach(func() {
		c = f.ClientSet
		ns = f.Namespace.Name

		e2eskipper.SkipUnlessProviderIs("gce", "gke")
		e2eskipper.SkipUnlessMultizone(c)
	})

	ginkgo.Describe("RegionalPD", func() {
		ginkgo.It("should provision storage [Slow]", func() {
			testVolumeProvisioning(c, ns)
		})

		ginkgo.It("should provision storage with delayed binding [Slow]", func() {
			testRegionalDelayedBinding(c, ns, 1 /* pvcCount */)
			testRegionalDelayedBinding(c, ns, 3 /* pvcCount */)
		})

		ginkgo.It("should provision storage in the allowedTopologies [Slow]", func() {
			testRegionalAllowedTopologies(c, ns)
		})

		ginkgo.It("should provision storage in the allowedTopologies with delayed binding [Slow]", func() {
			testRegionalAllowedTopologiesWithDelayedBinding(c, ns, 1 /* pvcCount */)
			testRegionalAllowedTopologiesWithDelayedBinding(c, ns, 3 /* pvcCount */)
		})

		ginkgo.It("should failover to a different zone when all nodes in one zone become unreachable [Slow] [Disruptive]", func() {
			testZonalFailover(c, ns)
		})
	})
})

func testVolumeProvisioning(c clientset.Interface, ns string) {
	cloudZones := getTwoRandomZones(c)

	// This test checks that dynamic provisioning can provision a volume
	// that can be used to persist data among pods.
	tests := []testsuites.StorageClassTest{
		{
			Name:           "HDD Regional PD on GCE/GKE",
			CloudProviders: []string{"gce", "gke"},
			Provisioner:    "kubernetes.io/gce-pd",
			Parameters: map[string]string{
				"type":             "pd-standard",
				"zones":            strings.Join(cloudZones, ","),
				"replication-type": "regional-pd",
			},
			ClaimSize:    repdMinSize,
			ExpectedSize: repdMinSize,
			PvCheck: func(claim *v1.PersistentVolumeClaim) {
				volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
				gomega.Expect(volume).NotTo(gomega.BeNil())

				err := checkGCEPD(volume, "pd-standard")
				framework.ExpectNoError(err, "checkGCEPD")
				err = verifyZonesInPV(volume, sets.NewString(cloudZones...), true /* match */)
				framework.ExpectNoError(err, "verifyZonesInPV")

			},
		},
		{
			Name:           "HDD Regional PD with auto zone selection on GCE/GKE",
			CloudProviders: []string{"gce", "gke"},
			Provisioner:    "kubernetes.io/gce-pd",
			Parameters: map[string]string{
				"type":             "pd-standard",
				"replication-type": "regional-pd",
			},
			ClaimSize:    repdMinSize,
			ExpectedSize: repdMinSize,
			PvCheck: func(claim *v1.PersistentVolumeClaim) {
				volume := testsuites.PVWriteReadSingleNodeCheck(c, claim, e2epod.NodeSelection{})
				gomega.Expect(volume).NotTo(gomega.BeNil())

				err := checkGCEPD(volume, "pd-standard")
				framework.ExpectNoError(err, "checkGCEPD")
				zones, err := e2enode.GetClusterZones(c)
				framework.ExpectNoError(err, "GetClusterZones")
				err = verifyZonesInPV(volume, zones, false /* match */)
				framework.ExpectNoError(err, "verifyZonesInPV")
			},
		},
	}

	for _, test := range tests {
		test.Client = c
		test.Class = newStorageClass(test, ns, "" /* suffix */)
		test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        test.ClaimSize,
			StorageClassName: &(test.Class.Name),
			VolumeMode:       &test.VolumeMode,
		}, ns)
		test.TestDynamicProvisioning()
	}
}

func testZonalFailover(c clientset.Interface, ns string) {
	cloudZones := getTwoRandomZones(c)
	testSpec := testsuites.StorageClassTest{
		Name:           "Regional PD Failover on GCE/GKE",
		CloudProviders: []string{"gce", "gke"},
		Provisioner:    "kubernetes.io/gce-pd",
		Parameters: map[string]string{
			"type":             "pd-standard",
			"zones":            strings.Join(cloudZones, ","),
			"replication-type": "regional-pd",
		},
		ClaimSize:    repdMinSize,
		ExpectedSize: repdMinSize,
	}
	class := newStorageClass(testSpec, ns, "" /* suffix */)
	claimTemplate := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
		NamePrefix:       pvcName,
		ClaimSize:        testSpec.ClaimSize,
		StorageClassName: &(class.Name),
		VolumeMode:       &testSpec.VolumeMode,
	}, ns)
	statefulSet, service, regionalPDLabels := newStatefulSet(claimTemplate, ns)

	ginkgo.By("creating a StorageClass " + class.Name)
	_, err := c.StorageV1().StorageClasses().Create(context.TODO(), class, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	defer func() {
		framework.Logf("deleting storage class %s", class.Name)
		framework.ExpectNoError(c.StorageV1().StorageClasses().Delete(context.TODO(), class.Name, metav1.DeleteOptions{}),
			"Error deleting StorageClass %s", class.Name)
	}()

	ginkgo.By("creating a StatefulSet")
	_, err = c.CoreV1().Services(ns).Create(context.TODO(), service, metav1.CreateOptions{})
	framework.ExpectNoError(err)
	_, err = c.AppsV1().StatefulSets(ns).Create(context.TODO(), statefulSet, metav1.CreateOptions{})
	framework.ExpectNoError(err)

	defer func() {
		framework.Logf("deleting statefulset%q/%q", statefulSet.Namespace, statefulSet.Name)
		// typically this claim has already been deleted
		framework.ExpectNoError(c.AppsV1().StatefulSets(ns).Delete(context.TODO(), statefulSet.Name, metav1.DeleteOptions{}),
			"Error deleting StatefulSet %s", statefulSet.Name)

		framework.Logf("deleting claims in namespace %s", ns)
		pvc := getPVC(c, ns, regionalPDLabels)
		framework.ExpectNoError(c.CoreV1().PersistentVolumeClaims(pvc.Namespace).Delete(context.TODO(), pvc.Name, metav1.DeleteOptions{}),
			"Error deleting claim %s.", pvc.Name)
		if pvc.Spec.VolumeName != "" {
			err = e2epv.WaitForPersistentVolumeDeleted(c, pvc.Spec.VolumeName, framework.Poll, pvDeletionTimeout)
			if err != nil {
				framework.Logf("WARNING: PV %s is not yet deleted, and subsequent tests may be affected.", pvc.Spec.VolumeName)
			}
		}
	}()

	err = waitForStatefulSetReplicasReady(statefulSet.Name, ns, c, framework.Poll, statefulSetReadyTimeout)
	if err != nil {
		pod := getPod(c, ns, regionalPDLabels)
		framework.ExpectEqual(podutil.IsPodReadyConditionTrue(pod.Status), true, "The statefulset pod has the following conditions: %s", pod.Status.Conditions)
		framework.ExpectNoError(err)
	}

	pvc := getPVC(c, ns, regionalPDLabels)

	ginkgo.By("getting zone information from pod")
	pod := getPod(c, ns, regionalPDLabels)
	nodeName := pod.Spec.NodeName
	node, err := c.CoreV1().Nodes().Get(context.TODO(), nodeName, metav1.GetOptions{})
	framework.ExpectNoError(err)
	podZone := node.Labels[v1.LabelFailureDomainBetaZone]

	ginkgo.By("tainting nodes in the zone the pod is scheduled in")
	selector := labels.SelectorFromSet(labels.Set(map[string]string{v1.LabelFailureDomainBetaZone: podZone}))
	nodesInZone, err := c.CoreV1().Nodes().List(context.TODO(), metav1.ListOptions{LabelSelector: selector.String()})
	framework.ExpectNoError(err)
	removeTaintFunc := addTaint(c, ns, nodesInZone.Items, podZone)

	defer func() {
		framework.Logf("removing previously added node taints")
		removeTaintFunc()
	}()

	ginkgo.By("deleting StatefulSet pod")
	err = c.CoreV1().Pods(ns).Delete(context.TODO(), pod.Name, metav1.DeleteOptions{})

	// Verify the pod is scheduled in the other zone.
	ginkgo.By("verifying the pod is scheduled in a different zone.")
	var otherZone string
	if cloudZones[0] == podZone {
		otherZone = cloudZones[1]
	} else {
		otherZone = cloudZones[0]
	}
	waitErr := wait.PollImmediate(framework.Poll, statefulSetReadyTimeout, func() (bool, error) {
		framework.Logf("Checking whether new pod is scheduled in zone %q", otherZone)
		pod := getPod(c, ns, regionalPDLabels)
		node, err := c.CoreV1().Nodes().Get(context.TODO(), pod.Spec.NodeName, metav1.GetOptions{})
		if err != nil {
			return false, nil
		}
		newPodZone := node.Labels[v1.LabelFailureDomainBetaZone]
		return newPodZone == otherZone, nil
	})
	framework.ExpectNoError(waitErr, "Error waiting for pod to be scheduled in a different zone (%q): %v", otherZone, err)

	err = waitForStatefulSetReplicasReady(statefulSet.Name, ns, c, 3*time.Second, framework.RestartPodReadyAgainTimeout)
	if err != nil {
		pod := getPod(c, ns, regionalPDLabels)
		framework.ExpectEqual(podutil.IsPodReadyConditionTrue(pod.Status), true, "The statefulset pod has the following conditions: %s", pod.Status.Conditions)
		framework.ExpectNoError(err)
	}

	ginkgo.By("verifying the same PVC is used by the new pod")
	framework.ExpectEqual(getPVC(c, ns, regionalPDLabels).Name, pvc.Name, "The same PVC should be used after failover.")

	ginkgo.By("verifying the container output has 2 lines, indicating the pod has been created twice using the same regional PD.")
	logs, err := e2epod.GetPodLogs(c, ns, pod.Name, "")
	framework.ExpectNoError(err,
		"Error getting logs from pod %s in namespace %s", pod.Name, ns)
	lineCount := len(strings.Split(strings.TrimSpace(logs), "\n"))
	expectedLineCount := 2
	framework.ExpectEqual(lineCount, expectedLineCount, "Line count of the written file should be %d.", expectedLineCount)

}

func addTaint(c clientset.Interface, ns string, nodes []v1.Node, podZone string) (removeTaint func()) {
	reversePatches := make(map[string][]byte)
	for _, node := range nodes {
		oldData, err := json.Marshal(node)
		framework.ExpectNoError(err)

		node.Spec.Taints = append(node.Spec.Taints, v1.Taint{
			Key:    taintKeyPrefix + ns,
			Value:  podZone,
			Effect: v1.TaintEffectNoSchedule,
		})

		newData, err := json.Marshal(node)
		framework.ExpectNoError(err)

		patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Node{})
		framework.ExpectNoError(err)

		reversePatchBytes, err := strategicpatch.CreateTwoWayMergePatch(newData, oldData, v1.Node{})
		framework.ExpectNoError(err)
		reversePatches[node.Name] = reversePatchBytes

		_, err = c.CoreV1().Nodes().Patch(context.TODO(), node.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
	}

	return func() {
		for nodeName, reversePatch := range reversePatches {
			_, err := c.CoreV1().Nodes().Patch(context.TODO(), nodeName, types.StrategicMergePatchType, reversePatch, metav1.PatchOptions{})
			framework.ExpectNoError(err)
		}
	}
}

func testRegionalDelayedBinding(c clientset.Interface, ns string, pvcCount int) {
	test := testsuites.StorageClassTest{
		Client:      c,
		Name:        "Regional PD storage class with waitForFirstConsumer test on GCE",
		Provisioner: "kubernetes.io/gce-pd",
		Parameters: map[string]string{
			"type":             "pd-standard",
			"replication-type": "regional-pd",
		},
		ClaimSize:    repdMinSize,
		DelayBinding: true,
	}

	suffix := "delayed-regional"
	test.Class = newStorageClass(test, ns, suffix)
	var claims []*v1.PersistentVolumeClaim
	for i := 0; i < pvcCount; i++ {
		claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        test.ClaimSize,
			StorageClassName: &(test.Class.Name),
			VolumeMode:       &test.VolumeMode,
		}, ns)
		claims = append(claims, claim)
	}
	pvs, node := test.TestBindingWaitForFirstConsumerMultiPVC(claims, nil /* node selector */, false /* expect unschedulable */)
	if node == nil {
		framework.Failf("unexpected nil node found")
	}
	zone, ok := node.Labels[v1.LabelFailureDomainBetaZone]
	if !ok {
		framework.Failf("label %s not found on Node", v1.LabelFailureDomainBetaZone)
	}
	for _, pv := range pvs {
		checkZoneFromLabelAndAffinity(pv, zone, false)
	}
}

func testRegionalAllowedTopologies(c clientset.Interface, ns string) {
	test := testsuites.StorageClassTest{
		Name:        "Regional PD storage class with allowedTopologies test on GCE",
		Provisioner: "kubernetes.io/gce-pd",
		Parameters: map[string]string{
			"type":             "pd-standard",
			"replication-type": "regional-pd",
		},
		ClaimSize:    repdMinSize,
		ExpectedSize: repdMinSize,
	}

	suffix := "topo-regional"
	test.Client = c
	test.Class = newStorageClass(test, ns, suffix)
	zones := getTwoRandomZones(c)
	addAllowedTopologiesToStorageClass(c, test.Class, zones)
	test.Claim = e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
		NamePrefix:       pvcName,
		ClaimSize:        test.ClaimSize,
		StorageClassName: &(test.Class.Name),
		VolumeMode:       &test.VolumeMode,
	}, ns)

	pv := test.TestDynamicProvisioning()
	checkZonesFromLabelAndAffinity(pv, sets.NewString(zones...), true)
}

func testRegionalAllowedTopologiesWithDelayedBinding(c clientset.Interface, ns string, pvcCount int) {
	test := testsuites.StorageClassTest{
		Client:      c,
		Name:        "Regional PD storage class with allowedTopologies and waitForFirstConsumer test on GCE",
		Provisioner: "kubernetes.io/gce-pd",
		Parameters: map[string]string{
			"type":             "pd-standard",
			"replication-type": "regional-pd",
		},
		ClaimSize:    repdMinSize,
		DelayBinding: true,
	}

	suffix := "topo-delayed-regional"
	test.Class = newStorageClass(test, ns, suffix)
	topoZones := getTwoRandomZones(c)
	addAllowedTopologiesToStorageClass(c, test.Class, topoZones)
	var claims []*v1.PersistentVolumeClaim
	for i := 0; i < pvcCount; i++ {
		claim := e2epv.MakePersistentVolumeClaim(e2epv.PersistentVolumeClaimConfig{
			ClaimSize:        test.ClaimSize,
			StorageClassName: &(test.Class.Name),
			VolumeMode:       &test.VolumeMode,
		}, ns)
		claims = append(claims, claim)
	}
	pvs, node := test.TestBindingWaitForFirstConsumerMultiPVC(claims, nil /* node selector */, false /* expect unschedulable */)
	if node == nil {
		framework.Failf("unexpected nil node found")
	}
	nodeZone, ok := node.Labels[v1.LabelFailureDomainBetaZone]
	if !ok {
		framework.Failf("label %s not found on Node", v1.LabelFailureDomainBetaZone)
	}
	zoneFound := false
	for _, zone := range topoZones {
		if zone == nodeZone {
			zoneFound = true
			break
		}
	}
	if !zoneFound {
		framework.Failf("zones specified in AllowedTopologies: %v does not contain zone of node where PV got provisioned: %s", topoZones, nodeZone)
	}
	for _, pv := range pvs {
		checkZonesFromLabelAndAffinity(pv, sets.NewString(topoZones...), true)
	}
}

func getPVC(c clientset.Interface, ns string, pvcLabels map[string]string) *v1.PersistentVolumeClaim {
	selector := labels.Set(pvcLabels).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	pvcList, err := c.CoreV1().PersistentVolumeClaims(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(pvcList.Items), 1, "There should be exactly 1 PVC matched.")

	return &pvcList.Items[0]
}

func getPod(c clientset.Interface, ns string, podLabels map[string]string) *v1.Pod {
	selector := labels.Set(podLabels).AsSelector()
	options := metav1.ListOptions{LabelSelector: selector.String()}
	podList, err := c.CoreV1().Pods(ns).List(context.TODO(), options)
	framework.ExpectNoError(err)
	framework.ExpectEqual(len(podList.Items), 1, "There should be exactly 1 pod matched.")

	return &podList.Items[0]
}

func addAllowedTopologiesToStorageClass(c clientset.Interface, sc *storagev1.StorageClass, zones []string) {
	term := v1.TopologySelectorTerm{
		MatchLabelExpressions: []v1.TopologySelectorLabelRequirement{
			{
				Key:    v1.LabelFailureDomainBetaZone,
				Values: zones,
			},
		},
	}
	sc.AllowedTopologies = append(sc.AllowedTopologies, term)
}

// Generates the spec of a StatefulSet with 1 replica that mounts a Regional PD.
func newStatefulSet(claimTemplate *v1.PersistentVolumeClaim, ns string) (sts *appsv1.StatefulSet, svc *v1.Service, labels map[string]string) {
	var replicas int32 = 1
	labels = map[string]string{"app": "regional-pd-workload"}

	svc = &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "regional-pd-service",
			Namespace: ns,
			Labels:    labels,
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{
				Port: 80,
				Name: "web",
			}},
			ClusterIP: v1.ClusterIPNone,
			Selector:  labels,
		},
	}

	sts = &appsv1.StatefulSet{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "regional-pd-sts",
			Namespace: ns,
		},
		Spec: appsv1.StatefulSetSpec{
			Selector: &metav1.LabelSelector{
				MatchLabels: labels,
			},
			ServiceName:          svc.Name,
			Replicas:             &replicas,
			Template:             *newPodTemplate(labels),
			VolumeClaimTemplates: []v1.PersistentVolumeClaim{*claimTemplate},
		},
	}

	return
}

func newPodTemplate(labels map[string]string) *v1.PodTemplateSpec {
	return &v1.PodTemplateSpec{
		ObjectMeta: metav1.ObjectMeta{
			Labels: labels,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				// This container writes its pod name to a file in the Regional PD
				// and prints the entire file to stdout.
				{
					Name:    "busybox",
					Image:   imageutils.GetE2EImage(imageutils.BusyBox),
					Command: []string{"sh", "-c"},
					Args: []string{
						"echo ${POD_NAME} >> /mnt/data/regional-pd/pods.txt;" +
							"cat /mnt/data/regional-pd/pods.txt;" +
							"sleep 3600;",
					},
					Env: []v1.EnvVar{{
						Name: "POD_NAME",
						ValueFrom: &v1.EnvVarSource{
							FieldRef: &v1.ObjectFieldSelector{
								FieldPath: "metadata.name",
							},
						},
					}},
					Ports: []v1.ContainerPort{{
						ContainerPort: 80,
						Name:          "web",
					}},
					VolumeMounts: []v1.VolumeMount{{
						Name:      pvcName,
						MountPath: "/mnt/data/regional-pd",
					}},
				},
			},
		},
	}
}

func getTwoRandomZones(c clientset.Interface) []string {
	zones, err := e2enode.GetClusterZones(c)
	framework.ExpectNoError(err)
	gomega.Expect(zones.Len()).To(gomega.BeNumerically(">=", 2),
		"The test should only be run in multizone clusters.")

	zone1, _ := zones.PopAny()
	zone2, _ := zones.PopAny()
	return []string{zone1, zone2}
}

// If match is true, check if zones in PV exactly match zones given.
// Otherwise, check whether zones in PV is superset of zones given.
func verifyZonesInPV(volume *v1.PersistentVolume, zones sets.String, match bool) error {
	pvZones, err := volumehelpers.LabelZonesToSet(volume.Labels[v1.LabelFailureDomainBetaZone])
	if err != nil {
		return err
	}

	if match && zones.Equal(pvZones) || !match && zones.IsSuperset(pvZones) {
		return nil
	}

	return fmt.Errorf("Zones in StorageClass are %v, but zones in PV are %v", zones, pvZones)

}

func checkZoneFromLabelAndAffinity(pv *v1.PersistentVolume, zone string, matchZone bool) {
	checkZonesFromLabelAndAffinity(pv, sets.NewString(zone), matchZone)
}

// checkZoneLabelAndAffinity checks the LabelFailureDomainBetaZone label of PV and terms
// with key LabelFailureDomainBetaZone in PV's node affinity contains zone
// matchZones is used to indicate if zones should match perfectly
func checkZonesFromLabelAndAffinity(pv *v1.PersistentVolume, zones sets.String, matchZones bool) {
	ginkgo.By("checking PV's zone label and node affinity terms match expected zone")
	if pv == nil {
		framework.Failf("nil pv passed")
	}
	pvLabel, ok := pv.Labels[v1.LabelFailureDomainBetaZone]
	if !ok {
		framework.Failf("label %s not found on PV", v1.LabelFailureDomainBetaZone)
	}

	zonesFromLabel, err := volumehelpers.LabelZonesToSet(pvLabel)
	if err != nil {
		framework.Failf("unable to parse zone labels %s: %v", pvLabel, err)
	}
	if matchZones && !zonesFromLabel.Equal(zones) {
		framework.Failf("value[s] of %s label for PV: %v does not match expected zone[s]: %v", v1.LabelFailureDomainBetaZone, zonesFromLabel, zones)
	}
	if !matchZones && !zonesFromLabel.IsSuperset(zones) {
		framework.Failf("value[s] of %s label for PV: %v does not contain expected zone[s]: %v", v1.LabelFailureDomainBetaZone, zonesFromLabel, zones)
	}
	if pv.Spec.NodeAffinity == nil {
		framework.Failf("node affinity not found in PV spec %v", pv.Spec)
	}
	if len(pv.Spec.NodeAffinity.Required.NodeSelectorTerms) == 0 {
		framework.Failf("node selector terms not found in PV spec %v", pv.Spec)
	}

	for _, term := range pv.Spec.NodeAffinity.Required.NodeSelectorTerms {
		keyFound := false
		for _, r := range term.MatchExpressions {
			if r.Key != v1.LabelFailureDomainBetaZone {
				continue
			}
			keyFound = true
			zonesFromNodeAffinity := sets.NewString(r.Values...)
			if matchZones && !zonesFromNodeAffinity.Equal(zones) {
				framework.Failf("zones from NodeAffinity of PV: %v does not equal expected zone[s]: %v", zonesFromNodeAffinity, zones)
			}
			if !matchZones && !zonesFromNodeAffinity.IsSuperset(zones) {
				framework.Failf("zones from NodeAffinity of PV: %v does not contain expected zone[s]: %v", zonesFromNodeAffinity, zones)
			}
			break
		}
		if !keyFound {
			framework.Failf("label %s not found in term %v", v1.LabelFailureDomainBetaZone, term)
		}
	}
}

// waitForStatefulSetReplicasReady waits for all replicas of a StatefulSet to become ready or until timeout occurs, whichever comes first.
func waitForStatefulSetReplicasReady(statefulSetName, ns string, c clientset.Interface, Poll, timeout time.Duration) error {
	framework.Logf("Waiting up to %v for StatefulSet %s to have all replicas ready", timeout, statefulSetName)
	for start := time.Now(); time.Since(start) < timeout; time.Sleep(Poll) {
		sts, err := c.AppsV1().StatefulSets(ns).Get(context.TODO(), statefulSetName, metav1.GetOptions{})
		if err != nil {
			framework.Logf("Get StatefulSet %s failed, ignoring for %v: %v", statefulSetName, Poll, err)
			continue
		}
		if sts.Status.ReadyReplicas == *sts.Spec.Replicas {
			framework.Logf("All %d replicas of StatefulSet %s are ready. (%v)", sts.Status.ReadyReplicas, statefulSetName, time.Since(start))
			return nil
		}
		framework.Logf("StatefulSet %s found but there are %d ready replicas and %d total replicas.", statefulSetName, sts.Status.ReadyReplicas, *sts.Spec.Replicas)
	}
	return fmt.Errorf("StatefulSet %s still has unready pods within %v", statefulSetName, timeout)
}
