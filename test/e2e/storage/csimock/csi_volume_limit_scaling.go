package csimock

import (
	"context"
	"fmt"

	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	storagev1 "k8s.io/api/storage/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	clientset "k8s.io/client-go/kubernetes"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"

	"k8s.io/kubernetes/pkg/features"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2epv "k8s.io/kubernetes/test/e2e/framework/pv"
	"k8s.io/kubernetes/test/e2e/storage/testsuites"
	"k8s.io/kubernetes/test/e2e/storage/utils"
)

// Tests for VolumeLimitScaling scheduling behavior with PreventPodSchedulingIfMissing
var _ = utils.SIGDescribe("CSI Mock VolumeLimitScaling scheduling", func() {
	f := framework.NewDefaultFramework("csi-mock-volumes-limit-sched")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	f.Context("VolumeLimitScaling scheduling gate", feature.Volumes, func() {
		tests := []struct {
			name              string
			preventOptIn      *bool
			featureTags       []interface{}
			expectSchedulable bool
		}{
			{
				name:              "prevent=true blocks scheduling when driver not installed",
				preventOptIn:      ptr.To(true),
				featureTags:       []interface{}{framework.WithFeatureGate(features.VolumeLimitScaling)},
				expectSchedulable: false,
			},
			{
				name:              "prevent=false allows scheduling when driver not installed",
				preventOptIn:      ptr.To(false),
				featureTags:       []interface{}{framework.WithFeatureGate(features.VolumeLimitScaling)},
				expectSchedulable: true,
			},
		}

		for _, t := range tests {
			// capture range variable
			tc := t
			testFunc := func(ctx context.Context) {
				// Create a CSIDriver for a fake, not-installed driver name
				fakeDriver := fmt.Sprintf("csi-mock-uninstalled-%s", f.Namespace.Name)
				csiDriver := &storagev1.CSIDriver{
					ObjectMeta: metav1.ObjectMeta{Name: fakeDriver},
					Spec:       storagev1.CSIDriverSpec{PreventPodSchedulingIfMissing: tc.preventOptIn},
				}
				_, err := f.ClientSet.StorageV1().CSIDrivers().Create(ctx, csiDriver, metav1.CreateOptions{})
				framework.ExpectNoError(err, "creating CSIDriver %s", fakeDriver)
				ginkgo.DeferCleanup(func() {
					_ = f.ClientSet.StorageV1().CSIDrivers().Delete(context.TODO(), fakeDriver, metav1.DeleteOptions{})
				})

				// Create a StorageClass that uses the fake driver with WaitForFirstConsumer
				scTest := testsuites.StorageClassTest{
					Name:         fakeDriver,
					Provisioner:  fakeDriver,
					DelayBinding: false,
					ClaimSize:    "1Gi",
				}
				sc := createSC(f.ClientSet, scTest, "", f.Namespace.Name)
				pvConfig := e2epv.PersistentVolumeConfig{
					Capacity:         "1Gi",
					StorageClassName: sc.Name,
					VolumeMode:       ptr.To(v1.PersistentVolumeFilesystem),
					AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
					ReclaimPolicy:    v1.PersistentVolumeReclaimDelete,
					PVSource: v1.PersistentVolumeSource{
						CSI: &v1.CSIPersistentVolumeSource{
							Driver:       fakeDriver,
							VolumeHandle: "test-volume-handle",
						},
					},
				}
				pvcConfig := e2epv.PersistentVolumeClaimConfig{
					ClaimSize:        "1Gi",
					StorageClassName: &sc.Name,
					VolumeMode:       ptr.To(v1.PersistentVolumeFilesystem),
					AccessModes:      []v1.PersistentVolumeAccessMode{v1.ReadWriteOnce},
				}
				volume, claim, err := e2epv.CreatePVPVC(ctx, f.ClientSet, f.Timeouts, pvConfig, pvcConfig, f.Namespace.Name, true)
				framework.ExpectNoError(err, "creating PV and PVC")
				err = e2epv.WaitOnPVandPVC(ctx, f.ClientSet, f.Timeouts, f.Namespace.Name, volume, claim)
				framework.ExpectNoError(err, "waiting for PV and PVC to be bound each other")

				// Create a pod using that PVC
				pod, err := createPodWithPVCWithoutNodeSelection(f.ClientSet, claim, f.Namespace.Name)
				framework.ExpectNoError(err, "creating pod")

				if tc.expectSchedulable {
					// Pod should get scheduled to some node (NodeName set). Allow more time in case binders are slow.
					err = e2epod.WaitForPodCondition(ctx, f.ClientSet, pod.Namespace, pod.Name, "pod is scheduled", f.Timeouts.PodStart, func(p *v1.Pod) (bool, error) {
						return p.Spec.NodeName != "", nil
					})
					framework.ExpectNoError(err, "waiting for pod to be scheduled when prevent=false")
				} else {
					// Expect FailedScheduling event with driver-not-installed message
					framework.Logf("Waiting for FailedScheduling event with driver-not-installed message")
					framework.Logf("Pod status: %+v", pod.Status)
					failedMsg := fmt.Sprintf("%s CSI driver is not installed on the node", fakeDriver)
					eventSelector := fields.Set{
						"involvedObject.kind":      "Pod",
						"involvedObject.name":      pod.Name,
						"involvedObject.namespace": pod.Namespace,
						"reason":                   "FailedScheduling",
					}.AsSelector().String()
					err = e2eevents.WaitTimeoutForEvent(ctx, f.ClientSet, pod.Namespace, eventSelector, failedMsg, f.Timeouts.PodStart)
					framework.ExpectNoError(err, "waiting for FailedScheduling due to missing CSI driver")
				}
			}

			// Compose It with feature tags
			args := []interface{}{tc.name, testFunc}
			args = append(args, tc.featureTags...)
			framework.It(args...)
		}
	})
})

func createPodWithPVCWithoutNodeSelection(cs clientset.Interface, pvc *v1.PersistentVolumeClaim, ns string) (*v1.Pod, error) {
	return startPausePodWithClaim(cs, pvc, e2epod.NodeSelection{}, ns)
}
