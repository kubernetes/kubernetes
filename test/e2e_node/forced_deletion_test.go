package e2enode

import (
	"context"
	"github.com/onsi/ginkgo/v2"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
	"time"
)

var _ = SIGDescribe("Forced Deletion", func() {
	f := framework.NewDefaultFramework("forced-deletion")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline
	ginkgo.Context("When forcing pods to be deleted ", func() {
		var podClient *e2epod.PodClient
		ginkgo.BeforeEach(func() {
			podClient = e2epod.NewPodClient(f)
		})
		ginkgo.It("should be deleted immediately", func() {
			const (
				gracePeriod      = 100
				gracePeriodForce = 0
			)
			podName := "test"
			podClient.CreateSync(getGracePeriodTestPod(podName, gracePeriod))
			err := podClient.Delete(context.TODO(), podName, *metav1.NewDeleteOptions(gracePeriod))
			framework.ExpectNoError(err)
			start := time.Now()
			podClient.DeleteSync(podName, *metav1.NewDeleteOptions(gracePeriodForce), 4*time.Second)
			framework.ExpectEqual(time.Since(start) < gracePeriod*time.Second, true, "cannot forced deletion")
		})
	})
})

func getGracePeriodTestPod(name string, gracePeriod int64) *v1.Pod {
	pod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:    name,
					Image:   busyboxImage,
					Command: []string{"sh", "-c", "999999"},
				},
			},
			TerminationGracePeriodSeconds: &gracePeriod,
		},
	}
	return pod
}
