package e2enode

import (
	"context"
	"time"

	imageutils "k8s.io/kubernetes/test/utils/image"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	kubeletconfig "k8s.io/kubernetes/pkg/kubelet/apis/config"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	kubeletevents "k8s.io/kubernetes/pkg/kubelet/events"
	"k8s.io/kubernetes/test/e2e/feature"
	"k8s.io/kubernetes/test/e2e/framework"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	admissionapi "k8s.io/pod-security-admission/api"
)

var _ = SIGDescribe("Container Restart", feature.CriProxy, framework.WithSerial(), func() {
	f := framework.NewDefaultFramework("container-restart")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.Context("Container restart backs off", func() {

		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
		})

		ginkgo.AfterEach(func() {
			err := resetCRIProxyInjector(e2eCriProxy)
			framework.ExpectNoError(err)
		})

		ginkgo.It("Container restart backs off.", func(ctx context.Context) {
			// 3 would take 10s best case, 6 would take 150s best case
			doTest(ctx, f, 5, time.Duration(80*time.Second), time.Duration(10*time.Second))
		})
	})

	ginkgo.Context("Alternate container restart backs off as expected", func() {

		tempSetCurrentKubeletConfig(f, func(ctx context.Context, initialConfig *kubeletconfig.KubeletConfiguration) {
			initialConfig.CrashLoopBackOff.MaxContainerRestartPeriod = &metav1.Duration{Duration: time.Duration(30 * time.Second)}
			initialConfig.FeatureGates = map[string]bool{"KubeletCrashLoopBackOffMax": true}
		})

		ginkgo.BeforeEach(func() {
			if err := resetCRIProxyInjector(e2eCriProxy); err != nil {
				ginkgo.Skip("Skip the test since the CRI Proxy is undefined.")
			}
		})

		ginkgo.AfterEach(func() {
			err := resetCRIProxyInjector(e2eCriProxy)
			framework.ExpectNoError(err)
		})

		ginkgo.It("Alternate restart backs off.", func(ctx context.Context) {
			doTest(ctx, f, 7, time.Duration(120*time.Second), time.Duration(10*time.Second))
		})
	})
})

func doTest(ctx context.Context, f *framework.Framework, maxRestarts int, target time.Duration, threshold time.Duration) {

	pod := e2epod.NewPodClient(f).Create(ctx, newFailAlwaysPod())
	podErr := e2epod.WaitForPodContainerToFail(ctx, f.ClientSet, f.Namespace.Name, pod.Name, 0, "CrashLoopBackOff", 1*time.Minute)
	gomega.Expect(podErr).To(gomega.HaveOccurred())

	// Wait for 120s worth of backoffs to occur so we can confirm the backoff growth.
	podErr = e2epod.WaitForContainerRestartedNTimes(ctx, f.ClientSet, f.Namespace.Name, pod.Name, "restart", 150*time.Second, maxRestarts)
	gomega.Expect(podErr).ShouldNot(gomega.HaveOccurred(), "Expected container to repeatedly back off container failures")

	d, err := getContainerRetryDuration(ctx, f, pod.Name)
	framework.ExpectNoError(err)

	gomega.Expect(d).Should(gomega.BeNumerically("~", target, threshold))
}

func getContainerRetryDuration(ctx context.Context, f *framework.Framework, podName string) (time.Duration, error) {

	var d time.Duration
	e, err := f.ClientSet.CoreV1().Events(f.Namespace.Name).List(ctx, metav1.ListOptions{})
	if err != nil {
		return d, err
	}

	for _, event := range e.Items {
		if event.InvolvedObject.Name == podName && event.Reason == kubeletevents.StartedContainer {
			return event.LastTimestamp.Time.Sub(event.FirstTimestamp.Time), nil
		}
	}
	return d, nil
}

func newFailAlwaysPod() *v1.Pod {
	podName := "container-restart" + string(uuid.NewUUID())
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name: podName,
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Name:            "restart",
					Image:           imageutils.GetBusyBoxImageName(),
					ImagePullPolicy: v1.PullAlways,
					Command:         []string{"exit 1"},
				},
			},
		},
	}
	return pod
}
