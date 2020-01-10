package e2enode

import (
    "context"
    "fmt"
    "io/ioutil"
    "k8s.io/apimachinery/pkg/api/errors"
    "path/filepath"
    "time"

    "github.com/onsi/ginkgo"
    "github.com/onsi/gomega"

    "k8s.io/api/core/v1"
    metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
    "k8s.io/kubernetes/test/e2e/framework"
    e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
)

func createStaticPodWithEphemeralContainer(f *framework.Framework) error {
    template := `
apiVersion: v1
kind: Pod
metadata:
  name: %s
  namespace: %s
spec:
  containers:
  - name: test-container
    image: %s
    command: ["sh", "-c", "sleep 3600"]
  ephemeralContainers:
  - name: ephemeral-container
    image: %s
    imagePullPolicy: Never
    tty: true
    stdin: true
    terminationMessagePolicy: FallbackToLogsOnError
    command: ["sh", "-c", "sleep 3600"]		
`
    return ioutil.WriteFile(filepath.Join(framework.TestContext.KubeletConfig.StaticPodPath, "test-static-pod.yaml"), []byte(fmt.Sprintf(template, "test-static-pod", f.Namespace.Name, busyboxImage, busyboxImage)), 0644)
}

func createPodWithoutEphemeralContainer(f *framework.Framework) *v1.Pod {
    podDesc := &v1.Pod{
        ObjectMeta: metav1.ObjectMeta{
            Name: "test-pod",
        },
        Spec: v1.PodSpec{
            Containers: []v1.Container{
                {
                    Name:    "main",
                    Image:   busyboxImage,
                    Command: []string{"sh", "-c", "while true; do echo main container; sleep 1; done"},
                }},
            RestartPolicy: v1.RestartPolicyNever,
        },
    }

    pod := f.PodClient().CreateSync(podDesc)

    return pod
}

func ephemeralContainersEnabled(f *framework.Framework) (bool, error) {
    pod := createPodWithoutEphemeralContainer(f)
    var gracePeriodSeconds int64 = 0
    defer f.PodClient().DeleteSync(pod.Name, metav1.DeleteOptions{GracePeriodSeconds: &gracePeriodSeconds}, 10*time.Second)
    _, err := f.PodClient().GetEphemeralContainers(context.TODO(), pod.Name, metav1.GetOptions{})
    if serr, ok := err.(*errors.StatusError); ok && serr.Status().Reason == metav1.StatusReasonNotFound {
        return false, nil
    }
    if err != nil {
        return false, err
    }
    return true, nil
}

var _ = framework.KubeDescribe("Ephemeral Containers [NodeFeature:EphemeralContainers]", func() {
    f := framework.NewDefaultFramework("ephemeral-containers-tests")
    ginkgo.Context("", func() {
        ginkgo.BeforeEach(func() {
            // First, check if ephemeral containers feature is enabled
            // Note: this feature MUST be enabled on both kubelet and API server.
            //  Enabling on kubelet alone is not enough to create ephemeral containers.
            ginkgo.By("check if ehpemral containers feature is enabled")
            enabled, err := ephemeralContainersEnabled(f)
            framework.ExpectNoError(err)
            if !enabled {
                e2eskipper.Skipf("ephemeral containers not enabled")
            }
        })

        ginkgo.Context("", func() {

            var pod *v1.Pod

            ginkgo.BeforeEach(func() {
                pod = createPodWithoutEphemeralContainer(f)
            })

            ginkgo.AfterEach(func() {
                deletePodsSync(f, []*v1.Pod{pod})
            })

            ginkgo.It("should add and start ephemeral container to the pod", func() {
                ephemeralContainer := v1.EphemeralContainer{
                    EphemeralContainerCommon: v1.EphemeralContainerCommon{
                        Name:                     "ephemeral-1",
                        Image:                    busyboxImage,
                        Command:                  []string{"sh"},
                        TTY:                      true,
                        Stdin:                    true,
                        ImagePullPolicy:          v1.PullIfNotPresent,                        // field required for ephemeral containers
                        TerminationMessagePolicy: v1.TerminationMessageFallbackToLogsOnError, // field required for ephemeral containers
                    },
                }
                ginkgo.By("add ephemeral container to pod")

                ec, err := f.PodClient().GetEphemeralContainers(context.TODO(), pod.Name, metav1.GetOptions{})

                framework.ExpectNoError(err)

                ec.EphemeralContainers = []v1.EphemeralContainer{ephemeralContainer}
                _, err = f.PodClient().UpdateEphemeralContainers(context.TODO(), pod.Name, ec, metav1.UpdateOptions{})

                framework.ExpectNoError(err)

                ginkgo.By("wait for ephemeral container to be ready")
                gomega.Eventually(func() error {
                    pod, err := f.PodClient().Get(context.TODO(), pod.Name, metav1.GetOptions{})
                    if err != nil {
                        return err
                    }
                    statuses := pod.Status.EphemeralContainerStatuses
                    if len(statuses) != 1 {
                        return fmt.Errorf("unexpected multiple ContainerStatus: %v", statuses)
                    }
                    status := statuses[0]
                    if status.State.Running == nil {
                        return fmt.Errorf("ephemeral container not yet running")
                    }
                    return nil
                }, 30*time.Second, framework.Poll).Should(gomega.Succeed())
            })

        })

    })
})
