package e2e_node

import (
	"crypto/md5"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strings"
	"time"

	. "github.com/onsi/ginkgo"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/test/e2e/framework"
)

const (
	testPod                 = "checkpoint-test-pod"
	testCheckpoint          = "checkpoint-test"
	dockershimCheckpointDir = "/var/lib/dockershim/sandbox"
	gcTimeout               = 3 * time.Minute
	testCheckpointContent   = `{"version":"v1","name":"fluentd-gcp-v2.0-vmnqx","namespace":"kube-system","data":{},"checksum":1799154314}`
)

var _ = framework.KubeDescribe("Dockershim [Serial] [Disruptive]", func() {
	f := framework.NewDefaultFramework("dockerhism-checkpoint-test")

	It("should clean up pod sandbox checkpoint after pod deletion", func() {
		runPodCheckpointTest(f, func() {
			By("No-op manipulation")
		})
	})

	It("should remove dangling checkpoint file", func() {
		filename := fmt.Sprintf("%x", md5.Sum([]byte(fmt.Sprintf("%s/%s", testCheckpoint, f.Namespace.Name))))
		fullpath := fmt.Sprintf("%s/%s", dockershimCheckpointDir, filename)

		By(fmt.Sprintf("Write a file at %q", fullpath))
		err := writeFileAndSync(fullpath, []byte(testCheckpointContent))
		framework.ExpectNoError(err, "Failed to create file %q", fullpath)

		By("Check if file is removed")
		removed := false
		for start := time.Now(); time.Since(start) < gcTimeout; time.Sleep(10 * time.Second) {
			if _, err := os.Stat(fullpath); os.IsNotExist(err) {
				removed = true
				break
			}
		}
		if !removed {
			framework.Failf("Expect dangling checkpoint file to be removed")
		}
	})

	Context("When pod sandbox checkpoint is missing", func() {
		It("should complete pod sandbox clean up", func() {
			runPodCheckpointTest(f, func() {
				checkpoints := findCheckpoints(f.Namespace.Name)
				if len(checkpoints) == 0 {
					framework.Failf("No checkpoint for the pod was found")
				}
				By("Removing checkpoint of test pod")
				for _, filename := range checkpoints {
					filename = cleanUpOutput(filename)
					if len(filename) == 0 {
						continue
					}
					framework.Logf("Removing checkpiont %q", filename)
					_, err := exec.Command("sudo", "rm", filename).CombinedOutput()
					framework.ExpectNoError(err, "Failed to remove checkpoint file %q: %v", string(filename), err)
				}
			})
		})
	})

	Context("When pod containers are missing", func() {
		It("should complete pod sandbox clean up based on the information in sandbox checkpoint", func() {
			runPodCheckpointTest(f, func() {
				By("Gathering pod container ids")
				stdout, err := exec.Command("sudo", "docker", "ps").CombinedOutput()
				framework.ExpectNoError(err, "Failed to run docker ps: %v", err)

				ids := []string{}
				lines := strings.Split(string(stdout), "\n")
				for _, line := range lines[1:] {
					if strings.Contains(line, f.Namespace.Name) {
						ids = append(ids, strings.Fields(line)[0])
					}
				}

				By("Stop and remove pod containers")
				dockerStopCmd := append([]string{"docker", "stop"}, ids...)
				_, err = exec.Command("sudo", dockerStopCmd...).CombinedOutput()
				framework.ExpectNoError(err, "Failed to run command %v: %v", dockerStopCmd, err)
				dockerRmCmd := append([]string{"docker", "rm"}, ids...)
				_, err = exec.Command("sudo", dockerRmCmd...).CombinedOutput()
				framework.ExpectNoError(err, "Failed to run command %v: %v", dockerRmCmd, err)
			})
		})
	})

	Context("When checkpoint file is corrupted", func() {
		It("should complete pod sandbox clean up", func() {
			runPodCheckpointTest(f, func() {
				By("Corrupt checkpoint file")
				checkpoints := findCheckpoints(f.Namespace.Name)
				if len(checkpoints) == 0 {
					framework.Failf("No checkpoint for the pod was found")
				}
				for _, file := range checkpoints {
					f, err := os.OpenFile(file, os.O_WRONLY|os.O_APPEND, 0644)
					framework.ExpectNoError(err, "Failed to open file %q", file)
					_, err = f.WriteString("blabblab")
					framework.ExpectNoError(err, "Failed to write to file %q", file)
					f.Sync()
					f.Close()
				}
			})
		})
	})
})

func runPodCheckpointTest(f *framework.Framework, twist func()) {
	By(fmt.Sprintf("Creating test pod: %s", testPod))
	f.PodClient().CreateSync(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: testPod},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image: "gcr.io/google_containers/busybox:1.24",
					Name:  "checkpoint-test-container",
					Command: []string{
						"sh",
						"-c", // Make 100 billion small files (more than we have inodes)
						"sleep 36000",
					},
				},
			},
		},
	})

	By("Performing disruptive operations")
	twist()

	By("Sanity check on test pod")
	f.WaitForPodRunning(testPod)

	By("Remove test pod")
	f.PodClient().DeleteSync(testPod, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

	By("Waiting for checkpoint to be removed")
	removed := false
	for start := time.Now(); time.Since(start) < gcTimeout; time.Sleep(10 * time.Second) {
		checkpoints := findCheckpoints(f.Namespace.Name)
		if len(checkpoints) == 0 {
			removed = true
			break
		}
		framework.Logf("Checkpoint of testPod still exists: %v", checkpoints)
	}
	if !removed {
		framework.Failf("Failed to observe checkpoint being removed within timeout.")
	}
}

func cleanUpOutput(output string) string {
	processed := strings.TrimSpace(string(output))
	regex := regexp.MustCompile(`\r?\n`)
	processed = regex.ReplaceAllString(processed, "")
	return processed
}

func writeFileAndSync(path string, data []byte) error {
	f, err := os.OpenFile(path, os.O_WRONLY|os.O_CREATE|os.O_TRUNC, 0644)
	if err != nil {
		return err
	}
	_, err = f.Write(data)
	if err != nil {
		return err
	}
	f.Sync()
	if err1 := f.Close(); err == nil {
		err = err1
	}
	return err
}

// findCheckpoints returns all checkpoint files containing input string
func findCheckpoints(match string) []string {
	By(fmt.Sprintf("Search checkpoints containing %q", match))
	checkpoints := []string{}
	stdout, err := exec.Command("sudo", "grep", "-rl", match, dockershimCheckpointDir).CombinedOutput()
	if err != nil {
		framework.Logf("grep from dockershim checkpoint directory returns error: %v", err)
	}
	if stdout == nil {
		return checkpoints
	}
	files := strings.Split(string(stdout), "\n")
	for _, file := range files {
		cleaned := cleanUpOutput(file)
		if len(cleaned) == 0 {
			continue
		}
		checkpoints = append(checkpoints, cleaned)
	}
	return checkpoints
}
