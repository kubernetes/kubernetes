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

package e2e_node

import (
	"crypto/md5"
	"fmt"
	"os"
	"os/exec"
	"path"
	"regexp"
	"strings"
	"time"

	"github.com/onsi/ginkgo"
	"github.com/onsi/gomega"
	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/util/uuid"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/kubernetes/test/e2e/framework"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	imageutils "k8s.io/kubernetes/test/utils/image"
)

const (
	testCheckpoint = "checkpoint-test"
	// Container GC Period is 1 minute
	gcTimeout             = 3 * time.Minute
	testCheckpointContent = `{"version":"v1","name":"fluentd-gcp-v2.0-vmnqx","namespace":"kube-system","data":{},"checksum":1799154314}`
)

var _ = SIGDescribe("Dockershim [Serial] [Disruptive] [Feature:Docker][Legacy:Docker]", func() {
	f := framework.NewDefaultFramework("dockerhism-checkpoint-test")

	ginkgo.BeforeEach(func() {
		framework.RunIfContainerRuntimeIs("docker")
	})

	ginkgo.It("should clean up pod sandbox checkpoint after pod deletion", func() {
		podName := "pod-checkpoint-no-disrupt"
		runPodCheckpointTest(f, podName, func() {
			checkpoints := findCheckpoints(podName)
			if len(checkpoints) == 0 {
				e2elog.Failf("No checkpoint for the pod was found")
			}
		})
	})

	ginkgo.It("should remove dangling checkpoint file", func() {
		filename := fmt.Sprintf("%x", md5.Sum([]byte(fmt.Sprintf("%s/%s", testCheckpoint, f.Namespace.Name))))
		fullpath := path.Join(framework.TestContext.DockershimCheckpointDir, filename)

		ginkgo.By(fmt.Sprintf("Write a file at %q", fullpath))
		err := writeFileAndSync(fullpath, []byte(testCheckpointContent))
		framework.ExpectNoError(err, "Failed to create file %q", fullpath)

		ginkgo.By("Check if file is removed")
		gomega.Eventually(func() bool {
			if _, err := os.Stat(fullpath); os.IsNotExist(err) {
				return true
			}
			return false
		}, gcTimeout, 10*time.Second).Should(gomega.BeTrue())

	})

	ginkgo.Context("When pod sandbox checkpoint is missing", func() {
		ginkgo.It("should complete pod sandbox clean up", func() {
			podName := "pod-checkpoint-missing"
			runPodCheckpointTest(f, podName, func() {
				checkpoints := findCheckpoints(podName)
				if len(checkpoints) == 0 {
					e2elog.Failf("No checkpoint for the pod was found")
				}
				ginkgo.By("Removing checkpoint of test pod")
				for _, filename := range checkpoints {
					if len(filename) == 0 {
						continue
					}
					e2elog.Logf("Removing checkpoint %q", filename)
					_, err := exec.Command("sudo", "rm", filename).CombinedOutput()
					framework.ExpectNoError(err, "Failed to remove checkpoint file %q: %v", string(filename), err)
				}
			})
		})
	})

	ginkgo.Context("When all containers in pod are missing", func() {
		ginkgo.It("should complete pod sandbox clean up based on the information in sandbox checkpoint", func() {
			runPodCheckpointTest(f, "pod-containers-missing", func() {
				ginkgo.By("Gathering pod container ids")
				stdout, err := exec.Command("sudo", "docker", "ps", "-q", "-f",
					fmt.Sprintf("name=%s", f.Namespace.Name)).CombinedOutput()
				framework.ExpectNoError(err, "Failed to run docker ps: %v", err)
				lines := strings.Split(string(stdout), "\n")
				ids := []string{}
				for _, id := range lines {
					id = cleanString(id)
					if len(id) > 0 {
						ids = append(ids, id)
					}
				}

				ginkgo.By("Stop and remove pod containers")
				dockerStopCmd := append([]string{"docker", "stop"}, ids...)
				_, err = exec.Command("sudo", dockerStopCmd...).CombinedOutput()
				framework.ExpectNoError(err, "Failed to run command %v: %v", dockerStopCmd, err)
				dockerRmCmd := append([]string{"docker", "rm"}, ids...)
				_, err = exec.Command("sudo", dockerRmCmd...).CombinedOutput()
				framework.ExpectNoError(err, "Failed to run command %v: %v", dockerRmCmd, err)
			})
		})
	})

	ginkgo.Context("When checkpoint file is corrupted", func() {
		ginkgo.It("should complete pod sandbox clean up", func() {
			podName := "pod-checkpoint-corrupted"
			runPodCheckpointTest(f, podName, func() {
				ginkgo.By("Corrupt checkpoint file")
				checkpoints := findCheckpoints(podName)
				if len(checkpoints) == 0 {
					e2elog.Failf("No checkpoint for the pod was found")
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

func runPodCheckpointTest(f *framework.Framework, podName string, twist func()) {
	podName = podName + string(uuid.NewUUID())
	ginkgo.By(fmt.Sprintf("Creating test pod: %s", podName))
	f.PodClient().CreateSync(&v1.Pod{
		ObjectMeta: metav1.ObjectMeta{Name: podName},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{
					Image: imageutils.GetPauseImageName(),
					Name:  "pause-container",
				},
			},
		},
	})

	ginkgo.By("Performing disruptive operations")
	twist()

	ginkgo.By("Remove test pod")
	f.PodClient().DeleteSync(podName, &metav1.DeleteOptions{}, framework.DefaultPodDeletionTimeout)

	ginkgo.By("Waiting for checkpoint to be removed")
	if err := wait.PollImmediate(10*time.Second, gcTimeout, func() (bool, error) {
		checkpoints := findCheckpoints(podName)
		if len(checkpoints) == 0 {
			return true, nil
		}
		e2elog.Logf("Checkpoint of %q still exists: %v", podName, checkpoints)
		return false, nil
	}); err != nil {
		e2elog.Failf("Failed to observe checkpoint being removed within timeout: %v", err)
	}
}

// cleanString cleans up any trailing spaces and new line character for the input string
func cleanString(output string) string {
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
	ginkgo.By(fmt.Sprintf("Search checkpoints containing %q", match))
	checkpoints := []string{}
	stdout, err := exec.Command("sudo", "grep", "-rl", match, framework.TestContext.DockershimCheckpointDir).CombinedOutput()
	if err != nil {
		e2elog.Logf("grep from dockershim checkpoint directory returns error: %v", err)
	}
	if stdout == nil {
		return checkpoints
	}
	files := strings.Split(string(stdout), "\n")
	for _, file := range files {
		cleaned := cleanString(file)
		if len(cleaned) == 0 {
			continue
		}
		checkpoints = append(checkpoints, cleaned)
	}
	return checkpoints
}
