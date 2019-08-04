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

package framework

import (
	"fmt"
	"strings"
	"time"

	"github.com/onsi/ginkgo"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	scaleclient "k8s.io/client-go/scale"
	api "k8s.io/kubernetes/pkg/apis/core"
	e2elog "k8s.io/kubernetes/test/e2e/framework/log"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	testutils "k8s.io/kubernetes/test/utils"
)

// RcByNamePort returns a ReplicationController with specified name and port
func RcByNamePort(name string, replicas int32, image string, containerArgs []string, port int, protocol v1.Protocol,
	labels map[string]string, gracePeriod *int64) *v1.ReplicationController {

	return RcByNameContainer(name, replicas, image, labels, v1.Container{
		Name:  name,
		Image: image,
		Args:  containerArgs,
		Ports: []v1.ContainerPort{{ContainerPort: int32(port), Protocol: protocol}},
	}, gracePeriod)
}

// RcByNameContainer returns a ReplicationController with specified name and container
func RcByNameContainer(name string, replicas int32, image string, labels map[string]string, c v1.Container,
	gracePeriod *int64) *v1.ReplicationController {

	zeroGracePeriod := int64(0)

	// Add "name": name to the labels, overwriting if it exists.
	labels["name"] = name
	if gracePeriod == nil {
		gracePeriod = &zeroGracePeriod
	}
	return &v1.ReplicationController{
		TypeMeta: metav1.TypeMeta{
			Kind:       "ReplicationController",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name: name,
		},
		Spec: v1.ReplicationControllerSpec{
			Replicas: func(i int32) *int32 { return &i }(replicas),
			Selector: map[string]string{
				"name": name,
			},
			Template: &v1.PodTemplateSpec{
				ObjectMeta: metav1.ObjectMeta{
					Labels: labels,
				},
				Spec: v1.PodSpec{
					Containers:                    []v1.Container{c},
					TerminationGracePeriodSeconds: gracePeriod,
				},
			},
		},
	}
}

type updateRcFunc func(d *v1.ReplicationController)

// UpdateReplicationControllerWithRetries retries updating the given rc on conflict with the following steps:
// 1. Get latest resource
// 2. applyUpdate
// 3. Update the resource
func UpdateReplicationControllerWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateRcFunc) (*v1.ReplicationController, error) {
	var rc *v1.ReplicationController
	var updateErr error
	pollErr := wait.PollImmediate(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		var err error
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, err = c.CoreV1().ReplicationControllers(namespace).Update(rc); err == nil {
			e2elog.Logf("Updating replication controller %q", name)
			return true, nil
		}
		updateErr = err
		return false, nil
	})
	if pollErr == wait.ErrWaitTimeout {
		pollErr = fmt.Errorf("couldn't apply the provided updated to rc %q: %v", name, updateErr)
	}
	return rc, pollErr
}

// DeleteRCAndWaitForGC deletes only the Replication Controller and waits for GC to delete the pods.
func DeleteRCAndWaitForGC(c clientset.Interface, ns, name string) error {
	return DeleteResourceAndWaitForGC(c, api.Kind("ReplicationController"), ns, name)
}

// ScaleRC scales Replication Controller to be desired size.
func ScaleRC(clientset clientset.Interface, scalesGetter scaleclient.ScalesGetter, ns, name string, size uint, wait bool) error {
	return ScaleResource(clientset, scalesGetter, ns, name, size, wait, api.Kind("ReplicationController"), api.Resource("replicationcontrollers"))
}

// RunRC Launches (and verifies correctness) of a Replication Controller
// and will wait for all pods it spawns to become "Running".
func RunRC(config testutils.RCConfig) error {
	ginkgo.By(fmt.Sprintf("creating replication controller %s in namespace %s", config.Name, config.Namespace))
	config.NodeDumpFunc = DumpNodeDebugInfo
	config.ContainerDumpFunc = LogFailedContainers
	return testutils.RunRC(config)
}

// WaitForRCPodToDisappear returns nil if the pod from the given replication controller (described by rcName) no longer exists.
// In case of failure or too long waiting time, an error is returned.
func WaitForRCPodToDisappear(c clientset.Interface, ns, rcName, podName string) error {
	label := labels.SelectorFromSet(labels.Set(map[string]string{"name": rcName}))
	// NodeController evicts pod after 5 minutes, so we need timeout greater than that to observe effects.
	// The grace period must be set to 0 on the pod for it to be deleted during the partition.
	// Otherwise, it goes to the 'Terminating' state till the kubelet confirms deletion.
	return e2epod.WaitForPodToDisappear(c, ns, podName, label, 20*time.Second, 10*time.Minute)
}

// WaitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func WaitForReplicationController(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.CoreV1().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			e2elog.Logf("Get ReplicationController %s in namespace %s failed (%v).", name, namespace, err)
			return !exist, nil
		}
		e2elog.Logf("ReplicationController %s in namespace %s found.", name, namespace)
		return exist, nil
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationController %s/%s %s: %v", namespace, name, stateMsg[exist], err)
	}
	return nil
}

// WaitForReplicationControllerwithSelector waits until any RC with given selector appears (exist == true), or disappears (exist == false)
func WaitForReplicationControllerwithSelector(c clientset.Interface, namespace string, selector labels.Selector, exist bool, interval,
	timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		rcs, err := c.CoreV1().ReplicationControllers(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		switch {
		case len(rcs.Items) != 0:
			e2elog.Logf("ReplicationController with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(rcs.Items) == 0:
			e2elog.Logf("ReplicationController with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		default:
			e2elog.Logf("List ReplicationController with %s in namespace %s failed: %v", selector.String(), namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationControllers with %s in namespace %s %s: %v", selector.String(), namespace, stateMsg[exist], err)
	}
	return nil
}

// trimDockerRegistry is the function for trimming the docker.io/library from the beginning of the imagename.
// If community docker installed it will not prefix the registry names with the dockerimages vs registry names prefixed with other runtimes or docker installed via RHEL extra repo.
// So this function will help to trim the docker.io/library if exists
func trimDockerRegistry(imagename string) string {
	imagename = strings.Replace(imagename, "docker.io/", "", 1)
	return strings.Replace(imagename, "library/", "", 1)
}

// validatorFn is the function which is individual tests will implement.
// we may want it to return more than just an error, at some point.
type validatorFn func(c clientset.Interface, podID string) error

// ValidateController is a generic mechanism for testing RC's that are running.
// It takes a container name, a test name, and a validator function which is plugged in by a specific test.
// "containername": this is grepped for.
// "containerImage" : this is the name of the image we expect to be launched.  Not to confuse w/ images (kitten.jpg)  which are validated.
// "testname":  which gets bubbled up to the logging/failure messages if errors happen.
// "validator" function: This function is given a podID and a client, and it can do some specific validations that way.
func ValidateController(c clientset.Interface, containerImage string, replicas int, containername string, testname string, validator validatorFn, ns string) {
	containerImage = trimDockerRegistry(containerImage)
	getPodsTemplate := "--template={{range.items}}{{.metadata.name}} {{end}}"
	// NB: kubectl adds the "exists" function to the standard template functions.
	// This lets us check to see if the "running" entry exists for each of the containers
	// we care about. Exists will never return an error and it's safe to check a chain of
	// things, any one of which may not exist. In the below template, all of info,
	// containername, and running might be nil, so the normal index function isn't very
	// helpful.
	// This template is unit-tested in kubectl, so if you change it, update the unit test.
	// You can read about the syntax here: http://golang.org/pkg/text/template/.
	getContainerStateTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if (and (eq .name "%s") (exists . "state" "running"))}}true{{end}}{{end}}{{end}}`, containername)

	getImageTemplate := fmt.Sprintf(`--template={{if (exists . "spec" "containers")}}{{range .spec.containers}}{{if eq .name "%s"}}{{.image}}{{end}}{{end}}{{end}}`, containername)

	ginkgo.By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
waitLoop:
	for start := time.Now(); time.Since(start) < PodStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := RunKubectlOrDie("get", "pods", "-o", "template", getPodsTemplate, "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			ginkgo.By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := RunKubectlOrDie("get", "pods", podID, "-o", "template", getContainerStateTemplate, fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				e2elog.Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := RunKubectlOrDie("get", "pods", podID, "-o", "template", getImageTemplate, fmt.Sprintf("--namespace=%v", ns))
			currentImage = trimDockerRegistry(currentImage)
			if currentImage != containerImage {
				e2elog.Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, containerImage, currentImage)
				continue waitLoop
			}

			// Call the generic validator function here.
			// This might validate for example, that (1) getting a url works and (2) url is serving correct content.
			if err := validator(c, podID); err != nil {
				e2elog.Logf("%s is running right image but validator function failed: %v", podID, err)
				continue waitLoop
			}

			e2elog.Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		// If we reach here, then all our checks passed.
		if len(runningPods) == replicas {
			return
		}
	}
	// Reaching here means that one of more checks failed multiple times.  Assuming its not a race condition, something is broken.
	e2elog.Failf("Timed out after %v seconds waiting for %s pods to reach valid state", PodStartTimeout.Seconds(), testname)
}
