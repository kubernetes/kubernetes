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

	. "github.com/onsi/ginkgo"

	"k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/util/wait"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/client/clientset_generated/internalclientset"
	testutils "k8s.io/kubernetes/test/utils"
)

// RcByNamePort returns a ReplicationController with specified name and port
func RcByNamePort(name string, replicas int32, image string, port int, protocol v1.Protocol,
	labels map[string]string, gracePeriod *int64) *v1.ReplicationController {

	return RcByNameContainer(name, replicas, image, labels, v1.Container{
		Name:  name,
		Image: image,
		Ports: []v1.ContainerPort{{ContainerPort: int32(port), Protocol: protocol}},
	}, gracePeriod)
}

// RcByNameContainer returns a ReplicationControoler with specified name and container
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
			APIVersion: testapi.Groups[v1.GroupName].GroupVersion().String(),
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

// ScaleRCByLabels scales an RC via ns/label lookup. If replicas == 0 it waits till
// none are running, otherwise it does what a synchronous scale operation would do.
func ScaleRCByLabels(clientset clientset.Interface, internalClientset internalclientset.Interface, ns string, l map[string]string, replicas uint) error {
	listOpts := metav1.ListOptions{LabelSelector: labels.SelectorFromSet(labels.Set(l)).String()}
	rcs, err := clientset.Core().ReplicationControllers(ns).List(listOpts)
	if err != nil {
		return err
	}
	if len(rcs.Items) == 0 {
		return fmt.Errorf("RC with labels %v not found in ns %v", l, ns)
	}
	Logf("Scaling %v RCs with labels %v in ns %v to %v replicas.", len(rcs.Items), l, ns, replicas)
	for _, labelRC := range rcs.Items {
		name := labelRC.Name
		if err := ScaleRC(clientset, internalClientset, ns, name, replicas, false); err != nil {
			return err
		}
		rc, err := clientset.Core().ReplicationControllers(ns).Get(name, metav1.GetOptions{})
		if err != nil {
			return err
		}
		if replicas == 0 {
			ps, err := podStoreForSelector(clientset, rc.Namespace, labels.SelectorFromSet(rc.Spec.Selector))
			if err != nil {
				return err
			}
			defer ps.Stop()
			if err = waitForPodsGone(ps, 10*time.Second, 10*time.Minute); err != nil {
				return fmt.Errorf("error while waiting for pods gone %s: %v", name, err)
			}
		} else {
			if err := testutils.WaitForPodsWithLabelRunning(
				clientset, ns, labels.SelectorFromSet(labels.Set(rc.Spec.Selector))); err != nil {
				return err
			}
		}
	}
	return nil
}

type updateRcFunc func(d *v1.ReplicationController)

func UpdateReplicationControllerWithRetries(c clientset.Interface, namespace, name string, applyUpdate updateRcFunc) (*v1.ReplicationController, error) {
	var rc *v1.ReplicationController
	var updateErr error
	pollErr := wait.PollImmediate(10*time.Millisecond, 1*time.Minute, func() (bool, error) {
		var err error
		if rc, err = c.Core().ReplicationControllers(namespace).Get(name, metav1.GetOptions{}); err != nil {
			return false, err
		}
		// Apply the update, then attempt to push it to the apiserver.
		applyUpdate(rc)
		if rc, err = c.Core().ReplicationControllers(namespace).Update(rc); err == nil {
			Logf("Updating replication controller %q", name)
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

func DeleteRCAndPods(clientset clientset.Interface, internalClientset internalclientset.Interface, ns, name string) error {
	return DeleteResourceAndPods(clientset, internalClientset, api.Kind("ReplicationController"), ns, name)
}

func ScaleRC(clientset clientset.Interface, internalClientset internalclientset.Interface, ns, name string, size uint, wait bool) error {
	return ScaleResource(clientset, internalClientset, ns, name, size, wait, api.Kind("ReplicationController"))
}

func RunRC(config testutils.RCConfig) error {
	By(fmt.Sprintf("creating replication controller %s in namespace %s", config.Name, config.Namespace))
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
	return WaitForPodToDisappear(c, ns, podName, label, 20*time.Second, 10*time.Minute)
}

// WaitForReplicationController waits until the RC appears (exist == true), or disappears (exist == false)
func WaitForReplicationController(c clientset.Interface, namespace, name string, exist bool, interval, timeout time.Duration) error {
	err := wait.PollImmediate(interval, timeout, func() (bool, error) {
		_, err := c.Core().ReplicationControllers(namespace).Get(name, metav1.GetOptions{})
		if err != nil {
			Logf("Get ReplicationController %s in namespace %s failed (%v).", name, namespace, err)
			return !exist, nil
		} else {
			Logf("ReplicationController %s in namespace %s found.", name, namespace)
			return exist, nil
		}
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
		rcs, err := c.Core().ReplicationControllers(namespace).List(metav1.ListOptions{LabelSelector: selector.String()})
		switch {
		case len(rcs.Items) != 0:
			Logf("ReplicationController with %s in namespace %s found.", selector.String(), namespace)
			return exist, nil
		case len(rcs.Items) == 0:
			Logf("ReplicationController with %s in namespace %s disappeared.", selector.String(), namespace)
			return !exist, nil
		default:
			Logf("List ReplicationController with %s in namespace %s failed: %v", selector.String(), namespace, err)
			return false, nil
		}
	})
	if err != nil {
		stateMsg := map[bool]string{true: "to appear", false: "to disappear"}
		return fmt.Errorf("error waiting for ReplicationControllers with %s in namespace %s %s: %v", selector.String(), namespace, stateMsg[exist], err)
	}
	return nil
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

	getImageTemplate := fmt.Sprintf(`--template={{if (exists . "status" "containerStatuses")}}{{range .status.containerStatuses}}{{if eq .name "%s"}}{{.image}}{{end}}{{end}}{{end}}`, containername)

	By(fmt.Sprintf("waiting for all containers in %s pods to come up.", testname)) //testname should be selector
waitLoop:
	for start := time.Now(); time.Since(start) < PodStartTimeout; time.Sleep(5 * time.Second) {
		getPodsOutput := RunKubectlOrDie("get", "pods", "-o", "template", getPodsTemplate, "-l", testname, fmt.Sprintf("--namespace=%v", ns))
		pods := strings.Fields(getPodsOutput)
		if numPods := len(pods); numPods != replicas {
			By(fmt.Sprintf("Replicas for %s: expected=%d actual=%d", testname, replicas, numPods))
			continue
		}
		var runningPods []string
		for _, podID := range pods {
			running := RunKubectlOrDie("get", "pods", podID, "-o", "template", getContainerStateTemplate, fmt.Sprintf("--namespace=%v", ns))
			if running != "true" {
				Logf("%s is created but not running", podID)
				continue waitLoop
			}

			currentImage := RunKubectlOrDie("get", "pods", podID, "-o", "template", getImageTemplate, fmt.Sprintf("--namespace=%v", ns))
			if currentImage != containerImage {
				Logf("%s is created but running wrong image; expected: %s, actual: %s", podID, containerImage, currentImage)
				continue waitLoop
			}

			// Call the generic validator function here.
			// This might validate for example, that (1) getting a url works and (2) url is serving correct content.
			if err := validator(c, podID); err != nil {
				Logf("%s is running right image but validator function failed: %v", podID, err)
				continue waitLoop
			}

			Logf("%s is verified up and running", podID)
			runningPods = append(runningPods, podID)
		}
		// If we reach here, then all our checks passed.
		if len(runningPods) == replicas {
			return
		}
	}
	// Reaching here means that one of more checks failed multiple times.  Assuming its not a race condition, something is broken.
	Failf("Timed out after %v seconds waiting for %s pods to reach valid state", PodStartTimeout.Seconds(), testname)
}
