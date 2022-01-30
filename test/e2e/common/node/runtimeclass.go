/*
Copyright 2018 The Kubernetes Authors.

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

package node

import (
	"context"
	"fmt"
	"time"

	v1 "k8s.io/api/core/v1"
	nodev1 "k8s.io/api/node/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	types "k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/kubernetes/pkg/kubelet/events"
	runtimeclasstest "k8s.io/kubernetes/pkg/kubelet/runtimeclass/testing"
	"k8s.io/kubernetes/test/e2e/framework"
	e2eevents "k8s.io/kubernetes/test/e2e/framework/events"
	e2enode "k8s.io/kubernetes/test/e2e/framework/node"
	e2epod "k8s.io/kubernetes/test/e2e/framework/pod"
	e2eskipper "k8s.io/kubernetes/test/e2e/framework/skipper"
	admissionapi "k8s.io/pod-security-admission/api"

	"github.com/onsi/ginkgo"
)

var _ = SIGDescribe("RuntimeClass", func() {
	f := framework.NewDefaultFramework("runtimeclass")
	f.NamespacePodSecurityEnforceLevel = admissionapi.LevelBaseline

	ginkgo.It("should reject a Pod requesting a non-existent RuntimeClass [NodeFeature:RuntimeHandler]", func() {
		rcName := f.Namespace.Name + "-nonexistent"
		expectPodRejection(f, e2enode.NewRuntimeClassPod(rcName))
	})

	ginkgo.It("should reject a Pod requesting a RuntimeClass with an unconfigured handler [NodeFeature:RuntimeHandler]", func() {
		handler := f.Namespace.Name + "-handler"
		rcName := createRuntimeClass(f, "unconfigured-handler", handler)
		defer deleteRuntimeClass(f, rcName)
		pod := f.PodClient().Create(e2enode.NewRuntimeClassPod(rcName))
		eventSelector := fields.Set{
			"involvedObject.kind":      "Pod",
			"involvedObject.name":      pod.Name,
			"involvedObject.namespace": f.Namespace.Name,
			"reason":                   events.FailedCreatePodSandBox,
		}.AsSelector().String()
		// Events are unreliable, don't depend on the event. It's used only to speed up the test.
		err := e2eevents.WaitTimeoutForEvent(f.ClientSet, f.Namespace.Name, eventSelector, handler, framework.PodEventTimeout)
		if err != nil {
			framework.Logf("Warning: did not get event about FailedCreatePodSandBox. Err: %v", err)
		}
		// Check the pod is still not running
		p, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Get(context.TODO(), pod.Name, metav1.GetOptions{})
		framework.ExpectNoError(err, "could not re-read the pod after event (or timeout)")
		framework.ExpectEqual(p.Status.Phase, v1.PodPending, "Pod phase isn't pending")
	})

	// This test requires that the PreconfiguredRuntimeHandler has already been set up on nodes.
	ginkgo.It("should run a Pod requesting a RuntimeClass with a configured handler [NodeFeature:RuntimeHandler]", func() {
		// Requires special setup of test-handler which is only done in GCE kube-up environment
		// see https://github.com/kubernetes/kubernetes/blob/eb729620c522753bc7ae61fc2c7b7ea19d4aad2f/cluster/gce/gci/configure-helper.sh#L3069-L3076
		e2eskipper.SkipUnlessProviderIs("gce")

		// The built-in docker runtime does not support configuring runtime handlers.
		handler := e2enode.PreconfiguredRuntimeClassHandler(framework.TestContext.ContainerRuntime)

		rcName := createRuntimeClass(f, "preconfigured-handler", handler)
		defer deleteRuntimeClass(f, rcName)
		pod := f.PodClient().Create(e2enode.NewRuntimeClassPod(rcName))
		expectPodSuccess(f, pod)
	})

	ginkgo.It("should reject a Pod requesting a deleted RuntimeClass [NodeFeature:RuntimeHandler]", func() {
		rcName := createRuntimeClass(f, "delete-me", "runc")
		rcClient := f.ClientSet.NodeV1().RuntimeClasses()

		ginkgo.By("Deleting RuntimeClass "+rcName, func() {
			err := rcClient.Delete(context.TODO(), rcName, metav1.DeleteOptions{})
			framework.ExpectNoError(err, "failed to delete RuntimeClass %s", rcName)

			ginkgo.By("Waiting for the RuntimeClass to disappear")
			framework.ExpectNoError(wait.PollImmediate(framework.Poll, time.Minute, func() (bool, error) {
				_, err := rcClient.Get(context.TODO(), rcName, metav1.GetOptions{})
				if apierrors.IsNotFound(err) {
					return true, nil // done
				}
				if err != nil {
					return true, err // stop wait with error
				}
				return false, nil
			}))
		})

		expectPodRejection(f, e2enode.NewRuntimeClassPod(rcName))
	})

	/*
		Release: v1.20
		Testname: RuntimeClass API
		Description:
		The node.k8s.io API group MUST exist in the /apis discovery document.
		The node.k8s.io/v1 API group/version MUST exist in the /apis/mode.k8s.io discovery document.
		The runtimeclasses resource MUST exist in the /apis/node.k8s.io/v1 discovery document.
		The runtimeclasses resource must support create, get, list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt(" should support RuntimeClasses API operations", func() {
		// Setup
		rcVersion := "v1"
		rcClient := f.ClientSet.NodeV1().RuntimeClasses()

		// This is a conformance test that must configure opaque handlers to validate CRUD operations.
		// Test should not use any existing handler like gVisor or runc
		//
		// All CRUD operations in this test are limited to the objects with the label test=f.UniqueName
		rc := runtimeclasstest.NewRuntimeClass(f.UniqueName+"-handler", f.UniqueName+"-conformance-runtime-class")
		rc.SetLabels(map[string]string{"test": f.UniqueName})
		rc2 := runtimeclasstest.NewRuntimeClass(f.UniqueName+"-handler2", f.UniqueName+"-conformance-runtime-class2")
		rc2.SetLabels(map[string]string{"test": f.UniqueName})
		rc3 := runtimeclasstest.NewRuntimeClass(f.UniqueName+"-handler3", f.UniqueName+"-conformance-runtime-class3")
		rc3.SetLabels(map[string]string{"test": f.UniqueName})

		// Discovery

		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == nodev1.GroupName {
					for _, version := range group.Versions {
						if version.Version == rcVersion {
							found = true
							break
						}
					}
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected RuntimeClass API group/version, got %#v", discoveryGroups.Groups))
		}

		ginkgo.By("getting /apis/node.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/node.k8s.io").Do(context.TODO()).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == rcVersion {
					found = true
					break
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected RuntimeClass API version, got %#v", group.Versions))
		}

		ginkgo.By("getting /apis/node.k8s.io/" + rcVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(nodev1.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			found := false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "runtimeclasses":
					found = true
				}
			}
			framework.ExpectEqual(found, true, fmt.Sprintf("expected runtimeclasses, got %#v", resources.APIResources))
		}

		// Main resource create/read/update/watch operations

		ginkgo.By("creating")
		createdRC, err := rcClient.Create(context.TODO(), rc, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = rcClient.Create(context.TODO(), rc, metav1.CreateOptions{})
		framework.ExpectEqual(apierrors.IsAlreadyExists(err), true, fmt.Sprintf("expected 409, got %#v", err))
		_, err = rcClient.Create(context.TODO(), rc2, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("watching")
		framework.Logf("starting watch")
		rcWatch, err := rcClient.Watch(context.TODO(), metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)

		// added for a watch
		_, err = rcClient.Create(context.TODO(), rc3, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		gottenRC, err := rcClient.Get(context.TODO(), rc.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(gottenRC.UID, createdRC.UID)

		ginkgo.By("listing")
		rcs, err := rcClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(rcs.Items), 3, "filtered list should have 3 items")

		ginkgo.By("patching")
		patchedRC, err := rcClient.Patch(context.TODO(), createdRC.Name, types.MergePatchType, []byte(`{"metadata":{"annotations":{"patched":"true"}}}`), metav1.PatchOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(patchedRC.Annotations["patched"], "true", "patched object should have the applied annotation")

		ginkgo.By("updating")
		csrToUpdate := patchedRC.DeepCopy()
		csrToUpdate.Annotations["updated"] = "true"
		updatedRC, err := rcClient.Update(context.TODO(), csrToUpdate, metav1.UpdateOptions{})
		framework.ExpectNoError(err)
		framework.ExpectEqual(updatedRC.Annotations["updated"], "true", "updated object should have the applied annotation")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAdded, sawPatched, sawUpdated := false, false, false; !sawAdded && !sawPatched && !sawUpdated; {
			select {
			case evt, ok := <-rcWatch.ResultChan():
				framework.ExpectEqual(ok, true, "watch channel should not close")
				if evt.Type == watch.Modified {
					watchedRC, isRC := evt.Object.(*nodev1.RuntimeClass)
					framework.ExpectEqual(isRC, true, fmt.Sprintf("expected RC, got %T", evt.Object))
					if watchedRC.Annotations["patched"] == "true" {
						framework.Logf("saw patched annotations")
						sawPatched = true
					} else if watchedRC.Annotations["updated"] == "true" {
						framework.Logf("saw updated annotations")
						sawUpdated = true
					} else {
						framework.Logf("missing expected annotations, waiting: %#v", watchedRC.Annotations)
					}
				} else if evt.Type == watch.Added {
					_, isRC := evt.Object.(*nodev1.RuntimeClass)
					framework.ExpectEqual(isRC, true, fmt.Sprintf("expected RC, got %T", evt.Object))
					sawAdded = true
				}

			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}
		rcWatch.Stop()

		// main resource delete operations

		ginkgo.By("deleting")
		err = rcClient.Delete(context.TODO(), createdRC.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = rcClient.Get(context.TODO(), createdRC.Name, metav1.GetOptions{})
		framework.ExpectEqual(apierrors.IsNotFound(err), true, fmt.Sprintf("expected 404, got %#v", err))
		rcs, err = rcClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(rcs.Items), 2, "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = rcClient.DeleteCollection(context.TODO(), metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		rcs, err = rcClient.List(context.TODO(), metav1.ListOptions{LabelSelector: "test=" + f.UniqueName})
		framework.ExpectNoError(err)
		framework.ExpectEqual(len(rcs.Items), 0, "filtered list should have 0 items")
	})
})

func deleteRuntimeClass(f *framework.Framework, name string) {
	err := f.ClientSet.NodeV1().RuntimeClasses().Delete(context.TODO(), name, metav1.DeleteOptions{})
	framework.ExpectNoError(err, "failed to delete RuntimeClass resource")
}

// createRuntimeClass generates a RuntimeClass with the desired handler and a "namespaced" name,
// synchronously creates it, and returns the generated name.
func createRuntimeClass(f *framework.Framework, name, handler string) string {
	uniqueName := fmt.Sprintf("%s-%s", f.Namespace.Name, name)
	rc := runtimeclasstest.NewRuntimeClass(uniqueName, handler)
	rc, err := f.ClientSet.NodeV1().RuntimeClasses().Create(context.TODO(), rc, metav1.CreateOptions{})
	framework.ExpectNoError(err, "failed to create RuntimeClass resource")
	return rc.GetName()
}

func expectPodRejection(f *framework.Framework, pod *v1.Pod) {
	_, err := f.ClientSet.CoreV1().Pods(f.Namespace.Name).Create(context.TODO(), pod, metav1.CreateOptions{})
	framework.ExpectError(err, "should be forbidden")
	framework.ExpectEqual(apierrors.IsForbidden(err), true, "should be forbidden error")
}

// expectPodSuccess waits for the given pod to terminate successfully.
func expectPodSuccess(f *framework.Framework, pod *v1.Pod) {
	framework.ExpectNoError(e2epod.WaitForPodSuccessInNamespace(
		f.ClientSet, pod.Name, f.Namespace.Name))
}
