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

package apimachinery

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"net/http"
	"sync"
	"sync/atomic"
	"time"

	"github.com/onsi/ginkgo/v2"
	"github.com/onsi/gomega"
	"github.com/prometheus/common/expfmt"
	"github.com/prometheus/common/model"

	flowcontrol "k8s.io/api/flowcontrol/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	utilrand "k8s.io/apimachinery/pkg/util/rand"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apimachinery/pkg/watch"
	"k8s.io/apiserver/pkg/util/apihelpers"
	clientset "k8s.io/client-go/kubernetes"
	"k8s.io/client-go/rest"
	clientsideflowcontrol "k8s.io/client-go/util/flowcontrol"
	"k8s.io/client-go/util/retry"
	apimachineryutils "k8s.io/kubernetes/test/e2e/common/apimachinery"
	"k8s.io/kubernetes/test/e2e/framework"
	admissionapi "k8s.io/pod-security-admission/api"
	"k8s.io/utils/ptr"
)

const (
	nominalConcurrencyLimitMetricName = "apiserver_flowcontrol_nominal_limit_seats"
	priorityLevelLabelName            = "priority_level"
)

var (
	errPriorityLevelNotFound = errors.New("cannot find a metric sample with a matching priority level name label")
)

var _ = SIGDescribe("API priority and fairness", func() {
	f := framework.NewDefaultFramework("apf")
	f.NamespacePodSecurityLevel = admissionapi.LevelPrivileged

	ginkgo.It("should ensure that requests can be classified by adding FlowSchema and PriorityLevelConfiguration", func(ctx context.Context) {
		testingFlowSchemaName := "e2e-testing-flowschema"
		testingPriorityLevelName := "e2e-testing-prioritylevel"
		matchingUsername := "noxu"
		nonMatchingUsername := "foo"

		ginkgo.By("creating a testing PriorityLevelConfiguration object")
		createdPriorityLevel := createPriorityLevel(ctx, f, testingPriorityLevelName, 1)

		ginkgo.By("creating a testing FlowSchema object")
		createdFlowSchema := createFlowSchema(ctx, f, testingFlowSchemaName, 1000, testingPriorityLevelName, []string{matchingUsername})

		ginkgo.By("waiting for testing FlowSchema and PriorityLevelConfiguration to reach steady state")
		waitForSteadyState(ctx, f, testingFlowSchemaName, testingPriorityLevelName)

		var response *http.Response
		ginkgo.By("response headers should contain the UID of the appropriate FlowSchema and PriorityLevelConfiguration for a matching user")
		response = makeRequest(f, matchingUsername)
		if plUIDWant, plUIDGot := string(createdPriorityLevel.UID), getPriorityLevelUID(response); plUIDWant != plUIDGot {
			framework.Failf("expected PriorityLevelConfiguration UID in the response header: %s, but got: %s, response header: %#v", plUIDWant, plUIDGot, response.Header)
		}
		if fsUIDWant, fsUIDGot := string(createdFlowSchema.UID), getFlowSchemaUID(response); fsUIDWant != fsUIDGot {
			framework.Failf("expected FlowSchema UID in the response header: %s, but got: %s, response header: %#v", fsUIDWant, fsUIDGot, response.Header)
		}

		ginkgo.By("response headers should contain non-empty UID of FlowSchema and PriorityLevelConfiguration for a non-matching user")
		response = makeRequest(f, nonMatchingUsername)
		if plUIDGot := getPriorityLevelUID(response); plUIDGot == "" {
			framework.Failf("expected a non-empty PriorityLevelConfiguration UID in the response header, but got: %s, response header: %#v", plUIDGot, response.Header)
		}
		if fsUIDGot := getFlowSchemaUID(response); fsUIDGot == "" {
			framework.Failf("expected a non-empty FlowSchema UID in the response header but got: %s, response header: %#v", fsUIDGot, response.Header)
		}
	})

	// This test creates two flow schemas and a corresponding priority level for
	// each flow schema. One flow schema has a higher match precedence. With two
	// clients making requests at different rates, we test to make sure that the
	// higher QPS client cannot drown out the other one despite having higher
	// priority.
	ginkgo.It("should ensure that requests can't be drowned out (priority)", func(ctx context.Context) {
		// See https://github.com/kubernetes/kubernetes/issues/96710
		ginkgo.Skip("skipping test until flakiness is resolved")

		flowSchemaNamePrefix := "e2e-testing-flowschema-" + f.UniqueName
		priorityLevelNamePrefix := "e2e-testing-prioritylevel-" + f.UniqueName
		loadDuration := 10 * time.Second
		highQPSClientName := "highqps-" + f.UniqueName
		lowQPSClientName := "lowqps-" + f.UniqueName

		type client struct {
			username                    string
			qps                         float64
			priorityLevelName           string  //lint:ignore U1000 field is actually used
			concurrencyMultiplier       float64 //lint:ignore U1000 field is actually used
			concurrency                 int32
			flowSchemaName              string //lint:ignore U1000 field is actually used
			matchingPrecedence          int32  //lint:ignore U1000 field is actually used
			completedRequests           int32
			expectedCompletedPercentage float64 //lint:ignore U1000 field is actually used
		}
		clients := []client{
			// "highqps" refers to a client that creates requests at a much higher
			// QPS than its counter-part and well above its concurrency share limit.
			// In contrast, "lowqps" stays under its concurrency shares.
			// Additionally, the "highqps" client also has a higher matching
			// precedence for its flow schema.
			{username: highQPSClientName, qps: 90, concurrencyMultiplier: 2.0, matchingPrecedence: 999, expectedCompletedPercentage: 0.90},
			{username: lowQPSClientName, qps: 4, concurrencyMultiplier: 0.5, matchingPrecedence: 1000, expectedCompletedPercentage: 0.90},
		}

		ginkgo.By("creating test priority levels and flow schemas")
		for i := range clients {
			clients[i].priorityLevelName = fmt.Sprintf("%s-%s", priorityLevelNamePrefix, clients[i].username)
			framework.Logf("creating PriorityLevel %q", clients[i].priorityLevelName)
			createPriorityLevel(ctx, f, clients[i].priorityLevelName, 1)

			clients[i].flowSchemaName = fmt.Sprintf("%s-%s", flowSchemaNamePrefix, clients[i].username)
			framework.Logf("creating FlowSchema %q", clients[i].flowSchemaName)
			createFlowSchema(ctx, f, clients[i].flowSchemaName, clients[i].matchingPrecedence, clients[i].priorityLevelName, []string{clients[i].username})

			ginkgo.By("waiting for testing FlowSchema and PriorityLevelConfiguration to reach steady state")
			waitForSteadyState(ctx, f, clients[i].flowSchemaName, clients[i].priorityLevelName)
		}

		ginkgo.By("getting request concurrency from metrics")
		for i := range clients {
			realConcurrency, err := getPriorityLevelNominalConcurrency(ctx, f.ClientSet, clients[i].priorityLevelName)
			framework.ExpectNoError(err)
			clients[i].concurrency = int32(float64(realConcurrency) * clients[i].concurrencyMultiplier)
			if clients[i].concurrency < 1 {
				clients[i].concurrency = 1
			}
			framework.Logf("request concurrency for %q will be %d (that is %d times client multiplier)", clients[i].username, clients[i].concurrency, realConcurrency)
		}

		ginkgo.By(fmt.Sprintf("starting uniform QPS load for %s", loadDuration.String()))
		var wg sync.WaitGroup
		for i := range clients {
			wg.Add(1)
			go func(c *client) {
				defer wg.Done()
				framework.Logf("starting uniform QPS load for %q: concurrency=%d, qps=%.1f", c.username, c.concurrency, c.qps)
				c.completedRequests = uniformQPSLoadConcurrent(f, c.username, c.concurrency, c.qps, loadDuration)
			}(&clients[i])
		}
		wg.Wait()

		ginkgo.By("checking completed requests with expected values")
		for _, client := range clients {
			// Each client should have 95% of its ideal number of completed requests.
			maxCompletedRequests := float64(client.concurrency) * client.qps * loadDuration.Seconds()
			fractionCompleted := float64(client.completedRequests) / maxCompletedRequests
			framework.Logf("client %q completed %d/%d requests (%.1f%%)", client.username, client.completedRequests, int32(maxCompletedRequests), 100*fractionCompleted)
			if fractionCompleted < client.expectedCompletedPercentage {
				framework.Failf("client %q: got %.1f%% completed requests, want at least %.1f%%", client.username, 100*fractionCompleted, 100*client.expectedCompletedPercentage)
			}
		}
	})

	// This test has two clients (different usernames) making requests at
	// different rates. Both clients' requests get mapped to the same flow schema
	// and priority level. We expect APF's "ByUser" flow distinguisher to isolate
	// the two clients and not allow one client to drown out the other despite
	// having a higher QPS.
	ginkgo.It("should ensure that requests can't be drowned out (fairness)", func(ctx context.Context) {
		// See https://github.com/kubernetes/kubernetes/issues/96710
		ginkgo.Skip("skipping test until flakiness is resolved")

		priorityLevelName := "e2e-testing-prioritylevel-" + f.UniqueName
		flowSchemaName := "e2e-testing-flowschema-" + f.UniqueName
		loadDuration := 10 * time.Second

		framework.Logf("creating PriorityLevel %q", priorityLevelName)
		createPriorityLevel(ctx, f, priorityLevelName, 1)

		highQPSClientName := "highqps-" + f.UniqueName
		lowQPSClientName := "lowqps-" + f.UniqueName
		framework.Logf("creating FlowSchema %q", flowSchemaName)
		createFlowSchema(ctx, f, flowSchemaName, 1000, priorityLevelName, []string{highQPSClientName, lowQPSClientName})

		ginkgo.By("waiting for testing flow schema and priority level to reach steady state")
		waitForSteadyState(ctx, f, flowSchemaName, priorityLevelName)

		type client struct {
			username                    string
			qps                         float64
			concurrencyMultiplier       float64 //lint:ignore U1000 field is actually used
			concurrency                 int32
			completedRequests           int32
			expectedCompletedPercentage float64 //lint:ignore U1000 field is actually used
		}
		clients := []client{
			{username: highQPSClientName, qps: 90, concurrencyMultiplier: 2.0, expectedCompletedPercentage: 0.90},
			{username: lowQPSClientName, qps: 4, concurrencyMultiplier: 0.5, expectedCompletedPercentage: 0.90},
		}

		framework.Logf("getting real concurrency")
		realConcurrency, err := getPriorityLevelNominalConcurrency(ctx, f.ClientSet, priorityLevelName)
		framework.ExpectNoError(err)
		for i := range clients {
			clients[i].concurrency = int32(float64(realConcurrency) * clients[i].concurrencyMultiplier)
			if clients[i].concurrency < 1 {
				clients[i].concurrency = 1
			}
			framework.Logf("request concurrency for %q will be %d", clients[i].username, clients[i].concurrency)
		}

		ginkgo.By(fmt.Sprintf("starting uniform QPS load for %s", loadDuration.String()))
		var wg sync.WaitGroup
		for i := range clients {
			wg.Add(1)
			go func(c *client) {
				defer wg.Done()
				framework.Logf("starting uniform QPS load for %q: concurrency=%d, qps=%.1f", c.username, c.concurrency, c.qps)
				c.completedRequests = uniformQPSLoadConcurrent(f, c.username, c.concurrency, c.qps, loadDuration)
			}(&clients[i])
		}
		wg.Wait()

		ginkgo.By("checking completed requests with expected values")
		for _, client := range clients {
			// Each client should have 95% of its ideal number of completed requests.
			maxCompletedRequests := float64(client.concurrency) * client.qps * float64(loadDuration/time.Second)
			fractionCompleted := float64(client.completedRequests) / maxCompletedRequests
			framework.Logf("client %q completed %d/%d requests (%.1f%%)", client.username, client.completedRequests, int32(maxCompletedRequests), 100*fractionCompleted)
			if fractionCompleted < client.expectedCompletedPercentage {
				framework.Failf("client %q: got %.1f%% completed requests, want at least %.1f%%", client.username, 100*fractionCompleted, 100*client.expectedCompletedPercentage)
			}
		}
	})

	/*
	   Release: v1.29
	   Testname: Priority and Fairness FlowSchema API
	   Description:
	   The flowcontrol.apiserver.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The flowcontrol.apiserver.k8s.io/v1 API group/version MUST exist
	     in the /apis/flowcontrol.apiserver.k8s.io discovery document.
	   The flowschemas and flowschemas/status resources MUST exist
	     in the /apis/flowcontrol.apiserver.k8s.io/v1 discovery document.
	   The flowschema resource must support create, get, list, watch,
	     update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support FlowSchema API operations", func(ctx context.Context) {
		fsVersion := "v1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == flowcontrol.GroupName {
					for _, version := range group.Versions {
						if version.Version == fsVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected flowcontrol API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/flowcontrol.apiserver.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/flowcontrol.apiserver.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == fsVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected flowschemas API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/flowcontrol.apiserver.k8s.io/" + fsVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(flowcontrol.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundFS, foundFSStatus := false, false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "flowschemas":
					foundFS = true
				case "flowschemas/status":
					foundFSStatus = true
				}
			}
			if !foundFS {
				framework.Failf("expected flowschemas, got %#v", resources.APIResources)
			}
			if !foundFSStatus {
				framework.Failf("expected flowschemas/status, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.FlowcontrolV1().FlowSchemas()
		labelKey, labelValue := "example-e2e-fs-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &flowcontrol.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-fs-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: flowcontrol.FlowSchemaSpec{
				MatchingPrecedence: 10000,
				PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
					Name: "global-default",
				},
				DistinguisherMethod: &flowcontrol.FlowDistinguisherMethod{
					Type: flowcontrol.FlowDistinguisherMethodByUserType,
				},
				Rules: []flowcontrol.PolicyRulesWithSubjects{
					{
						Subjects: []flowcontrol.Subject{
							{
								Kind: flowcontrol.SubjectKindUser,
								User: &flowcontrol.UserSubject{
									Name: "example-e2e-non-existent-user",
								},
							},
						},
						NonResourceRules: []flowcontrol.NonResourcePolicyRule{
							{
								Verbs:           []string{flowcontrol.VerbAll},
								NonResourceURLs: []string{flowcontrol.NonResourceAll},
							},
						},
					},
				},
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		fsCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		fsRead, err := client.Get(ctx, fsCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(fsRead.UID).To(gomega.Equal(fsCreated.UID))
		gomega.Expect(fsRead).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		fsWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"matchingPrecedence":9999}}`)
		fsPatched, err := client.Patch(ctx, fsCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(fsPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(fsPatched.Spec.MatchingPrecedence).To(gomega.Equal(int32(9999)), "patched object should have the applied spec")
		gomega.Expect(resourceversion.CompareResourceVersion(fsCreated.ResourceVersion, fsPatched.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var fsUpdated *flowcontrol.FlowSchema
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			fs, err := client.Get(ctx, fsCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			fsToUpdate := fs.DeepCopy()
			fsToUpdate.Annotations["updated"] = "true"
			fsToUpdate.Spec.MatchingPrecedence = int32(9000)

			fsUpdated, err = client.Update(ctx, fsToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update flowschema %q", fsCreated.Name)
		gomega.Expect(fsUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(fsUpdated.Spec.MatchingPrecedence).To(gomega.Equal(int32(9000)), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-fsWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				fsWatched, isFS := evt.Object.(*flowcontrol.FlowSchema)
				if !isFS {
					framework.Failf("expected an object of type: %T, but got %T", &flowcontrol.FlowSchema{}, evt.Object)
				}
				if fsWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					fsWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", fsWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("getting /status")
		resource := flowcontrol.SchemeGroupVersion.WithResource("flowschemas")
		fsStatusRead, err := f.DynamicClient.Resource(resource).Get(ctx, fsCreated.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(fsStatusRead.GetObjectKind().GroupVersionKind()).To(gomega.Equal(flowcontrol.SchemeGroupVersion.WithKind("FlowSchema")))
		gomega.Expect(fsStatusRead.GetUID()).To(gomega.Equal(fsCreated.UID))

		ginkgo.By("patching /status")
		patchBytes = []byte(`{"status":{"conditions":[{"type":"PatchStatusFailed","status":"False","reason":"e2e"}]}}`)
		fsStatusPatched, err := client.Patch(ctx, fsCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		condition := apihelpers.GetFlowSchemaConditionByType(fsStatusPatched, flowcontrol.FlowSchemaConditionType("PatchStatusFailed"))
		gomega.Expect(condition).NotTo(gomega.BeNil())

		ginkgo.By("updating /status")
		var fsStatusUpdated *flowcontrol.FlowSchema
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			fs, err := client.Get(ctx, fsCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			fsStatusToUpdate := fs.DeepCopy()
			fsStatusToUpdate.Status.Conditions = append(fsStatusToUpdate.Status.Conditions, flowcontrol.FlowSchemaCondition{
				Type:    "StatusUpdateFailed",
				Status:  flowcontrol.ConditionFalse,
				Reason:  "E2E",
				Message: "Set from an e2e test",
			})
			fsStatusUpdated, err = client.UpdateStatus(ctx, fsStatusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update status of flowschema %q", fsCreated.Name)
		condition = apihelpers.GetFlowSchemaConditionByType(fsStatusUpdated, flowcontrol.FlowSchemaConditionType("StatusUpdateFailed"))
		gomega.Expect(condition).NotTo(gomega.BeNil())

		ginkgo.By("deleting")
		err = client.Delete(ctx, fsCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = client.Get(ctx, fsCreated.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})

	/*
	   Release: v1.29
	   Testname: Priority and Fairness PriorityLevelConfiguration API
	   Description:
	   The flowcontrol.apiserver.k8s.io API group MUST exist in the
	     /apis discovery document.
	   The flowcontrol.apiserver.k8s.io/v1 API group/version MUST exist
	     in the /apis/flowcontrol.apiserver.k8s.io discovery document.
	   The prioritylevelconfiguration and prioritylevelconfiguration/status
	     resources MUST exist in the
	     /apis/flowcontrol.apiserver.k8s.io/v1 discovery document.
	   The prioritylevelconfiguration resource must support create, get,
	     list, watch, update, patch, delete, and deletecollection.
	*/
	framework.ConformanceIt("should support PriorityLevelConfiguration API operations", func(ctx context.Context) {
		plVersion := "v1"
		ginkgo.By("getting /apis")
		{
			discoveryGroups, err := f.ClientSet.Discovery().ServerGroups()
			framework.ExpectNoError(err)
			found := false
			for _, group := range discoveryGroups.Groups {
				if group.Name == flowcontrol.GroupName {
					for _, version := range group.Versions {
						if version.Version == plVersion {
							found = true
							break
						}
					}
				}
			}
			if !found {
				framework.Failf("expected flowcontrol API group/version, got %#v", discoveryGroups.Groups)
			}
		}

		ginkgo.By("getting /apis/flowcontrol.apiserver.k8s.io")
		{
			group := &metav1.APIGroup{}
			err := f.ClientSet.Discovery().RESTClient().Get().AbsPath("/apis/flowcontrol.apiserver.k8s.io").Do(ctx).Into(group)
			framework.ExpectNoError(err)
			found := false
			for _, version := range group.Versions {
				if version.Version == plVersion {
					found = true
					break
				}
			}
			if !found {
				framework.Failf("expected flowcontrol API version, got %#v", group.Versions)
			}
		}

		ginkgo.By("getting /apis/flowcontrol.apiserver.k8s.io/" + plVersion)
		{
			resources, err := f.ClientSet.Discovery().ServerResourcesForGroupVersion(flowcontrol.SchemeGroupVersion.String())
			framework.ExpectNoError(err)
			foundPL, foundPLStatus := false, false
			for _, resource := range resources.APIResources {
				switch resource.Name {
				case "prioritylevelconfigurations":
					foundPL = true
				case "prioritylevelconfigurations/status":
					foundPLStatus = true
				}
			}
			if !foundPL {
				framework.Failf("expected prioritylevelconfigurations, got %#v", resources.APIResources)
			}
			if !foundPLStatus {
				framework.Failf("expected prioritylevelconfigurations/status, got %#v", resources.APIResources)
			}
		}

		client := f.ClientSet.FlowcontrolV1().PriorityLevelConfigurations()
		labelKey, labelValue := "example-e2e-pl-label", utilrand.String(8)
		label := fmt.Sprintf("%s=%s", labelKey, labelValue)

		template := &flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				GenerateName: "e2e-example-pl-",
				Labels: map[string]string{
					labelKey: labelValue,
				},
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: ptr.To(int32(2)),
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject,
					},
				},
			},
		}

		ginkgo.DeferCleanup(func(ctx context.Context) {
			err := client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
			framework.ExpectNoError(err)
		})

		ginkgo.By("creating")
		_, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		_, err = client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)
		plCreated, err := client.Create(ctx, template, metav1.CreateOptions{})
		framework.ExpectNoError(err)

		ginkgo.By("getting")
		plRead, err := client.Get(ctx, plCreated.Name, metav1.GetOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(plRead.UID).To(gomega.Equal(plCreated.UID))
		gomega.Expect(plRead).To(apimachineryutils.HaveValidResourceVersion())

		ginkgo.By("listing")
		list, err := client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(3), "filtered list should have 3 items")

		ginkgo.By("watching")
		framework.Logf("starting watch")
		plWatch, err := client.Watch(ctx, metav1.ListOptions{ResourceVersion: list.ResourceVersion, LabelSelector: label})
		framework.ExpectNoError(err)

		ginkgo.By("patching")
		patchBytes := []byte(`{"metadata":{"annotations":{"patched":"true"}},"spec":{"limited":{"nominalConcurrencyShares":4}}}`)
		plPatched, err := client.Patch(ctx, plCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{})
		framework.ExpectNoError(err)
		gomega.Expect(plPatched.Annotations).To(gomega.HaveKeyWithValue("patched", "true"), "patched object should have the applied annotation")
		gomega.Expect(plPatched.Spec.Limited.NominalConcurrencyShares).To(gomega.Equal(ptr.To(int32(4))), "patched object should have the applied spec")
		gomega.Expect(resourceversion.CompareResourceVersion(plCreated.ResourceVersion, plPatched.ResourceVersion)).To(gomega.BeNumerically("==", -1), "patched object should have a larger resource version")

		ginkgo.By("updating")
		var plUpdated *flowcontrol.PriorityLevelConfiguration
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			pl, err := client.Get(ctx, plCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			plToUpdate := pl.DeepCopy()
			plToUpdate.Annotations["updated"] = "true"
			plToUpdate.Spec.Limited.NominalConcurrencyShares = ptr.To(int32(6))

			plUpdated, err = client.Update(ctx, plToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update prioritylevelconfiguration %q", plCreated.Name)
		gomega.Expect(plUpdated.Annotations).To(gomega.HaveKeyWithValue("updated", "true"), "updated object should have the applied annotation")
		gomega.Expect(plUpdated.Spec.Limited.NominalConcurrencyShares).To(gomega.Equal(ptr.To(int32(6))), "updated object should have the applied spec")

		framework.Logf("waiting for watch events with expected annotations")
		for sawAnnotation := false; !sawAnnotation; {
			select {
			case evt, ok := <-plWatch.ResultChan():
				if !ok {
					framework.Fail("watch channel should not close")
				}
				gomega.Expect(evt.Type).To(gomega.Equal(watch.Modified))
				plWatched, isPL := evt.Object.(*flowcontrol.PriorityLevelConfiguration)
				if !isPL {
					framework.Failf("expected an object of type: %T, but got %T", &flowcontrol.PriorityLevelConfiguration{}, evt.Object)
				}
				if plWatched.Annotations["patched"] == "true" {
					sawAnnotation = true
					plWatch.Stop()
				} else {
					framework.Logf("missing expected annotations, waiting: %#v", plWatched.Annotations)
				}
			case <-time.After(wait.ForeverTestTimeout):
				framework.Fail("timed out waiting for watch event")
			}
		}

		ginkgo.By("getting /status")
		resource := flowcontrol.SchemeGroupVersion.WithResource("prioritylevelconfigurations")
		plStatusRead, err := f.DynamicClient.Resource(resource).Get(ctx, plCreated.Name, metav1.GetOptions{}, "status")
		framework.ExpectNoError(err)
		gomega.Expect(plStatusRead.GetObjectKind().GroupVersionKind()).To(gomega.Equal(flowcontrol.SchemeGroupVersion.WithKind("PriorityLevelConfiguration")))
		gomega.Expect(plStatusRead.GetUID()).To(gomega.Equal(plCreated.UID))

		ginkgo.By("patching /status")
		patchBytes = []byte(`{"status":{"conditions":[{"type":"PatchStatusFailed","status":"False","reason":"e2e"}]}}`)
		plStatusPatched, err := client.Patch(ctx, plCreated.Name, types.MergePatchType, patchBytes, metav1.PatchOptions{}, "status")
		framework.ExpectNoError(err)
		condition := apihelpers.GetPriorityLevelConfigurationConditionByType(plStatusPatched, flowcontrol.PriorityLevelConfigurationConditionType("PatchStatusFailed"))
		gomega.Expect(condition).NotTo(gomega.BeNil())

		ginkgo.By("updating /status")
		var plStatusUpdated *flowcontrol.PriorityLevelConfiguration
		err = retry.RetryOnConflict(retry.DefaultRetry, func() error {
			pl, err := client.Get(ctx, plCreated.Name, metav1.GetOptions{})
			framework.ExpectNoError(err)

			plStatusToUpdate := pl.DeepCopy()
			plStatusToUpdate.Status.Conditions = append(plStatusToUpdate.Status.Conditions, flowcontrol.PriorityLevelConfigurationCondition{
				Type:    "StatusUpdateFailed",
				Status:  flowcontrol.ConditionFalse,
				Reason:  "E2E",
				Message: "Set from an e2e test",
			})
			plStatusUpdated, err = client.UpdateStatus(ctx, plStatusToUpdate, metav1.UpdateOptions{})
			return err
		})
		framework.ExpectNoError(err, "failed to update status of prioritylevelconfiguration %q", plCreated.Name)
		condition = apihelpers.GetPriorityLevelConfigurationConditionByType(plStatusUpdated, flowcontrol.PriorityLevelConfigurationConditionType("StatusUpdateFailed"))
		gomega.Expect(condition).NotTo(gomega.BeNil())

		ginkgo.By("deleting")
		err = client.Delete(ctx, plCreated.Name, metav1.DeleteOptions{})
		framework.ExpectNoError(err)
		_, err = client.Get(ctx, plCreated.Name, metav1.GetOptions{})
		if !apierrors.IsNotFound(err) {
			framework.Failf("expected 404, got %#v", err)
		}

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.HaveLen(2), "filtered list should have 2 items")

		ginkgo.By("deleting a collection")
		err = client.DeleteCollection(ctx, metav1.DeleteOptions{}, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)

		list, err = client.List(ctx, metav1.ListOptions{LabelSelector: label})
		framework.ExpectNoError(err)
		gomega.Expect(list.Items).To(gomega.BeEmpty(), "filtered list should have 0 items")
	})
})

// createPriorityLevel creates a priority level with the provided assured
// concurrency share.
func createPriorityLevel(ctx context.Context, f *framework.Framework, priorityLevelName string, nominalConcurrencyShares int32) *flowcontrol.PriorityLevelConfiguration {
	createdPriorityLevel, err := f.ClientSet.FlowcontrolV1().PriorityLevelConfigurations().Create(
		ctx,
		&flowcontrol.PriorityLevelConfiguration{
			ObjectMeta: metav1.ObjectMeta{
				Name: priorityLevelName,
			},
			Spec: flowcontrol.PriorityLevelConfigurationSpec{
				Type: flowcontrol.PriorityLevelEnablementLimited,
				Limited: &flowcontrol.LimitedPriorityLevelConfiguration{
					NominalConcurrencyShares: ptr.To(nominalConcurrencyShares),
					LimitResponse: flowcontrol.LimitResponse{
						Type: flowcontrol.LimitResponseTypeReject,
					},
				},
			},
		},
		metav1.CreateOptions{})
	framework.ExpectNoError(err)
	ginkgo.DeferCleanup(f.ClientSet.FlowcontrolV1().PriorityLevelConfigurations().Delete, priorityLevelName, metav1.DeleteOptions{})
	return createdPriorityLevel
}

func getPriorityLevelNominalConcurrency(ctx context.Context, c clientset.Interface, priorityLevelName string) (int32, error) {
	req := c.CoreV1().RESTClient().Get().AbsPath("/metrics")
	resp, err := req.DoRaw(ctx)
	if err != nil {
		return 0, fmt.Errorf("error requesting metrics; request=%#+v, request.URL()=%s: %w", req, req.URL(), err)
	}
	sampleDecoder := expfmt.SampleDecoder{
		Dec:  expfmt.NewDecoder(bytes.NewBuffer(resp), expfmt.NewFormat(expfmt.TypeTextPlain)),
		Opts: &expfmt.DecodeOptions{},
	}
	for {
		var v model.Vector
		err := sampleDecoder.Decode(&v)
		if err != nil {
			if err == io.EOF {
				break
			}
			return 0, err
		}
		for _, metric := range v {
			if string(metric.Metric[model.MetricNameLabel]) != nominalConcurrencyLimitMetricName {
				continue
			}
			if string(metric.Metric[priorityLevelLabelName]) != priorityLevelName {
				continue
			}
			return int32(metric.Value), nil
		}
	}
	return 0, errPriorityLevelNotFound
}

// createFlowSchema creates a flow schema referring to a particular priority
// level and matching the username provided.
func createFlowSchema(ctx context.Context, f *framework.Framework, flowSchemaName string, matchingPrecedence int32, priorityLevelName string, matchingUsernames []string) *flowcontrol.FlowSchema {
	var subjects []flowcontrol.Subject
	for _, matchingUsername := range matchingUsernames {
		subjects = append(subjects, flowcontrol.Subject{
			Kind: flowcontrol.SubjectKindUser,
			User: &flowcontrol.UserSubject{
				Name: matchingUsername,
			},
		})
	}

	createdFlowSchema, err := f.ClientSet.FlowcontrolV1().FlowSchemas().Create(
		ctx,
		&flowcontrol.FlowSchema{
			ObjectMeta: metav1.ObjectMeta{
				Name: flowSchemaName,
			},
			Spec: flowcontrol.FlowSchemaSpec{
				MatchingPrecedence: matchingPrecedence,
				PriorityLevelConfiguration: flowcontrol.PriorityLevelConfigurationReference{
					Name: priorityLevelName,
				},
				DistinguisherMethod: &flowcontrol.FlowDistinguisherMethod{
					Type: flowcontrol.FlowDistinguisherMethodByUserType,
				},
				Rules: []flowcontrol.PolicyRulesWithSubjects{
					{
						Subjects: subjects,
						NonResourceRules: []flowcontrol.NonResourcePolicyRule{
							{
								Verbs:           []string{flowcontrol.VerbAll},
								NonResourceURLs: []string{flowcontrol.NonResourceAll},
							},
						},
					},
				},
			},
		},
		metav1.CreateOptions{})
	framework.ExpectNoError(err)
	ginkgo.DeferCleanup(f.ClientSet.FlowcontrolV1().FlowSchemas().Delete, flowSchemaName, metav1.DeleteOptions{})
	return createdFlowSchema
}

// waitForSteadyState repeatedly polls the API server to check if the newly
// created flow schema and priority level have been seen by the APF controller
// by checking: (1) the dangling priority level reference condition in the flow
// schema status, and (2) metrics. The function times out after 30 seconds.
func waitForSteadyState(ctx context.Context, f *framework.Framework, flowSchemaName string, priorityLevelName string) {
	framework.ExpectNoError(wait.PollUntilContextTimeout(ctx, time.Second, 30*time.Second, false, func(ctx context.Context) (bool, error) {
		fs, err := f.ClientSet.FlowcontrolV1().FlowSchemas().Get(ctx, flowSchemaName, metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		condition := apihelpers.GetFlowSchemaConditionByType(fs, flowcontrol.FlowSchemaConditionDangling)
		if condition == nil || condition.Status != flowcontrol.ConditionFalse {
			// The absence of the dangling status object implies that the APF
			// controller isn't done with syncing the flow schema object. And, of
			// course, the condition being anything but false means that steady state
			// hasn't been achieved.
			return false, nil
		}
		_, err = getPriorityLevelNominalConcurrency(ctx, f.ClientSet, priorityLevelName)
		if err != nil {
			if err == errPriorityLevelNotFound {
				return false, nil
			}
			return false, err
		}
		return true, nil
	}))
}

// makeRequests creates a request to the API server and returns the response.
func makeRequest(f *framework.Framework, username string) *http.Response {
	config := f.ClientConfig()
	config.Impersonate.UserName = username
	config.RateLimiter = clientsideflowcontrol.NewFakeAlwaysRateLimiter()
	config.Impersonate.Groups = []string{"system:authenticated"}
	roundTripper, err := rest.TransportFor(config)
	framework.ExpectNoError(err)

	req, err := http.NewRequest(http.MethodGet, f.ClientSet.CoreV1().RESTClient().Get().AbsPath("version").URL().String(), nil)
	framework.ExpectNoError(err)

	response, err := roundTripper.RoundTrip(req)
	framework.ExpectNoError(err)
	return response
}

func getPriorityLevelUID(response *http.Response) string {
	return response.Header.Get(flowcontrol.ResponseHeaderMatchedPriorityLevelConfigurationUID)
}

func getFlowSchemaUID(response *http.Response) string {
	return response.Header.Get(flowcontrol.ResponseHeaderMatchedFlowSchemaUID)
}

// uniformQPSLoadSingle loads the API server with requests at a uniform <qps>
// for <loadDuration> time. The number of successfully completed requests is
// returned.
func uniformQPSLoadSingle(f *framework.Framework, username string, qps float64, loadDuration time.Duration) int32 {
	var completed int32
	var wg sync.WaitGroup
	ticker := time.NewTicker(time.Duration(float64(time.Second) / qps))
	defer ticker.Stop()
	timer := time.NewTimer(loadDuration)
	for {
		select {
		case <-ticker.C:
			wg.Add(1)
			// Each request will have a non-zero latency. In addition, there may be
			// multiple concurrent requests in-flight. As a result, a request may
			// take longer than the time between two different consecutive ticks
			// regardless of whether a requests is accepted or rejected. For example,
			// in cases with clients making requests far above their concurrency
			// share, with little time between consecutive requests, due to limited
			// concurrency, newer requests will be enqueued until older ones
			// complete. Hence the synchronisation with sync.WaitGroup.
			go func() {
				defer wg.Done()
				makeRequest(f, username)
				atomic.AddInt32(&completed, 1)
			}()
		case <-timer.C:
			// Still in-flight requests should not contribute to the completed count.
			totalCompleted := atomic.LoadInt32(&completed)
			wg.Wait() // do not leak goroutines
			return totalCompleted
		}
	}
}

// uniformQPSLoadConcurrent loads the API server with a <concurrency> number of
// clients impersonating to be <username>, each creating requests at a uniform
// rate defined by <qps>. The sum of number of successfully completed requests
// across all concurrent clients is returned.
func uniformQPSLoadConcurrent(f *framework.Framework, username string, concurrency int32, qps float64, loadDuration time.Duration) int32 {
	var completed int32
	var wg sync.WaitGroup
	wg.Add(int(concurrency))
	for i := int32(0); i < concurrency; i++ {
		go func() {
			defer wg.Done()
			atomic.AddInt32(&completed, uniformQPSLoadSingle(f, username, qps, loadDuration))
		}()
	}
	wg.Wait()
	return completed
}
