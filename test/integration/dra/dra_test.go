/*
Copyright 2025 The Kubernetes Authors.

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

package dra

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"

	v1 "k8s.io/api/core/v1"
	resourcealphaapi "k8s.io/api/resource/v1alpha3"
	resourceapi "k8s.io/api/resource/v1beta1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	"k8s.io/component-base/featuregate"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	kubeapiservertesting "k8s.io/kubernetes/cmd/kube-apiserver/app/testing"
	"k8s.io/kubernetes/pkg/features"
	st "k8s.io/kubernetes/pkg/scheduler/testing"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

var (
	// For more test data see pkg/scheduler/framework/plugin/dynamicresources/dynamicresources_test.go.

	podName          = "my-pod"
	namespace        = "default"
	resourceName     = "my-resource"
	className        = "my-resource-class"
	claimName        = podName + "-" + resourceName
	podWithClaimName = st.MakePod().Name(podName).Namespace(namespace).
				Container("my-container").
				PodResourceClaims(v1.PodResourceClaim{Name: resourceName, ResourceClaimName: &claimName}).
				Obj()
	claim = st.MakeResourceClaim().
		Name(claimName).
		Namespace(namespace).
		Request(className).
		Obj()
)

// createTestNamespace creates a namespace with a name that is derived from the
// current test name:
// - Non-alpha-numeric characters replaced by hyphen.
// - Truncated in the middle to make it short enough for GenerateName.
// - Hyphen plus random suffix added by the apiserver.
func createTestNamespace(tCtx ktesting.TContext) string {
	tCtx.Helper()
	name := regexp.MustCompile(`[^[:alnum:]_-]`).ReplaceAllString(tCtx.Name(), "-")
	name = strings.ToLower(name)
	if len(name) > 63 {
		name = name[:30] + "--" + name[len(name)-30:]
	}
	ns := &v1.Namespace{ObjectMeta: metav1.ObjectMeta{GenerateName: name + "-"}}
	ns, err := tCtx.Client().CoreV1().Namespaces().Create(tCtx, ns, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create test namespace")
	tCtx.CleanupCtx(func(tCtx ktesting.TContext) {
		tCtx.ExpectNoError(tCtx.Client().CoreV1().Namespaces().Delete(tCtx, ns.Name, metav1.DeleteOptions{}), "delete test namespace")
	})
	return ns.Name
}

func TestDRA(t *testing.T) {
	// Each sub-test brings up the API server in a certain
	// configuration. These sub-tests must run sequentially because they
	// change the global DefaultFeatureGate. For each configuration,
	// multiple tests can run in parallel as long as they are careful
	// about what they create.
	for name, tc := range map[string]struct {
		apis     map[schema.GroupVersion]bool
		features map[featuregate.Feature]bool
		f        func(tCtx ktesting.TContext)
	}{
		"default": {
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, false) })
				tCtx.Run("APIDisabled", testAPIDisabled)
			},
		},
		"core": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{features.DynamicResourceAllocation: true},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, false) })
				tCtx.Run("Pod", func(tCtx ktesting.TContext) { testPod(tCtx, true) })
			},
		},
		"all": {
			apis: map[schema.GroupVersion]bool{
				resourceapi.SchemeGroupVersion:      true,
				resourcealphaapi.SchemeGroupVersion: true,
			},
			features: map[featuregate.Feature]bool{
				features.DynamicResourceAllocation: true,
				// Additional DRA feature gates go here,
				// in alphabetical order,
				// as needed by tests for them.
				features.DRAAdminAccess: true,
			},
			f: func(tCtx ktesting.TContext) {
				tCtx.Run("AdminAccess", func(tCtx ktesting.TContext) { testAdminAccess(tCtx, true) })
				tCtx.Run("Convert", testConvert)
			},
		},
	} {
		t.Run(name, func(t *testing.T) {
			tCtx := ktesting.Init(t)
			var entries []string
			for key, value := range tc.features {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			for key, value := range tc.apis {
				entries = append(entries, fmt.Sprintf("%s=%t", key, value))
			}
			sort.Strings(entries)
			t.Logf("Config: %s", strings.Join(entries, ","))

			for key, value := range tc.features {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, key, value)
			}

			etcdOptions := framework.SharedEtcd()
			apiServerOptions := kubeapiservertesting.NewDefaultTestServerOptions()
			apiServerFlags := framework.DefaultTestServerFlags()
			// Default kube-apiserver behavior, must be requested explicitly for test server.
			runtimeConfigs := []string{"api/alpha=false", "api/beta=false"}
			for key, value := range tc.apis {
				runtimeConfigs = append(runtimeConfigs, fmt.Sprintf("%s=%t", key, value))
			}
			apiServerFlags = append(apiServerFlags, "--runtime-config="+strings.Join(runtimeConfigs, ","))
			server := kubeapiservertesting.StartTestServerOrDie(t, apiServerOptions, apiServerFlags, etcdOptions)
			tCtx.Cleanup(server.TearDownFn)

			tCtx = ktesting.WithRESTConfig(tCtx, server.ClientConfig)
			tc.f(tCtx)
		})
	}
}

// testPod creates a pod with a resource claim reference and then checks
// whether that field is or isn't getting dropped.
func testPod(tCtx ktesting.TContext, draEnabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx)
	podWithClaimName := podWithClaimName.DeepCopy()
	podWithClaimName.Namespace = namespace
	pod, err := tCtx.Client().CoreV1().Pods(namespace).Create(tCtx, podWithClaimName, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create pod")
	if draEnabled {
		assert.NotEmpty(tCtx, pod.Spec.ResourceClaims, "should store resource claims in pod spec")
	} else {
		assert.Empty(tCtx, pod.Spec.ResourceClaims, "should drop resource claims from pod spec")
	}
}

// testAPIDisabled checks that the resource.k8s.io API is disabled.
func testAPIDisabled(tCtx ktesting.TContext) {
	tCtx.Parallel()
	_, err := tCtx.Client().ResourceV1beta1().ResourceClaims(claim.Namespace).Create(tCtx, claim, metav1.CreateOptions{})
	if !apierrors.IsNotFound(err) {
		tCtx.Fatalf("expected 'resource not found' error, got %v", err)
	}
}

// testConvert creates a claim using a one API version and reads it with another.
func testConvert(tCtx ktesting.TContext) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx)
	claim := claim.DeepCopy()
	claim.Namespace = namespace
	claim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim")
	claimAlpha, err := tCtx.Client().ResourceV1alpha3().ResourceClaims(namespace).Get(tCtx, claim.Name, metav1.GetOptions{})
	tCtx.ExpectNoError(err, "get claim")
	// We could check more fields, but there are unit tests which cover this better.
	assert.Equal(tCtx, claim.Name, claimAlpha.Name, "claim name")
}

// testAdminAccess creates a claim with AdminAccess and then checks
// whether that field is or isn't getting dropped.
func testAdminAccess(tCtx ktesting.TContext, adminAccessEnabled bool) {
	tCtx.Parallel()
	namespace := createTestNamespace(tCtx)
	claim := claim.DeepCopy()
	claim.Namespace = namespace
	claim.Spec.Devices.Requests[0].AdminAccess = ptr.To(true)
	claim, err := tCtx.Client().ResourceV1beta1().ResourceClaims(namespace).Create(tCtx, claim, metav1.CreateOptions{})
	tCtx.ExpectNoError(err, "create claim")
	if adminAccessEnabled {
		if !ptr.Deref(claim.Spec.Devices.Requests[0].AdminAccess, false) {
			tCtx.Fatal("should store AdminAccess in ResourceClaim")
		}
	} else {
		if claim.Spec.Devices.Requests[0].AdminAccess != nil {
			tCtx.Fatal("should drop AdminAccess in ResourceClaim")
		}
	}
}
