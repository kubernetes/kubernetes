/*
Copyright The Kubernetes Authors.

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

package apiserver

import (
	"errors"
	"fmt"
	"strings"
	"testing"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/resourceversion"
	"k8s.io/client-go/dynamic"
	"k8s.io/client-go/metadata"
	"k8s.io/kubernetes/test/integration/framework"
	"k8s.io/utils/ptr"
)

func TestDeleteResult(t *testing.T) {
	ctx, clientSet, kubeConfig, tearDownFn := setup(t)
	defer tearDownFn()

	dynamicClient, err := dynamic.NewForConfig(kubeConfig)
	if err != nil {
		t.Fatal(err)
	}

	metadataClient, err := metadata.NewForConfig(kubeConfig)
	if err != nil {
		t.Fatal(err)
	}

	ns := framework.CreateNamespaceOrDie(clientSet, "delete-status", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

	sa := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: ns.Name}}
	if _, err := clientSet.CoreV1().ServiceAccounts(ns.Name).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create default service account: %v", err)
	}

	podGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"}
	cmGVR := schema.GroupVersionResource{Group: "", Version: "v1", Resource: "configmaps"}

	// 1. Test ReturnDeletedObject = true with Pods
	t.Run("ReturnDeletedObject=true", func(t *testing.T) {
		// a. Typed Client
		{
			pod1 := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod1", Namespace: ns.Name},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c", Image: "nginx"}}},
			}
			if _, err := clientSet.CoreV1().Pods(ns.Name).Create(ctx, pod1, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create pod: %v", err)
			}
			res, err := clientSet.CoreV1().Pods(ns.Name).DeleteWithResult(ctx, "pod1", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 && statusCode != 202 {
				t.Errorf("expected status code 200 or 202, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			podObj, ok := obj.(*v1.Pod)
			if !ok {
				t.Fatalf("expected *v1.Pod, got %T", obj)
			}
			if podObj.ResourceVersion == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}

		// b. Dynamic Client
		{
			pod2 := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod2", Namespace: ns.Name},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c", Image: "nginx"}}},
			}
			if _, err := clientSet.CoreV1().Pods(ns.Name).Create(ctx, pod2, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create pod: %v", err)
			}
			res, err := dynamicClient.Resource(podGVR).Namespace(ns.Name).DeleteWithResult(ctx, "pod2", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 && statusCode != 202 {
				t.Errorf("expected status code 200 or 202, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			unstObj, ok := obj.(*unstructured.Unstructured)
			if !ok {
				t.Fatalf("expected *unstructured.Unstructured, got %T", obj)
			}
			if unstObj.GetResourceVersion() == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}

		// c. Metadata Client
		{
			pod3 := &v1.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "pod3", Namespace: ns.Name},
				Spec:       v1.PodSpec{Containers: []v1.Container{{Name: "c", Image: "nginx"}}},
			}
			if _, err := clientSet.CoreV1().Pods(ns.Name).Create(ctx, pod3, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create pod: %v", err)
			}
			res, err := metadataClient.Resource(podGVR).Namespace(ns.Name).DeleteWithResult(ctx, "pod3", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 && statusCode != 202 {
				t.Errorf("expected status code 200 or 202, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			metaObj, ok := obj.(*metav1.PartialObjectMetadata)
			if !ok {
				t.Fatalf("expected *metav1.PartialObjectMetadata, got %T", obj)
			}
			if metaObj.ResourceVersion == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}
	})

	// 2. Test ReturnDeletedObject = false with ConfigMaps
	t.Run("ReturnDeletedObject=false", func(t *testing.T) {
		// a. Typed Client
		{
			cm1 := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm1", Namespace: ns.Name},
				Data:       map[string]string{"foo": "bar"},
			}
			if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm1, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create cm: %v", err)
			}
			res, err := clientSet.CoreV1().ConfigMaps(ns.Name).DeleteWithResult(ctx, "cm1", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 {
				t.Errorf("expected status code 200, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			statusObj, ok := obj.(*metav1.Status)
			if !ok {
				t.Fatalf("expected *metav1.Status, got %T", obj)
			}
			if statusObj.ResourceVersion == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}

		// b. Dynamic Client
		{
			cm2 := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm2", Namespace: ns.Name},
				Data:       map[string]string{"foo": "bar"},
			}
			if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm2, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create cm: %v", err)
			}
			res, err := dynamicClient.Resource(cmGVR).Namespace(ns.Name).DeleteWithResult(ctx, "cm2", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 {
				t.Errorf("expected status code 200, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			statusObj2, ok := obj.(*metav1.Status)
			if !ok {
				t.Fatalf("expected *metav1.Status, got %T", obj)
			}
			if statusObj2.ResourceVersion == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}

		// c. Metadata Client
		{
			cm3 := &v1.ConfigMap{
				ObjectMeta: metav1.ObjectMeta{Name: "cm3", Namespace: ns.Name},
				Data:       map[string]string{"foo": "bar"},
			}
			if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm3, metav1.CreateOptions{}); err != nil {
				t.Fatalf("failed to create cm: %v", err)
			}
			res, err := metadataClient.Resource(cmGVR).Namespace(ns.Name).DeleteWithResult(ctx, "cm3", metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 200 {
				t.Errorf("expected status code 200, got %d", statusCode)
			}
			obj, err := res.Get()
			if err != nil {
				t.Fatalf("unexpected error on Get(): %v", err)
			}
			statusObj3, ok := obj.(*metav1.Status)
			if !ok {
				t.Fatalf("expected *metav1.Status, got %T", obj)
			}
			if statusObj3.ResourceVersion == "" {
				t.Fatalf("expected ResourceVersion to be populated, got empty string")
			}
		}
	})

	// 3. Test Error Case (Get HTTP code from error and result)
	t.Run("ErrorCase", func(t *testing.T) {
		res, err := clientSet.CoreV1().Pods(ns.Name).DeleteWithResult(ctx, "non-existent-pod", metav1.DeleteOptions{})
		if err == nil {
			t.Fatalf("expected error, got nil")
		}
		if res != nil {
			var statusCode int
			res.StatusCode(&statusCode)
			if statusCode != 404 {
				t.Errorf("expected status code 404 from result, got %d", statusCode)
			}
		}
		var statusErr *apierrors.StatusError
		if !errors.As(err, &statusErr) {
			t.Fatalf("expected *apierrors.StatusError, got %T (%v)", err, err)
		}
		if statusErr.ErrStatus.Code != 404 {
			t.Errorf("expected HTTP status 404, got %d", statusErr.ErrStatus.Code)
		}
		if statusErr.Status().Code != 404 {
			t.Errorf("expected APIStatus code 404, got %d", statusErr.Status().Code)
		}
	})
}

func TestDeletionResponseMatrix(t *testing.T) {
	ctx, clientSet, _, tearDownFn := setup(t)
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(clientSet, "matrix-test", t)
	defer framework.DeleteNamespaceOrDie(clientSet, ns, t)

	// Explicitly create the default service account in the namespace before creating Pods
	sa := &v1.ServiceAccount{ObjectMeta: metav1.ObjectMeta{Name: "default", Namespace: ns.Name}}
	if _, err := clientSet.CoreV1().ServiceAccounts(ns.Name).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
		t.Fatalf("failed to create default service account: %v", err)
	}

	testCases := []struct {
		name         string
		resource     string // "pods", "serviceaccounts", "configmaps"
		initialDT    bool
		initialFin   bool
		deleteType   string // "Normal", "Orphan", "Foreground", "Shorten Future", "Shorten Past"
		expectedType string
		expectedCode int
		expectRVInc  bool // true if we expect RV to increment, false if we expect it to stay equal
	}{
		// 1. Pods (Graceful Deletion)
		{"Pod_No_dT_No_Fin_Normal_Normal", "pods", false, false, "Normal", "*v1.Pod", 200, true},
		{"Pod_No_dT_No_Fin_Orphan_Normal", "pods", false, false, "Orphan", "*v1.Pod", 200, true},
		{"Pod_No_dT_No_Fin_Foreground_Normal", "pods", false, false, "Foreground", "*v1.Pod", 200, true},
		{"Pod_No_dT_No_Fin_Shorten_Future_Normal", "pods", false, false, "Shorten Future", "*v1.Pod", 200, true},
		{"Pod_No_dT_No_Fin_Shorten_Past_Normal", "pods", false, false, "Shorten Past", "*v1.Pod", 200, true},
		{"Pod_dT_set_No_Fin_Normal_Normal", "pods", true, false, "Normal", "*v1.Pod", 200, false},
		{"Pod_dT_set_No_Fin_Shorten_Future_Normal", "pods", true, false, "Shorten Future", "*v1.Pod", 200, true},
		{"Pod_dT_set_No_Fin_Shorten_Past_Normal", "pods", true, false, "Shorten Past", "*v1.Pod", 200, true},
		{"Pod_dT_set_Active_Fin_Normal_Normal", "pods", true, true, "Normal", "*v1.Pod", 200, false},
		{"Pod_dT_set_Active_Fin_Orphan_Normal", "pods", true, true, "Orphan", "*v1.Pod", 200, false},
		{"Pod_dT_set_Active_Fin_Foreground_Normal", "pods", true, true, "Foreground", "*v1.Pod", 200, false},
		{"Pod_dT_set_Active_Fin_Shorten_Future_Normal", "pods", true, true, "Shorten Future", "*v1.Pod", 200, true},
		{"Pod_dT_set_Active_Fin_Shorten_Past_Normal", "pods", true, true, "Shorten Past", "*v1.Pod", 200, true},

		// 2. ServiceAccount (ReturnDeletedObject=true)
		{"SA_No_dT_No_Fin_Normal_Normal", "serviceaccounts", false, false, "Normal", "*v1.ServiceAccount", 200, true},
		{"SA_No_dT_No_Fin_Orphan_Normal", "serviceaccounts", false, false, "Orphan", "*v1.ServiceAccount", 200, true},
		{"SA_No_dT_No_Fin_Foreground_Normal", "serviceaccounts", false, false, "Foreground", "*v1.ServiceAccount", 200, true},
		{"SA_dT_set_Active_Fin_Normal_Normal", "serviceaccounts", true, true, "Normal", "*v1.ServiceAccount", 200, false},
		{"SA_dT_set_Active_Fin_Orphan_Normal", "serviceaccounts", true, true, "Orphan", "*v1.ServiceAccount", 200, true},
		{"SA_dT_set_Active_Fin_Foreground_Normal", "serviceaccounts", true, true, "Foreground", "*v1.ServiceAccount", 200, true},

		// 3. ConfigMap (ReturnDeletedObject=false)
		{"CM_No_dT_No_Fin_Normal_Normal", "configmaps", false, false, "Normal", "*metav1.Status", 200, true},
		{"CM_No_dT_No_Fin_Orphan_Normal", "configmaps", false, false, "Orphan", "*v1.ConfigMap", 200, true},
		{"CM_No_dT_No_Fin_Foreground_Normal", "configmaps", false, false, "Foreground", "*v1.ConfigMap", 200, true},
		{"CM_dT_set_Active_Fin_Normal_Normal", "configmaps", true, true, "Normal", "*v1.ConfigMap", 200, false},
		{"CM_dT_set_Active_Fin_Orphan_Normal", "configmaps", true, true, "Orphan", "*v1.ConfigMap", 200, true},
		{"CM_dT_set_Active_Fin_Foreground_Normal", "configmaps", true, true, "Foreground", "*v1.ConfigMap", 200, true},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			objName := strings.ToLower(strings.ReplaceAll("obj-"+tc.name, "_", "-"))
			var finalizers []string
			if tc.initialFin {
				finalizers = []string{"kubernetes.io/finalizer"}
			}

			// 1. Create object
			switch tc.resource {
			case "pods":
				pod := &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{Name: objName, Namespace: ns.Name, Finalizers: finalizers},
					Spec:       v1.PodSpec{NodeName: "node1", Containers: []v1.Container{{Name: "c", Image: "nginx"}}},
				}
				if _, err := clientSet.CoreV1().Pods(ns.Name).Create(ctx, pod, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create pod: %v", err)
				}
			case "serviceaccounts":
				sa := &v1.ServiceAccount{
					ObjectMeta: metav1.ObjectMeta{Name: objName, Namespace: ns.Name, Finalizers: finalizers},
				}
				if _, err := clientSet.CoreV1().ServiceAccounts(ns.Name).Create(ctx, sa, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create sa: %v", err)
				}
			case "configmaps":
				cm := &v1.ConfigMap{
					ObjectMeta: metav1.ObjectMeta{Name: objName, Namespace: ns.Name, Finalizers: finalizers},
					Data:       map[string]string{"foo": "bar"},
				}
				if _, err := clientSet.CoreV1().ConfigMaps(ns.Name).Create(ctx, cm, metav1.CreateOptions{}); err != nil {
					t.Fatalf("failed to create cm: %v", err)
				}
			}

			// 2. Establish Initial State if dT set
			if tc.initialDT && !tc.initialFin {
				// Delete with grace period 30s so dT is set but Pod remains
				err := clientSet.CoreV1().Pods(ns.Name).Delete(ctx, objName, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](30)})
				if err != nil {
					t.Fatalf("failed to establish initial state dT set, No Fin: %v", err)
				}
			} else if tc.initialDT && tc.initialFin {
				// Delete with default options. Since it has a finalizer, dT is set but object remains
				var err error
				switch tc.resource {
				case "pods":
					err = clientSet.CoreV1().Pods(ns.Name).Delete(ctx, objName, metav1.DeleteOptions{GracePeriodSeconds: ptr.To[int64](30)})
				case "serviceaccounts":
					err = clientSet.CoreV1().ServiceAccounts(ns.Name).Delete(ctx, objName, metav1.DeleteOptions{})
				case "configmaps":
					err = clientSet.CoreV1().ConfigMaps(ns.Name).Delete(ctx, objName, metav1.DeleteOptions{})
				}
				if err != nil {
					t.Fatalf("failed to establish initial state dT set, Active Fin: %v", err)
				}
			}

			var preDeleteRV string
			switch tc.resource {
			case "pods":
				p, err := clientSet.CoreV1().Pods(ns.Name).Get(ctx, objName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get pod before delete: %v", err)
				}
				preDeleteRV = p.ResourceVersion
			case "serviceaccounts":
				sa, err := clientSet.CoreV1().ServiceAccounts(ns.Name).Get(ctx, objName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get sa before delete: %v", err)
				}
				preDeleteRV = sa.ResourceVersion
			case "configmaps":
				cm, err := clientSet.CoreV1().ConfigMaps(ns.Name).Get(ctx, objName, metav1.GetOptions{})
				if err != nil {
					t.Fatalf("failed to get cm before delete: %v", err)
				}
				preDeleteRV = cm.ResourceVersion
			}

			// 3. Prepare DeleteOptions for main request
			opts := metav1.DeleteOptions{}
			switch tc.deleteType {
			case "Orphan":
				opts.PropagationPolicy = ptr.To(metav1.DeletePropagationOrphan)
			case "Foreground":
				opts.PropagationPolicy = ptr.To(metav1.DeletePropagationForeground)
			case "Shorten Future":
				opts.GracePeriodSeconds = ptr.To[int64](15)
			case "Shorten Past":
				opts.GracePeriodSeconds = ptr.To[int64](0)
			}

			// 4. Execute main delete request
			var statusCode int
			res := clientSet.CoreV1().RESTClient().Delete().
				Resource(tc.resource).
				Namespace(ns.Name).
				Name(objName).
				Body(&opts).
				Do(ctx).
				StatusCode(&statusCode)

			obj, err := res.Get()
			var retType string
			var rv string
			if err != nil {
				var statusErr *apierrors.StatusError
				if errors.As(err, &statusErr) {
					retType = fmt.Sprintf("*metav1.Status (%s)", statusErr.ErrStatus.Reason)
					rv = statusErr.ErrStatus.ResourceVersion
				} else {
					retType = fmt.Sprintf("error (%v)", err)
				}
			} else {
				switch o := obj.(type) {
				case *v1.Pod:
					retType = "*v1.Pod"
					rv = o.ResourceVersion
				case *v1.ServiceAccount:
					retType = "*v1.ServiceAccount"
					rv = o.ResourceVersion
				case *v1.ConfigMap:
					retType = "*v1.ConfigMap"
					rv = o.ResourceVersion
				case *metav1.Status:
					retType = "*metav1.Status"
					rv = o.ResourceVersion
				default:
					retType = fmt.Sprintf("%T", obj)
				}
			}

			if retType != tc.expectedType {
				t.Errorf("expected type %s, got %s", tc.expectedType, retType)
			}
			if statusCode != tc.expectedCode {
				t.Errorf("expected status code %d, got %d", tc.expectedCode, statusCode)
			}
			if rv == "" {
				t.Errorf("expected non-empty ResourceVersion")
			} else {
				cmp, err := resourceversion.CompareResourceVersion(rv, preDeleteRV)
				if err != nil {
					t.Errorf("failed to compare resource versions: %v", err)
				} else if tc.expectRVInc {
					if cmp <= 0 {
						t.Errorf("expected returned ResourceVersion %q to be greater than pre-deletion ResourceVersion %q", rv, preDeleteRV)
					}
				} else {
					if cmp != 0 {
						t.Errorf("expected returned ResourceVersion %q to be equal to pre-deletion ResourceVersion %q", rv, preDeleteRV)
					}
				}
			}
		})

	}
}
