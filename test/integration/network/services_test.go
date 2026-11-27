/*
Copyright 2021 The Kubernetes Authors.

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

package network

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/apimachinery/pkg/util/strategicpatch"
	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/apiserver/pkg/registry/generic"
	"k8s.io/apiserver/pkg/storage"
	"k8s.io/apiserver/pkg/storage/storagebackend"
	"k8s.io/apiserver/pkg/storage/storagebackend/factory"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/cmd/kube-apiserver/app/options"
	"k8s.io/kubernetes/pkg/controlplane"

	"k8s.io/ktesting"
	"k8s.io/kubernetes/test/integration/framework"
)

// TestServicesFinalizersRepairLoop tests that Services participate in the object
// deletion when using finalizers, and that the Services Repair controller doesn't,
// mistakenly, repair the ClusterIP assigned to the Service that is being deleted.
// https://issues.k8s.io/87603
func TestServicesFinalizersRepairLoop(t *testing.T) {
	serviceCIDR := "10.0.0.0/16"
	clusterIP := "10.0.0.20"
	interval := 5 * time.Second

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
		},
		ModifyServerConfig: func(cfg *controlplane.Config) {
			cfg.Extra.RepairServicesInterval = interval
		},
	})
	defer tearDownFn()

	// verify client is working
	if err := wait.PollImmediate(5*time.Second, 2*time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Endpoints(metav1.NamespaceDefault).Get(tCtx, "kubernetes", metav1.GetOptions{})
		if err != nil {
			t.Logf("error fetching endpoints: %v", err)
			return false, nil
		}
		return true, nil
	}); err != nil {
		t.Errorf("server without enabled endpoints failed to register: %v", err)
	}

	// Create a NodePort service with one finalizer
	svcNodePort := v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:       "svc",
			Finalizers: []string{"foo.bar/some-finalizer"},
		},
		Spec: v1.ServiceSpec{
			ClusterIP: clusterIP,
			Ports: []v1.ServicePort{{
				Port:       8443,
				NodePort:   30443,
				TargetPort: intstr.FromInt32(8443),
				Protocol:   v1.ProtocolTCP,
			}},
			Type: v1.ServiceTypeNodePort,
		},
	}

	// Create service
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(tCtx, &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error creating service: %v", err)
	}
	t.Logf("Created service: %s", svcNodePort.Name)

	// Check the service has been created correctly
	svc, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
	if err != nil || svc.Spec.ClusterIP != clusterIP {
		t.Errorf("created service is not correct: %v", err)
	}
	t.Logf("Service created successfully: %v", svc)

	// Delete service
	if err := client.CoreV1().Services(metav1.NamespaceDefault).Delete(tCtx, svcNodePort.Name, metav1.DeleteOptions{}); err != nil {
		t.Errorf("unexpected error deleting service: %v", err)
	}
	t.Logf("Deleted service: %s", svcNodePort.Name)

	// wait for the repair loop to recover the deleted resources
	time.Sleep(interval + 1)

	// Check that the service was not deleted and the IP is already allocated
	svc, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
	if err != nil || svc.Spec.ClusterIP != clusterIP {
		t.Errorf("created service is not correct: %v", err)
	}
	t.Logf("Service after Delete: %v", svc)

	// Remove the finalizer
	if _, err = client.CoreV1().Services(metav1.NamespaceDefault).Patch(tCtx, svcNodePort.Name, types.JSONPatchType, []byte(`[{"op":"remove","path":"/metadata/finalizers"}]`), metav1.PatchOptions{}); err != nil {
		t.Errorf("unexpected error removing finalizer: %v", err)
	}
	t.Logf("Removed service finalizer: %s", svcNodePort.Name)

	// Check that the service was deleted
	_, err = client.CoreV1().Services(metav1.NamespaceDefault).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
	if err == nil {
		t.Errorf("service was not delete: %v", err)
	}

	// Try to create service again
	if _, err := client.CoreV1().Services(metav1.NamespaceDefault).Create(tCtx, &svcNodePort, metav1.CreateOptions{}); err != nil {
		t.Errorf("unexpected error creating service: %v", err)
	}
	t.Logf("Created service: %s", svcNodePort.Name)
}

func TestServicesFinalizersPatchStatus(t *testing.T) {
	serviceCIDR := "10.0.0.0/16"
	clusterIP := "10.0.0.21"
	nodePort := 30443
	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
		},
	})
	defer tearDownFn()

	for _, testcase := range []string{"spec", "status"} {
		t.Run(testcase, func(t *testing.T) {
			// Create a NodePort service with one finalizer
			svcNodePort := v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:       "svc" + testcase,
					Finalizers: []string{"foo.bar/some-finalizer"},
				},
				Spec: v1.ServiceSpec{
					ClusterIP: clusterIP,
					Ports: []v1.ServicePort{{
						Port:       8443,
						NodePort:   int32(nodePort),
						TargetPort: intstr.FromInt32(8443),
						Protocol:   v1.ProtocolTCP,
					}},
					Type: v1.ServiceTypeNodePort,
				},
			}

			ns := framework.CreateNamespaceOrDie(client, "test-service-finalizers-"+testcase, t)
			defer framework.DeleteNamespaceOrDie(client, ns, t)

			// Create service
			if _, err := client.CoreV1().Services(ns.Name).Create(tCtx, &svcNodePort, metav1.CreateOptions{}); err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			t.Logf("Created service: %s", svcNodePort.Name)

			// Check the service has been created correctly
			svc, err := client.CoreV1().Services(ns.Name).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
			if err != nil || svc.Spec.ClusterIP != clusterIP {
				t.Fatalf("created service is not correct: %v", err)
			}
			t.Logf("Service created successfully: %+v", svc)

			// Delete service
			if err := client.CoreV1().Services(ns.Name).Delete(tCtx, svcNodePort.Name, metav1.DeleteOptions{}); err != nil {
				t.Fatalf("unexpected error deleting service: %v", err)
			}
			t.Logf("Deleted service: %s", svcNodePort.Name)

			// Check that the service was not deleted and the IP is already allocated
			svc, err = client.CoreV1().Services(ns.Name).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
			if err != nil ||
				svc.Spec.ClusterIP != clusterIP ||
				int(svc.Spec.Ports[0].NodePort) != nodePort ||
				svc.DeletionTimestamp == nil ||
				len(svc.ObjectMeta.Finalizers) != 1 {
				t.Fatalf("Service expected to be deleting and with the same values: %v", err)
			}
			t.Logf("Service after Delete: %+v", svc)

			// Remove the finalizer
			updated := svc.DeepCopy()
			updated.ObjectMeta.Finalizers = []string{}
			patchBytes, err := getPatchBytes(svc, updated)
			if err != nil {
				t.Fatalf("unexpected error getting patch bytes: %v", err)
			}

			if testcase == "spec" {
				if _, err = client.CoreV1().Services(ns.Name).Patch(tCtx, svcNodePort.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}); err != nil {
					t.Fatalf("unexpected error removing finalizer: %v", err)
				}
			} else {
				if _, err = client.CoreV1().Services(ns.Name).Patch(tCtx, svcNodePort.Name, types.StrategicMergePatchType, patchBytes, metav1.PatchOptions{}, "status"); err != nil {
					t.Fatalf("unexpected error removing finalizer: %v", err)
				}
			}
			t.Logf("Removed service finalizer: %s", svcNodePort.Name)

			// Check that the service was deleted
			_, err = client.CoreV1().Services(ns.Name).Get(tCtx, svcNodePort.Name, metav1.GetOptions{})
			if err == nil {
				t.Fatalf("service was not delete: %v", err)
			}

			// Try to create service again without the finalizer to check the ClusterIP and NodePort are deallocated
			svc = svcNodePort.DeepCopy()
			svc.Finalizers = []string{}
			if _, err := client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{}); err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			// Delete service
			if err := client.CoreV1().Services(ns.Name).Delete(tCtx, svc.Name, metav1.DeleteOptions{}); err != nil {
				t.Fatalf("unexpected error deleting service: %v", err)
			}
		})
	}
}

// Regresion test for https://issues.k8s.io/115316
func TestServiceCIDR28bits(t *testing.T) {
	serviceCIDR := "10.0.0.0/28"

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerRunOptions: func(opts *options.ServerRunOptions) {
			opts.ServiceClusterIPRanges = serviceCIDR
		},
	})
	defer tearDownFn()

	// Wait until the default "kubernetes" service is created.
	if err := wait.Poll(250*time.Millisecond, time.Minute, func() (bool, error) {
		_, err := client.CoreV1().Services(metav1.NamespaceDefault).Get(tCtx, "kubernetes", metav1.GetOptions{})
		if err != nil {
			return false, err
		}
		return true, nil
	}); err != nil {
		t.Fatalf("creating kubernetes service timed out")
	}

	ns := framework.CreateNamespaceOrDie(client, "test-regression", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	service := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name: "test-1234",
		},
		Spec: v1.ServiceSpec{
			Type: v1.ServiceTypeClusterIP,
			Ports: []v1.ServicePort{{
				Port: int32(80),
			}},
			Selector: map[string]string{
				"foo": "bar",
			},
		},
	}

	_, err := client.CoreV1().Services(ns.Name).Create(tCtx, service, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("Error creating test service: %v", err)
	}
}

// serviceStorageRESTOptionsGetter wraps the storage of the Services, so the
// tests can inject faults between the apiserver and etcd.
type serviceStorageRESTOptionsGetter struct {
	generic.RESTOptionsGetter
	wrap func(storage.Interface) storage.Interface
}

func (g *serviceStorageRESTOptionsGetter) GetRESTOptions(resource schema.GroupResource, example runtime.Object) (generic.RESTOptions, error) {
	opts, err := g.RESTOptionsGetter.GetRESTOptions(resource, example)
	if err != nil || resource != v1.Resource("services") {
		return opts, err
	}
	decorator := opts.Decorator
	opts.Decorator = func(
		config *storagebackend.ConfigForResource,
		resourcePrefix string,
		cacheKeyFunc generic.CacheKeyFunc,
		reverseKeyFunc storage.ReverseKeyFunc,
		newFunc func() runtime.Object,
		newListFunc func() runtime.Object,
		getAttrsFunc storage.AttrFunc,
		trigger storage.IndexerFuncs,
		indexers *cache.Indexers) (storage.Interface, factory.DestroyFunc, error) {
		s, destroy, err := decorator(config, resourcePrefix, cacheKeyFunc, reverseKeyFunc, newFunc, newListFunc, getAttrsFunc, trigger, indexers)
		if err != nil {
			return nil, nil, err
		}
		return g.wrap(s), destroy, nil
	}
	return opts, nil
}

// lostCreateResponseStorage persists the Service with the given name, but the
// apiserver does not get the response before the deadline of the request.
// This is what happens when etcd is slow to commit the transaction.
type lostCreateResponseStorage struct {
	storage.Interface
	name string
}

func (s *lostCreateResponseStorage) Create(ctx context.Context, key string, obj, out runtime.Object, ttl uint64) error {
	if !strings.HasSuffix(key, "/"+s.name) {
		return s.Interface.Create(ctx, key, obj, out, ttl)
	}
	// etcd commits the transaction, it does not matter how long it takes ...
	if err := s.Interface.Create(context.WithoutCancel(ctx), key, obj, out, ttl); err != nil {
		return err
	}
	// ... but the apiserver stops waiting for the response at the deadline.
	<-ctx.Done()
	return ctx.Err()
}

// concurrentUpdateStorage makes each update of the Service with the given
// name lose the race against another update: the Service is modified after
// the update is computed, but before it is written.
type concurrentUpdateStorage struct {
	storage.Interface
	name string
}

func (s *concurrentUpdateStorage) GuaranteedUpdate(ctx context.Context, key string, destination runtime.Object, ignoreNotFound bool,
	preconditions *storage.Preconditions, tryUpdate storage.UpdateFunc, cachedExistingObject runtime.Object) error {
	if !strings.HasSuffix(key, "/"+s.name) {
		return s.Interface.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, tryUpdate, cachedExistingObject)
	}
	raced := false
	racedUpdate := func(input runtime.Object, res storage.ResponseMeta) (runtime.Object, *uint64, error) {
		output, ttl, err := tryUpdate(input, res)
		if err != nil || raced {
			return output, ttl, err
		}
		raced = true
		err = s.Interface.GuaranteedUpdate(ctx, key, input.DeepCopyObject(), false, nil,
			storage.SimpleUpdate(func(obj runtime.Object) (runtime.Object, error) {
				accessor, err := meta.Accessor(obj)
				if err != nil {
					return nil, err
				}
				accessor.SetAnnotations(map[string]string{"concurrent": "update"})
				return obj, nil
			}), nil)
		return output, ttl, err
	}
	return s.Interface.GuaranteedUpdate(ctx, key, destination, ignoreNotFound, preconditions, racedUpdate, cachedExistingObject)
}

// TestServiceCreateTimeoutKeepsAllocations tests that the ClusterIP and the
// NodePort of a Service are not assigned to another Service if the create
// request timed out but the Service was persisted.
// https://issues.k8s.io/144480
func TestServiceCreateTimeoutKeepsAllocations(t *testing.T) {
	const lostName = "lost-response"

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerConfig: func(cfg *controlplane.Config) {
			cfg.ControlPlane.Generic.RESTOptionsGetter = &serviceStorageRESTOptionsGetter{
				RESTOptionsGetter: cfg.ControlPlane.Generic.RESTOptionsGetter,
				wrap: func(s storage.Interface) storage.Interface {
					return &lostCreateResponseStorage{Interface: s, name: lostName}
				},
			}
		},
	})
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "test-create-timeout", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	lost := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: lostName},
		Spec: v1.ServiceSpec{
			Type:  v1.ServiceTypeLoadBalancer,
			Ports: []v1.ServicePort{{Port: 80}},
		},
	}
	// Request.Timeout() also sets the deadline on the client, and it expires
	// at the same time that the server sends the response.
	err := client.CoreV1().RESTClient().Post().
		Namespace(ns.Name).
		Resource("services").
		Param("timeout", "1s").
		Body(lost).
		Do(tCtx).
		Error()
	if !apierrors.IsTimeout(err) {
		t.Fatalf("expected the request to create the Service to time out, got: %v", err)
	}
	t.Logf("Service create request failed: %v", err)

	lost, err = client.CoreV1().Services(ns.Name).Get(tCtx, lostName, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("expected the Service to be persisted: %v", err)
	}
	if len(lost.Spec.Ports) != 1 || lost.Spec.Ports[0].NodePort == 0 || lost.Spec.ClusterIP == "" {
		t.Fatalf("expected the Service to have a ClusterIP and a NodePort: %+v", lost.Spec)
	}
	nodePort := lost.Spec.Ports[0].NodePort
	clusterIP := lost.Spec.ClusterIP

	conflicts := map[string]*v1.Service{
		fmt.Sprintf("NodePort %d", nodePort): {
			ObjectMeta: metav1.ObjectMeta{Name: "same-nodeport"},
			Spec: v1.ServiceSpec{
				Type:  v1.ServiceTypeLoadBalancer,
				Ports: []v1.ServicePort{{Port: 80, NodePort: nodePort}},
			},
		},
		fmt.Sprintf("ClusterIP %s", clusterIP): {
			ObjectMeta: metav1.ObjectMeta{Name: "same-clusterip"},
			Spec: v1.ServiceSpec{
				Type:      v1.ServiceTypeClusterIP,
				ClusterIP: clusterIP,
				Ports:     []v1.ServicePort{{Port: 80}},
			},
		},
	}
	// The apiserver continues to process the request after it sends the
	// timeout to the client, so wait to see the final state of the allocations.
	err = wait.PollUntilContextTimeout(tCtx, 100*time.Millisecond, 3*time.Second, true, func(context.Context) (bool, error) {
		for allocation, svc := range conflicts {
			_, err := client.CoreV1().Services(ns.Name).Create(tCtx, svc, metav1.CreateOptions{})
			if err == nil {
				t.Errorf("%s is in use by Service %s, but it was also assigned to Service %s", allocation, lostName, svc.Name)
				delete(conflicts, allocation)
			} else if !apierrors.IsInvalid(err) {
				return false, fmt.Errorf("creating Service %s: %w", svc.Name, err)
			}
		}
		return len(conflicts) == 0, nil
	})
	if err != nil && !wait.Interrupted(err) {
		t.Fatalf("unexpected error: %v", err)
	}
}

// TestServiceUpdateConflictKeepsAllocations tests that the NodePort of a
// Service is not assigned to another Service if the update that releases it
// fails with a conflict.
func TestServiceUpdateConflictKeepsAllocations(t *testing.T) {
	const name = "concurrent-update"

	tCtx := ktesting.Init(t)
	client, _, tearDownFn := framework.StartTestServer(tCtx, t, framework.TestServerSetup{
		ModifyServerConfig: func(cfg *controlplane.Config) {
			cfg.ControlPlane.Generic.RESTOptionsGetter = &serviceStorageRESTOptionsGetter{
				RESTOptionsGetter: cfg.ControlPlane.Generic.RESTOptionsGetter,
				wrap: func(s storage.Interface) storage.Interface {
					return &concurrentUpdateStorage{Interface: s, name: name}
				},
			}
		},
	})
	defer tearDownFn()

	ns := framework.CreateNamespaceOrDie(client, "test-update-conflict", t)
	defer framework.DeleteNamespaceOrDie(client, ns, t)

	svc, err := client.CoreV1().Services(ns.Name).Create(tCtx, &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: name},
		Spec: v1.ServiceSpec{
			Type:  v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{{Port: 80}},
		},
	}, metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error creating the Service: %v", err)
	}
	nodePort := svc.Spec.Ports[0].NodePort

	// A ClusterIP Service does not have a NodePort.
	svc.Spec.Type = v1.ServiceTypeClusterIP
	_, err = client.CoreV1().Services(ns.Name).Update(tCtx, svc, metav1.UpdateOptions{})
	if !apierrors.IsConflict(err) {
		t.Fatalf("expected the request to update the Service to fail with a conflict, got: %v", err)
	}
	t.Logf("Service update request failed: %v", err)

	svc, err = client.CoreV1().Services(ns.Name).Get(tCtx, name, metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error getting the Service: %v", err)
	}
	if svc.Spec.Type != v1.ServiceTypeNodePort || svc.Spec.Ports[0].NodePort != nodePort {
		t.Fatalf("expected the Service to keep the NodePort %d: %+v", nodePort, svc.Spec)
	}

	conflict := &v1.Service{
		ObjectMeta: metav1.ObjectMeta{Name: "same-nodeport"},
		Spec: v1.ServiceSpec{
			Type:  v1.ServiceTypeNodePort,
			Ports: []v1.ServicePort{{Port: 80, NodePort: nodePort}},
		},
	}
	_, err = client.CoreV1().Services(ns.Name).Create(tCtx, conflict, metav1.CreateOptions{})
	if err == nil {
		t.Errorf("NodePort %d is in use by Service %s, but it was also assigned to Service %s", nodePort, name, conflict.Name)
	} else if !apierrors.IsInvalid(err) {
		t.Errorf("unexpected error creating Service %s: %v", conflict.Name, err)
	}
}

func getPatchBytes(oldSvc, newSvc *v1.Service) ([]byte, error) {
	oldData, err := json.Marshal(oldSvc)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal oldData for svc %s/%s: %v", oldSvc.Namespace, oldSvc.Name, err)
	}

	newData, err := json.Marshal(newSvc)
	if err != nil {
		return nil, fmt.Errorf("failed to Marshal newData for svc %s/%s: %v", newSvc.Namespace, newSvc.Name, err)
	}

	patchBytes, err := strategicpatch.CreateTwoWayMergePatch(oldData, newData, v1.Service{})
	if err != nil {
		return nil, fmt.Errorf("failed to CreateTwoWayMergePatch for svc %s/%s: %v", oldSvc.Namespace, oldSvc.Name, err)
	}
	return patchBytes, nil

}
