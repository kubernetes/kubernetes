/*
Copyright 2015 The Kubernetes Authors.

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

package storage

import (
	"context"
	"fmt"
	"net"
	"reflect"
	stdruntime "runtime"
	"strings"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/util/intstr"
	machineryutilnet "k8s.io/apimachinery/pkg/util/net"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/registry/generic"
	genericregistrytest "k8s.io/apiserver/pkg/registry/generic/testing"
	"k8s.io/apiserver/pkg/registry/rest"
	etcd3testing "k8s.io/apiserver/pkg/storage/etcd3/testing"
	utilfeature "k8s.io/apiserver/pkg/util/feature"
	featuregatetesting "k8s.io/component-base/featuregate/testing"
	epstest "k8s.io/kubernetes/pkg/api/endpoints/testing"
	svctest "k8s.io/kubernetes/pkg/api/service/testing"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/features"
	endpointstore "k8s.io/kubernetes/pkg/registry/core/endpoint/storage"
	podstore "k8s.io/kubernetes/pkg/registry/core/pod/storage"
	"k8s.io/kubernetes/pkg/registry/core/service/ipallocator"
	"k8s.io/kubernetes/pkg/registry/core/service/portallocator"
	"k8s.io/kubernetes/pkg/registry/registrytest"
	netutils "k8s.io/utils/net"
)

// Most tests will use this to create a registry to run tests against.
func newStorage(t *testing.T, ipFamilies []api.IPFamily) (*wrapperRESTForTests, *StatusREST, *etcd3testing.EtcdTestServer) {
	return newStorageWithPods(t, ipFamilies, nil, nil)
}

func newStorageWithPods(t *testing.T, ipFamilies []api.IPFamily, pods []api.Pod, endpoints []*api.Endpoints) (*wrapperRESTForTests, *StatusREST, *etcd3testing.EtcdTestServer) {
	etcdStorage, server := registrytest.NewEtcdStorage(t, "")
	restOptions := generic.RESTOptions{
		StorageConfig:           etcdStorage.ForResource(schema.GroupResource{Resource: "services"}),
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 1,
		ResourcePrefix:          "services",
	}

	ipAllocs := map[api.IPFamily]ipallocator.Interface{}
	for _, fam := range ipFamilies {
		switch fam {
		case api.IPv4Protocol:
			_, cidr, _ := netutils.ParseCIDRSloppy("10.0.0.0/16")
			ipAllocs[fam] = makeIPAllocator(cidr)
		case api.IPv6Protocol:
			_, cidr, _ := netutils.ParseCIDRSloppy("2000::/108")
			ipAllocs[fam] = makeIPAllocator(cidr)
		default:
			t.Fatalf("Unknown IPFamily: %v", fam)
		}
	}

	portAlloc := makePortAllocator(*(machineryutilnet.ParsePortRangeOrDie("30000-32767")))

	// Not all tests will specify pods and endpoints.
	podStorage, err := podstore.NewStorage(generic.RESTOptions{
		StorageConfig:           etcdStorage,
		Decorator:               generic.UndecoratedStorage,
		DeleteCollectionWorkers: 3,
		ResourcePrefix:          "pods",
	}, nil, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	if pods != nil && len(pods) > 0 {
		ctx := genericapirequest.NewDefaultContext()
		for ix := range pods {
			key, _ := podStorage.Pod.KeyFunc(ctx, pods[ix].Name)
			if err := podStorage.Pod.Storage.Create(ctx, key, &pods[ix], nil, 0, false); err != nil {
				t.Fatalf("Couldn't create pod: %v", err)
			}
		}
	}

	endpointsStorage, err := endpointstore.NewREST(generic.RESTOptions{
		StorageConfig:  etcdStorage,
		Decorator:      generic.UndecoratedStorage,
		ResourcePrefix: "endpoints",
	})
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	if endpoints != nil && len(endpoints) > 0 {
		ctx := genericapirequest.NewDefaultContext()
		for ix := range endpoints {
			key, _ := endpointsStorage.KeyFunc(ctx, endpoints[ix].Name)
			if err := endpointsStorage.Store.Storage.Create(ctx, key, endpoints[ix], nil, 0, false); err != nil {
				t.Fatalf("Couldn't create endpoint: %v", err)
			}
		}
	}

	serviceStorage, statusStorage, _, err := NewREST(restOptions, ipFamilies[0], ipAllocs, portAlloc, endpointsStorage, podStorage.Pod, nil)
	if err != nil {
		t.Fatalf("unexpected error from REST storage: %v", err)
	}
	return &wrapperRESTForTests{serviceStorage}, statusStorage, server
}

func makeIPAllocator(cidr *net.IPNet) ipallocator.Interface {
	al, err := ipallocator.NewInMemory(cidr)
	if err != nil {
		panic(fmt.Sprintf("error creating IP allocator: %v", err))
	}
	return al
}

func makePortAllocator(ports machineryutilnet.PortRange) portallocator.Interface {
	al, err := portallocator.NewInMemory(ports)
	if err != nil {
		panic(fmt.Sprintf("error creating port allocator: %v", err))
	}
	return al
}

// wrapperRESTForTests is a *trivial* wrapper for the real REST, which allows us to do
// things that are specifically to enhance test safety.
type wrapperRESTForTests struct {
	*REST
}

func (f *wrapperRESTForTests) Create(ctx context.Context, obj runtime.Object, createValidation rest.ValidateObjectFunc, options *metav1.CreateOptions) (runtime.Object, error) {
	// Making a DeepCopy here ensures that any in-place mutations of the input
	// are not going to propagate to verification code, which used to happen
	// resulting in tests that passed when they shouldn't have.
	obj = obj.DeepCopyObject()
	return f.REST.Create(ctx, obj, createValidation, options)
}

//
// Generic registry tests
//

// This is used in generic registry tests.
func validService() *api.Service {
	return svctest.MakeService("foo",
		svctest.SetClusterIPs(api.ClusterIPNone),
		svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
		svctest.SetIPFamilies(api.IPv4Protocol))
}

func TestGenericCreate(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	svc := validService()
	svc.ObjectMeta = metav1.ObjectMeta{} // because genericregistrytest
	test.TestCreate(
		// valid
		svc,
		// invalid
		&api.Service{
			Spec: api.ServiceSpec{},
		},
	)
}

func TestGenericUpdate(t *testing.T) {
	clusterInternalTrafficPolicy := api.ServiceInternalTrafficPolicyCluster

	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestUpdate(
		// valid
		validService(),
		// updateFunc
		func(obj runtime.Object) runtime.Object {
			object := obj.(*api.Service)
			object.Spec = api.ServiceSpec{
				Selector:        map[string]string{"bar": "baz2"},
				ClusterIP:       api.ClusterIPNone,
				ClusterIPs:      []string{api.ClusterIPNone},
				SessionAffinity: api.ServiceAffinityNone,
				Type:            api.ServiceTypeClusterIP,
				Ports: []api.ServicePort{{
					Port:       6502,
					Protocol:   api.ProtocolTCP,
					TargetPort: intstr.FromInt32(6502),
				}},
				InternalTrafficPolicy: &clusterInternalTrafficPolicy,
			}
			return object
		},
	)
}

func TestGenericDelete(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate().ReturnDeletedObject()
	test.TestDelete(validService())
}

func TestGenericGet(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestGet(validService())
}

func TestGenericList(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store).AllowCreateOnUpdate()
	test.TestList(validService())
}

func TestGenericWatch(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	test := genericregistrytest.New(t, storage.Store)
	test.TestWatch(
		validService(),
		// matching labels
		[]labels.Set{},
		// not matching labels
		[]labels.Set{
			{"foo": "bar"},
		},
		// matching fields
		[]fields.Set{
			{"metadata.name": "foo"},
		},
		// not matching fields
		[]fields.Set{
			{"metadata.name": "bar"},
		},
	)
}

func TestGenericShortNames(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"svc"}
	registrytest.AssertShortNames(t, storage, expected)
}

func TestGenericCategories(t *testing.T) {
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	expected := []string{"all"}
	registrytest.AssertCategories(t, storage, expected)
}

//
// Tests of internal functions
//

func TestNormalizeClusterIPs(t *testing.T) {
	makeServiceWithClusterIp := func(clusterIP string, clusterIPs []string) *api.Service {
		return &api.Service{
			Spec: api.ServiceSpec{
				ClusterIP:  clusterIP,
				ClusterIPs: clusterIPs,
			},
		}
	}

	testCases := []struct {
		name               string
		oldService         *api.Service
		newService         *api.Service
		expectedClusterIP  string
		expectedClusterIPs []string
	}{{
		name:               "new - only clusterip used",
		oldService:         nil,
		newService:         makeServiceWithClusterIp("10.0.0.10", nil),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "new - only clusterips used",
		oldService:         nil,
		newService:         makeServiceWithClusterIp("", []string{"10.0.0.10"}),
		expectedClusterIP:  "", // this is a validation issue, and validation will catch it
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "new - both used",
		oldService:         nil,
		newService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "update - no change",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "update - malformed change",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("10.0.0.11", []string{"10.0.0.11"}),
		expectedClusterIP:  "10.0.0.11",
		expectedClusterIPs: []string{"10.0.0.11"},
	}, {
		name:               "update - malformed change on secondary ip",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10", "2000::1"}),
		newService:         makeServiceWithClusterIp("10.0.0.11", []string{"10.0.0.11", "3000::1"}),
		expectedClusterIP:  "10.0.0.11",
		expectedClusterIPs: []string{"10.0.0.11", "3000::1"},
	}, {
		name:               "update - upgrade",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10", "2000::1"}),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10", "2000::1"},
	}, {
		name:               "update - downgrade",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10", "2000::1"}),
		newService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "update - user cleared cluster IP",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("", []string{"10.0.0.10"}),
		expectedClusterIP:  "",
		expectedClusterIPs: nil,
	}, {
		name:               "update - user cleared clusterIPs", // *MUST* REMAIN FOR OLD CLIENTS
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("10.0.0.10", nil),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "update - user cleared both",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("", nil),
		expectedClusterIP:  "",
		expectedClusterIPs: nil,
	}, {
		name:               "update - user cleared ClusterIP but changed clusterIPs",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("", []string{"10.0.0.11"}),
		expectedClusterIP:  "", /* validation catches this */
		expectedClusterIPs: []string{"10.0.0.11"},
	}, {
		name:               "update - user cleared ClusterIPs but changed ClusterIP",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10", "2000::1"}),
		newService:         makeServiceWithClusterIp("10.0.0.11", nil),
		expectedClusterIP:  "10.0.0.11",
		expectedClusterIPs: nil,
	}, {
		name:               "update - user changed from None to ClusterIP",
		oldService:         makeServiceWithClusterIp("None", []string{"None"}),
		newService:         makeServiceWithClusterIp("10.0.0.10", []string{"None"}),
		expectedClusterIP:  "10.0.0.10",
		expectedClusterIPs: []string{"10.0.0.10"},
	}, {
		name:               "update - user changed from ClusterIP to None",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10"}),
		newService:         makeServiceWithClusterIp("None", []string{"10.0.0.10"}),
		expectedClusterIP:  "None",
		expectedClusterIPs: []string{"None"},
	}, {
		name:               "update - user changed from ClusterIP to None and changed ClusterIPs in a dual stack (new client making a mistake)",
		oldService:         makeServiceWithClusterIp("10.0.0.10", []string{"10.0.0.10", "2000::1"}),
		newService:         makeServiceWithClusterIp("None", []string{"10.0.0.11", "2000::1"}),
		expectedClusterIP:  "None",
		expectedClusterIPs: []string{"10.0.0.11", "2000::1"},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			normalizeClusterIPs(After{tc.newService}, Before{tc.oldService})

			if tc.newService == nil {
				t.Fatalf("unexpected new service to be nil")
			}

			if tc.newService.Spec.ClusterIP != tc.expectedClusterIP {
				t.Fatalf("expected clusterIP [%v] got [%v]", tc.expectedClusterIP, tc.newService.Spec.ClusterIP)
			}

			if len(tc.newService.Spec.ClusterIPs) != len(tc.expectedClusterIPs) {
				t.Fatalf("expected  clusterIPs %v got %v", tc.expectedClusterIPs, tc.newService.Spec.ClusterIPs)
			}

			for idx, clusterIP := range tc.newService.Spec.ClusterIPs {
				if clusterIP != tc.expectedClusterIPs[idx] {
					t.Fatalf("expected clusterIP [%v] at index[%v] got [%v]", tc.expectedClusterIPs[idx], idx, tc.newService.Spec.ClusterIPs[idx])

				}
			}
		})
	}
}

func TestPatchAllocatedValues(t *testing.T) {
	testCases := []struct {
		name                    string
		before                  *api.Service
		update                  *api.Service
		expectSameClusterIPs    bool
		expectReducedClusterIPs bool
		expectSameNodePort      bool
		expectSameHCNP          bool
	}{{
		name: "all_patched",
		before: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		expectSameClusterIPs: true,
		expectSameNodePort:   true,
		expectSameHCNP:       true,
	}, {
		name: "IPs_patched",
		before: svctest.MakeService("foo",
			svctest.SetTypeClusterIP,
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			// these are not valid, but prove the test
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeClusterIP),
		expectSameClusterIPs: true,
	}, {
		name: "NPs_patched",
		before: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			// this is not valid, but proves the test
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetClusterIPs("10.0.0.93", "2000::76")),
		expectSameClusterIPs: true,
		expectSameNodePort:   true,
	}, {
		name: "HCNP_patched",
		before: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts),
		expectSameClusterIPs: true,
		expectSameNodePort:   true,
		expectSameHCNP:       true,
	}, {
		name: "nothing_patched",
		before: svctest.MakeService("foo",
			svctest.SetTypeExternalName,
			// these are not valid, but prove the test
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeExternalName,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
	}, {
		name: "reset_NodePort",
		before: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetNodePorts(0)),
		expectSameClusterIPs: true,
		expectSameNodePort:   false,
	}, {
		name: "reset_partial_NodePorts",
		before: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetPorts(
				svctest.MakeServicePort("", 93, intstr.FromInt32(76), api.ProtocolTCP),
				svctest.MakeServicePort("", 94, intstr.FromInt32(76), api.ProtocolTCP),
			),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetPorts(
				svctest.MakeServicePort("", 93, intstr.FromInt32(76), api.ProtocolTCP),
				svctest.MakeServicePort("", 94, intstr.FromInt32(76), api.ProtocolTCP),
			),
			svctest.SetUniqueNodePorts,
			func(service *api.Service) {
				service.Spec.Ports[1].NodePort = 0
			}),
		expectSameClusterIPs: true,
		expectSameNodePort:   false,
	}, {
		name: "keep_NodePort",
		before: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(true),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetUniqueNodePorts,
			svctest.SetHealthCheckNodePort(31234)),
		update: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(true),
			svctest.SetClusterIPs("10.0.0.93", "2000::76"),
			svctest.SetNodePorts(0)),
		expectSameClusterIPs: true,
		expectSameNodePort:   true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			update := tc.update.DeepCopy()
			patchAllocatedValues(After{update}, Before{tc.before})

			beforeIP := tc.before.Spec.ClusterIP
			updateIP := update.Spec.ClusterIP
			if tc.expectSameClusterIPs || tc.expectReducedClusterIPs {
				if beforeIP != updateIP {
					t.Errorf("expected clusterIP to be patched: %q != %q", beforeIP, updateIP)
				}
			} else if beforeIP == updateIP {
				t.Errorf("expected clusterIP to not be patched: %q == %q", beforeIP, updateIP)
			}

			beforeIPs := tc.before.Spec.ClusterIPs
			updateIPs := update.Spec.ClusterIPs
			if tc.expectSameClusterIPs {
				if !cmp.Equal(beforeIPs, updateIPs) {
					t.Errorf("expected clusterIPs to be patched: %q != %q", beforeIPs, updateIPs)
				}
			} else if tc.expectReducedClusterIPs {
				if len(updateIPs) != 1 || beforeIPs[0] != updateIPs[0] {
					t.Errorf("expected clusterIPs to be trim-patched: %q -> %q", beforeIPs, updateIPs)
				}
			} else if cmp.Equal(beforeIPs, updateIPs) {
				t.Errorf("expected clusterIPs to not be patched: %q == %q", beforeIPs, updateIPs)
			}

			bNodePorts, uNodePorts := make([]int32, 0), make([]int32, 0)
			for _, item := range tc.before.Spec.Ports {
				bNodePorts = append(bNodePorts, item.NodePort)
			}
			for _, item := range update.Spec.Ports {
				uNodePorts = append(uNodePorts, item.NodePort)
			}
			if tc.expectSameNodePort && !reflect.DeepEqual(bNodePorts, uNodePorts) {
				t.Errorf("expected nodePort to be patched: %v != %v", bNodePorts, uNodePorts)
			} else if !tc.expectSameNodePort && reflect.DeepEqual(bNodePorts, uNodePorts) {
				t.Errorf("expected nodePort to not be patched: %v == %v", bNodePorts, uNodePorts)
			}

			if b, u := tc.before.Spec.HealthCheckNodePort, update.Spec.HealthCheckNodePort; tc.expectSameHCNP && b != u {
				t.Errorf("expected healthCheckNodePort to be patched: %d != %d", b, u)
			} else if !tc.expectSameHCNP && b == u {
				t.Errorf("expected healthCheckNodePort to not be patched: %d == %d", b, u)
			}
		})
	}
}

func TestServiceDefaultOnRead(t *testing.T) {
	// Helper makes a mostly-valid ServiceList.  Test-cases can tweak it as needed.
	makeServiceList := func(tweaks ...svctest.Tweak) *api.ServiceList {
		svc := svctest.MakeService("foo", tweaks...)
		list := &api.ServiceList{
			Items: []api.Service{*svc},
		}
		return list
	}

	testCases := []struct {
		name   string
		input  runtime.Object
		expect runtime.Object
	}{{
		name:  "single v4",
		input: svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:  "single v6",
		input: svctest.MakeService("foo", svctest.SetClusterIPs("2000::1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("2000::1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv6Protocol)),
	}, {
		name:  "missing clusterIPs v4",
		input: svctest.MakeService("foo", svctest.SetClusterIP("10.0.0.1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:  "missing clusterIPs v6",
		input: svctest.MakeService("foo", svctest.SetClusterIP("2000::1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("2000::1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv6Protocol)),
	}, {
		name:  "list v4",
		input: makeServiceList(svctest.SetClusterIPs("10.0.0.1")),
		expect: makeServiceList(svctest.SetClusterIPs("10.0.0.1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:  "list missing clusterIPs v4",
		input: makeServiceList(svctest.SetClusterIP("10.0.0.1")),
		expect: makeServiceList(svctest.SetClusterIPs("10.0.0.1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name:   "external name",
		input:  makeServiceList(svctest.SetTypeExternalName, svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster)),
		expect: makeServiceList(svctest.SetTypeExternalName),
	}, {
		name:  "dual v4v6",
		input: svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1", "2000::1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1", "2000::1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
	}, {
		name:  "dual v6v4",
		input: svctest.MakeService("foo", svctest.SetClusterIPs("2000::1", "10.0.0.1")),
		expect: svctest.MakeService("foo", svctest.SetClusterIPs("2000::1", "10.0.0.1"),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
	}, {
		name:  "headless",
		input: svctest.MakeService("foo", svctest.SetHeadless),
		expect: svctest.MakeService("foo", svctest.SetHeadless,
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name: "headless selectorless",
		input: svctest.MakeService("foo", svctest.SetHeadless,
			svctest.SetSelector(map[string]string{})),
		expect: svctest.MakeService("foo", svctest.SetHeadless,
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
	}, {
		name: "headless selectorless pre-set",
		input: svctest.MakeService("foo", svctest.SetHeadless,
			svctest.SetSelector(map[string]string{}),
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv6Protocol)),
		expect: svctest.MakeService("foo", svctest.SetHeadless,
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
			svctest.SetIPFamilies(api.IPv6Protocol)),
	}, {
		name:  "not Service or ServiceList",
		input: &api.Pod{},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			tmp := tc.input.DeepCopyObject()
			storage.defaultOnRead(tmp)

			svc, ok := tmp.(*api.Service)
			if !ok {
				list, ok := tmp.(*api.ServiceList)
				if !ok {
					return
				}
				svc = &list.Items[0]
			}

			exp, ok := tc.expect.(*api.Service)
			if !ok {
				list, ok := tc.expect.(*api.ServiceList)
				if !ok {
					return
				}
				exp = &list.Items[0]
			}

			// Verify fields we know are affected
			if want, got := exp.Spec.ClusterIP, svc.Spec.ClusterIP; want != got {
				t.Errorf("clusterIP: expected %v, got %v", want, got)
			}
			if want, got := exp.Spec.ClusterIPs, svc.Spec.ClusterIPs; !reflect.DeepEqual(want, got) {
				t.Errorf("clusterIPs: expected %v, got %v", want, got)
			}
			if want, got := fmtIPFamilyPolicy(exp.Spec.IPFamilyPolicy), fmtIPFamilyPolicy(svc.Spec.IPFamilyPolicy); want != got {
				t.Errorf("ipFamilyPolicy: expected %v, got %v", want, got)
			}
			if want, got := exp.Spec.IPFamilies, svc.Spec.IPFamilies; !reflect.DeepEqual(want, got) {
				t.Errorf("ipFamilies: expected %v, got %v", want, got)
			}
			if want, got := fmtInternalTrafficPolicy(exp.Spec.InternalTrafficPolicy), fmtInternalTrafficPolicy(svc.Spec.InternalTrafficPolicy); want != got {
				t.Errorf("internalTrafficPolicy: expected %v, got %v", want, got)
			}
		})
	}
}

//
// Scaffolding for create-update-delete tests.  Many tests can and should be
// written in terms of this.
//

type cudTestCase struct {
	name         string
	line         string // if not empty, will be logged with errors, use line() to set
	create       svcTestCase
	beforeUpdate func(t *testing.T, storage *wrapperRESTForTests)
	update       svcTestCase
}

type svcTestCase struct {
	svc         *api.Service
	expectError bool

	// We could calculate these by looking at the Service, but that's a
	// vector for test bugs and more importantly it makes the test cases less
	// self-documenting.
	expectClusterIPs          bool
	expectStackDowngrade      bool
	expectHeadless            bool
	expectNodePorts           bool
	expectHealthCheckNodePort bool

	// Additional proofs, provided by the tests which use this.
	prove []svcTestProof
}

type svcTestProof func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service)

// Most tests will call this.
func helpTestCreateUpdateDelete(t *testing.T, testCases []cudTestCase) {
	t.Helper()
	helpTestCreateUpdateDeleteWithFamilies(t, testCases, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
}

func helpTestCreateUpdateDeleteWithFamilies(t *testing.T, testCases []cudTestCase, ipFamilies []api.IPFamily) {
	// NOTE: do not call t.Helper() here.  It's more useful for errors to be
	// attributed to lines in this function than the caller of it.

	storage, _, server := newStorage(t, ipFamilies)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	for _, tc := range testCases {
		name := tc.name
		if tc.line != "" {
			name += "__@L" + tc.line
		}
		t.Run(name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()

			// Create the object as specified and check the results.
			obj, err := storage.Create(ctx, tc.create.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if tc.create.expectError && err != nil {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			defer storage.Delete(ctx, tc.create.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{}) // in case
			if tc.create.expectError && err == nil {
				t.Fatalf("unexpected success creating service")
			}
			createdSvc := obj.(*api.Service)
			if !verifyEquiv(t, "create", &tc.create, createdSvc) {
				return
			}
			verifyExpectations(t, storage, tc.create, tc.create.svc, createdSvc)
			lastSvc := createdSvc

			// The update phase is optional.
			if tc.update.svc != nil {
				// Allow callers to do something between create and update.
				if tc.beforeUpdate != nil {
					tc.beforeUpdate(t, storage)
				}

				// Update the object to the new state and check the results.
				obj, created, err := storage.Update(ctx, tc.update.svc.Name,
					rest.DefaultUpdatedObjectInfo(tc.update.svc), rest.ValidateAllObjectFunc,
					rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
				if tc.update.expectError && err != nil {
					return
				}
				if err != nil {
					t.Fatalf("unexpected error updating service: %v", err)
				}
				if tc.update.expectError && err == nil {
					t.Fatalf("unexpected success updating service")
				}
				if created {
					t.Fatalf("unexpected create-on-update")
				}
				updatedSvc := obj.(*api.Service)
				if !verifyEquiv(t, "update", &tc.update, updatedSvc) {
					return
				}
				verifyExpectations(t, storage, tc.update, createdSvc, updatedSvc)
				lastSvc = updatedSvc
			}

			// Delete the object and check the results.
			_, _, err = storage.Delete(ctx, tc.create.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error deleting service: %v", err)
			}
			verifyExpectations(t, storage, svcTestCase{ /* all false */ }, lastSvc, nil)
		})
	}
}

// line returns the line number of the caller, if possible.  This is useful in
// tests with a large number of cases - when something goes wrong you can find
// which case more easily.
func line() string {
	_, _, line, ok := stdruntime.Caller(1)
	var s string
	if ok {
		s = fmt.Sprintf("%d", line)
	} else {
		s = "<??>"
	}
	return s
}

// This makes the test-helpers testable.
type testingTInterface interface {
	Helper()
	Errorf(format string, args ...interface{})
}

type fakeTestingT struct {
	t *testing.T
}

func (f fakeTestingT) Helper() {}

func (f fakeTestingT) Errorf(format string, args ...interface{}) {}

func verifyEquiv(t testingTInterface, call string, tc *svcTestCase, got *api.Service) bool {
	t.Helper()

	// For when we compare objects.
	options := []cmp.Option{
		// These are system-assigned values, we don't need to compare them.
		cmpopts.IgnoreFields(api.Service{}, "UID", "ResourceVersion", "CreationTimestamp"),
		// Treat nil slices and empty slices as the same (e.g. clusterIPs).
		cmpopts.EquateEmpty(),
	}

	// For allocated fields, we want to be able to compare cleanly whether the
	// input specified values or not.
	want := tc.svc.DeepCopy()
	if tc.expectClusterIPs || tc.expectHeadless {
		if want.Spec.ClusterIP == "" {
			want.Spec.ClusterIP = got.Spec.ClusterIP
		}
		if want.Spec.IPFamilyPolicy == nil {
			want.Spec.IPFamilyPolicy = got.Spec.IPFamilyPolicy
		}
		if tc.expectStackDowngrade && len(want.Spec.ClusterIPs) > len(got.Spec.ClusterIPs) {
			want.Spec.ClusterIPs = want.Spec.ClusterIPs[0:1]
		} else if len(got.Spec.ClusterIPs) > len(want.Spec.ClusterIPs) {
			want.Spec.ClusterIPs = append(want.Spec.ClusterIPs, got.Spec.ClusterIPs[len(want.Spec.ClusterIPs):]...)
		}
		if tc.expectStackDowngrade && len(want.Spec.IPFamilies) > len(got.Spec.ClusterIPs) {
			want.Spec.IPFamilies = want.Spec.IPFamilies[0:1]
		} else if len(got.Spec.IPFamilies) > len(want.Spec.IPFamilies) {
			want.Spec.IPFamilies = append(want.Spec.IPFamilies, got.Spec.IPFamilies[len(want.Spec.IPFamilies):]...)
		}
	}

	if tc.expectNodePorts {
		for i := range want.Spec.Ports {
			p := &want.Spec.Ports[i]
			if p.NodePort == 0 {
				p.NodePort = got.Spec.Ports[i].NodePort
			}
		}
	}
	if tc.expectHealthCheckNodePort {
		if want.Spec.HealthCheckNodePort == 0 {
			want.Spec.HealthCheckNodePort = got.Spec.HealthCheckNodePort
		}
	}

	if !cmp.Equal(want, got, options...) {
		t.Errorf("unexpected result from %s:\n%s", call, cmp.Diff(want, got, options...))
		return false
	}
	return true
}

// Quis custodiet ipsos custodes?
func TestVerifyEquiv(t *testing.T) {
	testCases := []struct {
		name   string
		input  svcTestCase
		output *api.Service
		expect bool
	}{{
		name: "ExternalName",
		input: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		output: svctest.MakeService("foo", svctest.SetTypeExternalName),
		expect: true,
	}, {
		name: "ClusterIPs_unspecified",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1", "2000:1")),
		expect: true,
	}, {
		name: "ClusterIPs_specified",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1", "2000:1")),
			expectClusterIPs: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1", "2000:1")),
		expect: true,
	}, {
		name: "ClusterIPs_wrong",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.0", "2000:0")),
			expectClusterIPs: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1", "2000:1")),
		expect: false,
	}, {
		name: "ClusterIPs_partial",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeClusterIP, svctest.SetClusterIPs("10.0.0.1", "2000:1")),
		expect: true,
	}, {
		name: "NodePort_unspecified",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeNodePort, svctest.SetUniqueNodePorts),
		expect: true,
	}, {
		name: "NodePort_specified",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort, svctest.SetNodePorts(93)),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeNodePort, svctest.SetNodePorts(93)),
		expect: true,
	}, {
		name: "NodePort_wrong",
		input: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort, svctest.SetNodePorts(93)),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeNodePort, svctest.SetNodePorts(76)),
		expect: false,
	}, {
		name: "NodePort_partial",
		input: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
				svctest.SetNodePorts(93)),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetNodePorts(93, 76)),
		expect: true,
	}, {
		name: "HealthCheckNodePort_unspecified",
		input: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetHealthCheckNodePort(93)),
		expect: true,
	}, {
		name: "HealthCheckNodePort_specified",
		input: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(93)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetHealthCheckNodePort(93)),
		expect: true,
	}, {
		name: "HealthCheckNodePort_wrong",
		input: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(93)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		output: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
			svctest.SetHealthCheckNodePort(76)),
		expect: false,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result := verifyEquiv(fakeTestingT{t}, "test", &tc.input, tc.output)
			if result != tc.expect {
				t.Errorf("expected %v, got %v", tc.expect, result)
			}
		})
	}
}

func verifyExpectations(t *testing.T, storage *wrapperRESTForTests, tc svcTestCase, before, after *api.Service) {
	t.Helper()

	if tc.expectClusterIPs {
		proveClusterIPsAllocated(t, storage, before, after)
	} else if tc.expectHeadless {
		proveHeadless(t, storage, before, after)
	} else {
		proveClusterIPsDeallocated(t, storage, before, after)
	}
	if tc.expectNodePorts {
		proveNodePortsAllocated(t, storage, before, after)
	} else {
		proveNodePortsDeallocated(t, storage, before, after)
	}
	if tc.expectHealthCheckNodePort {
		proveHealthCheckNodePortAllocated(t, storage, before, after)
	} else {
		proveHealthCheckNodePortDeallocated(t, storage, before, after)
	}

	for _, p := range tc.prove {
		p(t, storage, before, after)
	}
}

func callName(before, after *api.Service) string {
	if before == nil && after != nil {
		return "create"
	}
	if before != nil && after != nil {
		return "update"
	}
	if before != nil && after == nil {
		return "delete"
	}
	panic("this test is broken: before and after are both nil")
}

func ipIsAllocated(t *testing.T, alloc ipallocator.Interface, ipstr string) bool {
	t.Helper()
	ip := netutils.ParseIPSloppy(ipstr)
	if ip == nil {
		t.Errorf("error parsing IP %q", ipstr)
		return false
	}
	return alloc.Has(ip)
}

func portIsAllocated(t *testing.T, alloc portallocator.Interface, port int32) bool {
	t.Helper()
	if port == 0 {
		t.Errorf("port is 0")
		return false
	}
	return alloc.Has(int(port))
}

func proveClusterIPsAllocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if sing, plur := after.Spec.ClusterIP, after.Spec.ClusterIPs[0]; sing != plur {
		t.Errorf("%s: expected clusterIP == clusterIPs[0]: %q != %q", callName(before, after), sing, plur)
	}

	for _, clip := range after.Spec.ClusterIPs {
		if !ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[familyOf(clip)], clip) {
			t.Errorf("%s: expected clusterIP to be allocated: %q", callName(before, after), clip)
		}
	}

	if lc, lf := len(after.Spec.ClusterIPs), len(after.Spec.IPFamilies); lc != lf {
		t.Errorf("%s: expected same number of clusterIPs and ipFamilies: %d != %d", callName(before, after), lc, lf)
	}

	for i, fam := range after.Spec.IPFamilies {
		if want, got := fam, familyOf(after.Spec.ClusterIPs[i]); want != got {
			t.Errorf("%s: clusterIP is the wrong IP family: want %s, got %s", callName(before, after), want, got)
		}
	}

	if after.Spec.IPFamilyPolicy == nil {
		t.Errorf("%s: expected ipFamilyPolicy to be set", callName(before, after))
	} else {
		pol := *after.Spec.IPFamilyPolicy
		fams := len(after.Spec.IPFamilies)
		clus := 1
		if storage.secondaryIPFamily != "" {
			clus = 2
		}
		if pol == api.IPFamilyPolicySingleStack && fams != 1 {
			t.Errorf("%s: expected 1 ipFamily, got %d", callName(before, after), fams)
		} else if pol == api.IPFamilyPolicyRequireDualStack && fams != 2 {
			t.Errorf("%s: expected 2 ipFamilies, got %d", callName(before, after), fams)
		} else if pol == api.IPFamilyPolicyPreferDualStack && fams != clus {
			t.Errorf("%s: expected %d ipFamilies, got %d", callName(before, after), clus, fams)
		}
	}

	if before != nil {
		if before.Spec.ClusterIP != "" {
			if want, got := before.Spec.ClusterIP, after.Spec.ClusterIP; want != got {
				t.Errorf("%s: wrong clusterIP: wanted %q, got %q", callName(before, after), want, got)
			}
		}
		min := func(x, y int) int {
			if x < y {
				return x
			}
			return y
		}
		for i := 0; i < min(len(before.Spec.ClusterIPs), len(after.Spec.ClusterIPs)); i++ {
			if want, got := before.Spec.ClusterIPs[i], after.Spec.ClusterIPs[i]; want != got {
				t.Errorf("%s: wrong clusterIPs[%d]: wanted %q, got %q", callName(before, after), i, want, got)
			}
		}
		for i := 0; i < min(len(before.Spec.IPFamilies), len(after.Spec.IPFamilies)); i++ {
			if want, got := before.Spec.IPFamilies[i], after.Spec.IPFamilies[i]; want != got {
				t.Errorf("%s: wrong ipFamilies[%d]: wanted %q, got %q", callName(before, after), i, want, got)
			}
		}
	}
}

func proveClusterIPsDeallocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if after != nil && after.Spec.ClusterIP != api.ClusterIPNone {
		if after.Spec.ClusterIP != "" {
			t.Errorf("%s: expected clusterIP to be unset: %q", callName(before, after), after.Spec.ClusterIP)
		}
		if len(after.Spec.ClusterIPs) != 0 {
			t.Errorf("%s: expected clusterIPs to be unset: %q", callName(before, after), after.Spec.ClusterIPs)
		}
	}

	if before != nil && before.Spec.ClusterIP != api.ClusterIPNone {
		for _, clip := range before.Spec.ClusterIPs {
			if ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[familyOf(clip)], clip) {
				t.Errorf("%s: expected clusterIP to be deallocated: %q", callName(before, after), clip)
			}
		}
	}
}

func proveHeadless(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if sing, plur := after.Spec.ClusterIP, after.Spec.ClusterIPs[0]; sing != plur {
		t.Errorf("%s: expected clusterIP == clusterIPs[0]: %q != %q", callName(before, after), sing, plur)
	}
	if len(after.Spec.ClusterIPs) != 1 || after.Spec.ClusterIPs[0] != api.ClusterIPNone {
		t.Errorf("%s: expected clusterIPs to be [%q]: %q", callName(before, after), api.ClusterIPNone, after.Spec.ClusterIPs)
	}
}

func proveNodePortsAllocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	for _, p := range after.Spec.Ports {
		if !portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
			t.Errorf("%s: expected nodePort to be allocated: %d", callName(before, after), p.NodePort)
		}
	}
}

func proveNodePortsDeallocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if after != nil {
		for _, p := range after.Spec.Ports {
			if p.NodePort != 0 {
				t.Errorf("%s: expected nodePort to be unset: %d", callName(before, after), p.NodePort)
			}
		}
	}

	if before != nil {
		for _, p := range before.Spec.Ports {
			if p.NodePort != 0 && portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
				t.Errorf("%s: expected nodePort to be deallocated: %d", callName(before, after), p.NodePort)
			}
		}
	}
}

func proveHealthCheckNodePortAllocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if !portIsAllocated(t, storage.alloc.serviceNodePorts, after.Spec.HealthCheckNodePort) {
		t.Errorf("%s: expected healthCheckNodePort to be allocated: %d", callName(before, after), after.Spec.HealthCheckNodePort)
	}
}

func proveHealthCheckNodePortDeallocated(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
	t.Helper()

	if after != nil {
		if after.Spec.HealthCheckNodePort != 0 {
			t.Errorf("%s: expected healthCheckNodePort to be unset: %d", callName(before, after), after.Spec.HealthCheckNodePort)
		}
	}

	if before != nil {
		if before.Spec.HealthCheckNodePort != 0 && portIsAllocated(t, storage.alloc.serviceNodePorts, before.Spec.HealthCheckNodePort) {
			t.Errorf("%s: expected healthCheckNodePort to be deallocated: %d", callName(before, after), before.Spec.HealthCheckNodePort)
		}
	}
}

//
// functional tests of the registry
//

func fmtIPFamilyPolicy(pol *api.IPFamilyPolicy) string {
	if pol == nil {
		return "<nil>"
	}
	return string(*pol)
}

func fmtInternalTrafficPolicy(pol *api.ServiceInternalTrafficPolicy) string {
	if pol == nil {
		return "<nil>"
	}
	return string(*pol)
}

func fmtIPFamilies(fams []api.IPFamily) string {
	if fams == nil {
		return "[]"
	}
	return fmt.Sprintf("%v", fams)
}

// Prove that create ignores IP and IPFamily stuff when type is ExternalName.
func TestCreateIgnoresIPsForExternalName(t *testing.T) {
	type testCase struct {
		name        string
		svc         *api.Service
		expectError bool
	}
	// These cases were chosen from the full gamut to ensure all "interesting"
	// cases are covered.
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		cases           []testCase
	}{{
		name:            "singlestack:v6",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		cases: []testCase{{
			name: "Policy:unset_Families:unset",
			svc:  svctest.MakeService("foo"),
		}, {
			name: "Policy:SingleStack_Families:v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:PreferDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:RequireDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		}},
	}, {
		name:            "dualstack:v6v4",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		cases: []testCase{{
			name: "Policy:unset_Families:unset",
			svc:  svctest.MakeService("foo"),
		}, {
			name: "Policy:SingleStack_Families:v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		}, {
			name: "Policy:PreferDualStack_Families:v6v4",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		}, {
			name: "Policy:RequireDualStack_Families:v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		}},
	}}

	for _, otc := range testCases {
		t.Run(otc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, otc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			for _, itc := range otc.cases {
				t.Run(itc.name, func(t *testing.T) {
					// This test is ONLY ExternalName services.
					itc.svc.Spec.Type = api.ServiceTypeExternalName
					itc.svc.Spec.ExternalName = "example.com"

					ctx := genericapirequest.NewDefaultContext()
					createdObj, err := storage.Create(ctx, itc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
					if itc.expectError && err != nil {
						return
					}
					if err != nil {
						t.Fatalf("unexpected error creating service: %v", err)
					}
					defer storage.Delete(ctx, itc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
					if itc.expectError && err == nil {
						t.Fatalf("unexpected success creating service")
					}
					createdSvc := createdObj.(*api.Service)

					if want, got := fmtIPFamilyPolicy(nil), fmtIPFamilyPolicy(createdSvc.Spec.IPFamilyPolicy); want != got {
						t.Errorf("wrong IPFamilyPolicy: want %s, got %s", want, got)
					}
					if want, got := fmtIPFamilies(nil), fmtIPFamilies(createdSvc.Spec.IPFamilies); want != got {
						t.Errorf("wrong IPFamilies: want %s, got %s", want, got)
					}
					if len(createdSvc.Spec.ClusterIP) != 0 {
						t.Errorf("expected no clusterIP, got %q", createdSvc.Spec.ClusterIP)
					}
					if len(createdSvc.Spec.ClusterIPs) != 0 {
						t.Errorf("expected no clusterIPs, got %q", createdSvc.Spec.ClusterIPs)
					}
				})
			}
		})
	}
}

// Prove that create initializes clusterIPs from clusterIP.  This simplifies
// later tests to not need to re-prove this.
func TestCreateInitClusterIPsFromClusterIP(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		svc             *api.Service
	}{{
		name:            "singlestack:v4_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v4_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("10.0.0.1")),
	}, {
		name:            "singlestack:v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("2000::1")),
	}, {
		name:            "dualstack:v4v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v4v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("10.0.0.1")),
	}, {
		name:            "dualstack:v6v4_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v6v4_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIP("2000::1")),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			if createdSvc.Spec.ClusterIP == "" {
				t.Errorf("expected ClusterIP to be set")

			}
			if tc.svc.Spec.ClusterIP != "" {
				if want, got := tc.svc.Spec.ClusterIP, createdSvc.Spec.ClusterIP; want != got {
					t.Errorf("wrong ClusterIP: want %s, got %s", want, got)
				}
			}
			if len(createdSvc.Spec.ClusterIPs) == 0 {
				t.Errorf("expected ClusterIPs to be set")
			}
			if want, got := createdSvc.Spec.ClusterIP, createdSvc.Spec.ClusterIPs[0]; want != got {
				t.Errorf("wrong ClusterIPs[0]: want %s, got %s", want, got)
			}
		})
	}
}

// Prove that create initializes IPFamily fields correctly.
func TestCreateInitIPFields(t *testing.T) {
	type testCase struct {
		name           string
		line           string
		svc            *api.Service
		expectError    bool
		expectPolicy   api.IPFamilyPolicy
		expectFamilies []api.IPFamily
		expectHeadless bool
	}
	// These cases were chosen from the full gamut to ensure all "interesting"
	// cases are covered.
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		cases           []testCase
	}{
		{
			name:            "singlestack:v4",
			clusterFamilies: []api.IPFamily{api.IPv4Protocol},
			cases: []testCase{
				//----------------------------------------
				// singlestack:v4 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					line:           line(),
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v4 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "singlestack:v6",
			clusterFamilies: []api.IPFamily{api.IPv6Protocol},
			cases: []testCase{
				//----------------------------------------
				// singlestack:v6 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					line:           line(),
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// singlestack:v6 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "dualstack:v4v6",
			clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
			cases: []testCase{
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					line:           line(),
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v4v6 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v4v6 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
				//----------------------------------------
				// dualstack:v4v6 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		}, {
			name:            "dualstack:v6v4",
			clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
			cases: []testCase{
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:unset
				//----------------------------------------
				{
					name:           "ClusterIPs:unset_Policy:unset_Families:unset",
					line:           line(),
					svc:            svctest.MakeService("foo"),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:unset_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v4
				//----------------------------------------
				{
					name: "ClusterIPs:v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v6
				//----------------------------------------
				{
					name: "ClusterIPs:v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1")),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v4v6
				//----------------------------------------
				{
					name: "ClusterIPs:v4v6_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
				}, {
					name: "ClusterIPs:v4v6_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("10.0.0.1", "2000::1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				},
				//----------------------------------------
				// dualstack:v6v4 ClusterIPs:v6v4
				//----------------------------------------
				{
					name: "ClusterIPs:v6v4_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1")),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "ClusterIPs:v6v4_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetClusterIPs("2000::1", "10.0.0.1"),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
				},
				//----------------------------------------
				// dualstack:v6v4 Headless
				//----------------------------------------
				{
					name: "Headless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "Headless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
				//----------------------------------------
				// dualstack:v6v4 HeadlessSelectorless
				//----------------------------------------
				{
					name: "HeadlessSelectorless_Policy:unset_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:unset_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicySingleStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:SingleStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectError: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:PreferDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyPreferDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:unset",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v4v6",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
					expectHeadless: true,
				}, {
					name: "HeadlessSelectorless_Policy:RequireDualStack_Families:v6v4",
					line: line(),
					svc: svctest.MakeService("foo",
						svctest.SetHeadless,
						svctest.SetSelector(nil),
						svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
						svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
					expectPolicy:   api.IPFamilyPolicyRequireDualStack,
					expectFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
					expectHeadless: true,
				},
			},
		},
	}

	for _, otc := range testCases {
		t.Run(otc.name, func(t *testing.T) {

			// Do this in the outer loop for performance.
			storage, _, server := newStorage(t, otc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			for _, itc := range otc.cases {
				t.Run(itc.name+"__@L"+itc.line, func(t *testing.T) {
					ctx := genericapirequest.NewDefaultContext()
					createdObj, err := storage.Create(ctx, itc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
					if itc.expectError && err != nil {
						return
					}
					if err != nil {
						t.Fatalf("unexpected error creating service: %v", err)
					}
					defer storage.Delete(ctx, itc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
					if itc.expectError && err == nil {
						t.Fatalf("unexpected success creating service")
					}
					createdSvc := createdObj.(*api.Service)

					if want, got := fmtIPFamilyPolicy(&itc.expectPolicy), fmtIPFamilyPolicy(createdSvc.Spec.IPFamilyPolicy); want != got {
						t.Errorf("wrong IPFamilyPolicy: want %s, got %s", want, got)
					}
					if want, got := fmtIPFamilies(itc.expectFamilies), fmtIPFamilies(createdSvc.Spec.IPFamilies); want != got {
						t.Errorf("wrong IPFamilies: want %s, got %s", want, got)
					}
					if itc.expectHeadless {
						proveHeadless(t, storage, nil, createdSvc)
						return
					}
					proveClusterIPsAllocated(t, storage, nil, createdSvc)
				})
			}
		})
	}
}

// There are enough corner-cases that it's useful to have a test that asserts
// the errors.  Some of these are in other tests, but this is clearer.
func TestCreateInvalidClusterIPInputs(t *testing.T) {
	testCases := []struct {
		name     string
		families []api.IPFamily
		svc      *api.Service
		expect   []string
	}{{
		name:     "bad_ipFamilyPolicy",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicy("garbage"))),
		expect: []string{"Unsupported value"},
	}, {
		name:     "requiredual_ipFamilyPolicy_on_singlestack",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
		expect: []string{"cluster is not configured for dual-stack"},
	}, {
		name:     "bad_ipFamilies_0_value",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPFamily("garbage"))),
		expect: []string{"Unsupported value"},
	}, {
		name:     "bad_ipFamilies_1_value",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPFamily("garbage"))),
		expect: []string{"Unsupported value"},
	}, {
		name:     "bad_ipFamilies_2_value",
		families: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol, api.IPFamily("garbage"))),
		expect: []string{"Unsupported value"},
	}, {
		name:     "wrong_ipFamily",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv6Protocol)),
		expect: []string{"not configured on this cluster"},
	}, {
		name:     "too_many_ipFamilies_on_singlestack",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
		expect: []string{"when multiple IP families are specified"},
	}, {
		name:     "dup_ipFamily_singlestack",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv4Protocol)),
		expect: []string{"Duplicate value"},
	}, {
		name:     "dup_ipFamily_dualstack",
		families: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol, api.IPv6Protocol)),
		expect: []string{"Duplicate value"},
	}, {
		name:     "bad_IP",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("garbage")),
		expect: []string{"must be a valid IP"},
	}, {
		name:     "IP_wrong_family",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("2000::1")),
		expect: []string{"not configured on this cluster"},
	}, {
		name:     "IP_doesnt_match_family",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetIPFamilies(api.IPv4Protocol),
			svctest.SetClusterIPs("2000::1")),
		expect: []string{"expected an IPv4 value as indicated"},
	}, {
		name:     "too_many_IPs_singlestack",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("10.0.0.1", "10.0.0.2")),
		expect: []string{"no more than one IP for each IP family"},
	}, {
		name:     "too_many_IPs_dualstack",
		families: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("10.0.0.1", "2000::1", "10.0.0.2")),
		expect: []string{"only hold up to 2 values"},
	}, {
		name:     "dup_IPs",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("10.0.0.1", "10.0.0.1")),
		expect: []string{"no more than one IP for each IP family"},
	}, {
		name:     "empty_IP",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("")),
		expect: []string{"must be empty when", "must be a valid IP"},
	}, {
		name:     "None_IP_1",
		families: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo",
			svctest.SetClusterIPs("10.0.0.1", "None")),
		expect: []string{"must be a valid IP"},
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.families)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			_, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err == nil {
				t.Fatalf("unexpected success creating service")
			}
			for _, s := range tc.expect {
				if !strings.Contains(err.Error(), s) {
					t.Errorf("expected to find %q in the error:\n  %s", s, err.Error())
				}
			}
		})
	}
}

func TestCreateDeleteReuse(t *testing.T) {
	testCases := []struct {
		name string
		svc  *api.Service
	}{{
		name: "v4",
		svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
			svctest.SetIPFamilies(api.IPv4Protocol)),
	}, {
		name: "v6",
		svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
			svctest.SetIPFamilies(api.IPv6Protocol)),
	}, {
		name: "v4v6",
		svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
			svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
			svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()

			// Create it
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			// Ensure IPs and ports were allocated
			proveClusterIPsAllocated(t, storage, tc.svc, createdSvc)
			proveNodePortsAllocated(t, storage, tc.svc, createdSvc)

			// Delete it
			_, _, err = storage.Delete(ctx, tc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}

			// Ensure IPs and ports were deallocated
			proveClusterIPsDeallocated(t, storage, createdSvc, nil)
			proveNodePortsDeallocated(t, storage, createdSvc, nil)

			// Force the same IPs and ports
			svc2 := tc.svc.DeepCopy()
			svc2.Name += "2"
			svc2.Spec.ClusterIP = createdSvc.Spec.ClusterIP
			svc2.Spec.ClusterIPs = createdSvc.Spec.ClusterIPs
			svc2.Spec.Ports = createdSvc.Spec.Ports

			// Create again
			_, err = storage.Create(ctx, svc2, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}

			// Ensure IPs and ports were allocated
			proveClusterIPsAllocated(t, storage, svc2, createdSvc)
			proveNodePortsAllocated(t, storage, svc2, createdSvc)
		})
	}
}

func TestCreateInitNodePorts(t *testing.T) {
	testCases := []struct {
		name            string
		svc             *api.Service
		expectError     bool
		expectNodePorts bool
	}{{
		name:            "type:ExternalName",
		svc:             svctest.MakeService("foo"),
		expectNodePorts: false,
	}, {
		name: "type:ExternalName_with_NodePorts",
		svc: svctest.MakeService("foo",
			svctest.SetUniqueNodePorts),
		expectError: true,
	}, {
		name:            "type:ClusterIP",
		svc:             svctest.MakeService("foo"),
		expectNodePorts: false,
	}, {
		name: "type:ClusterIP_with_NodePorts",
		svc: svctest.MakeService("foo",
			svctest.SetUniqueNodePorts),
		expectError: true,
	}, {
		name: "type:NodePort_single_port_unspecified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_single_port_specified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort, svctest.SetUniqueNodePorts),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_unspecified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_specified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetUniqueNodePorts),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_same",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetNodePorts(30080, 30080)),
		expectError: true,
	}, {
		name: "type:NodePort_multiport_multiproto_unspecified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP))),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_multiproto_specified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP)),
			svctest.SetUniqueNodePorts),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_multiproto_same",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		expectNodePorts: true,
	}, {
		name: "type:NodePort_multiport_multiproto_conflict",
		svc: svctest.MakeService("foo",
			svctest.SetTypeNodePort,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 93, intstr.FromInt32(93), api.ProtocolTCP),
				svctest.MakeServicePort("q", 76, intstr.FromInt32(76), api.ProtocolUDP)),
			svctest.SetNodePorts(30093, 30093)),
		expectError: true,
	}, {
		name: "type:LoadBalancer_single_port_unspecified:on_alloc:false",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts: false,
	}, {
		name: "type:LoadBalancer_single_port_unspecified:on_alloc:true",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_single_port_specified:on_alloc:false",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetUniqueNodePorts,
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_single_port_specified:on_alloc:true",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetUniqueNodePorts,
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_unspecified:on_alloc:false",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts: false,
	}, {
		name: "type:LoadBalancer_multiport_unspecified:on_alloc:true",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_specified:on_alloc:false",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetUniqueNodePorts,
			svctest.SetAllocateLoadBalancerNodePorts(false)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_specified:on_alloc:true",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetUniqueNodePorts,
			svctest.SetAllocateLoadBalancerNodePorts(true)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_same",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP)),
			svctest.SetNodePorts(30080, 30080)),
		expectError: true,
	}, {
		name: "type:LoadBalancer_multiport_multiproto_unspecified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP))),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_multiproto_specified",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP)),
			svctest.SetUniqueNodePorts),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_multiproto_same",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
				svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP)),
			svctest.SetNodePorts(30053, 30053)),
		expectNodePorts: true,
	}, {
		name: "type:LoadBalancer_multiport_multiproto_conflict",
		svc: svctest.MakeService("foo",
			svctest.SetTypeLoadBalancer,
			svctest.SetPorts(
				svctest.MakeServicePort("p", 93, intstr.FromInt32(93), api.ProtocolTCP),
				svctest.MakeServicePort("q", 76, intstr.FromInt32(76), api.ProtocolUDP)),
			svctest.SetNodePorts(30093, 30093)),
		expectError: true,
	}}

	// Do this in the outer scope for performance.
	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if tc.expectError && err != nil {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			defer storage.Delete(ctx, tc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
			if tc.expectError && err == nil {
				t.Fatalf("unexpected success creating service")
			}
			createdSvc := createdObj.(*api.Service)

			// Produce a map of port index to nodeport value, excluding zero.
			ports := map[int]*api.ServicePort{}
			for i := range createdSvc.Spec.Ports {
				p := &createdSvc.Spec.Ports[i]
				if p.NodePort != 0 {
					ports[i] = p
				}
			}

			if tc.expectNodePorts && len(ports) == 0 {
				t.Fatalf("expected NodePorts to be allocated, found none")
			}
			if !tc.expectNodePorts && len(ports) > 0 {
				t.Fatalf("expected NodePorts to not be allocated, found %v", ports)
			}
			if !tc.expectNodePorts {
				return
			}

			// Make sure we got the right number of allocations
			if want, got := len(ports), len(tc.svc.Spec.Ports); want != got {
				t.Fatalf("expected %d NodePorts, found %d", want, got)
			}

			// Make sure they are all allocated
			for _, p := range ports {
				if !portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
					t.Errorf("expected port to be allocated: %v", p)
				}
			}

			// Make sure we got any specific allocations
			for i, p := range tc.svc.Spec.Ports {
				if p.NodePort != 0 {
					if ports[i].NodePort != p.NodePort {
						t.Errorf("expected Ports[%d].NodePort to be %d, got %d", i, p.NodePort, ports[i].NodePort)
					}
					// Remove requested ports from the set
					delete(ports, i)
				}
			}

			// Make sure any allocated ports are unique
			seen := map[int32]int32{}
			for i, p := range ports {
				// We allow the same NodePort for different protocols of the
				// same Port.
				if prev, found := seen[p.NodePort]; found && prev != p.Port {
					t.Errorf("found non-unique allocation in Ports[%d].NodePort: %d -> %d", i, p.NodePort, p.Port)
				}
				seen[p.NodePort] = p.Port
			}
		})
	}
}

// Prove that create skips allocations for Headless services.
func TestCreateSkipsAllocationsForHeadless(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		svc             *api.Service
		expectError     bool
	}{{
		name:            "singlestack:v4",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v6",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v4v6",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "dualstack:v6v4",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v4_type:NodePort",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeNodePort),
		expectError:     true,
	}, {
		name:            "singlestack:v6_type:LoadBalancer",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
		expectError:     true,
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			// This test is ONLY headless services.
			tc.svc.Spec.ClusterIP = api.ClusterIPNone

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if tc.expectError && err != nil {
				return
			}
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			if tc.expectError && err == nil {
				t.Fatalf("unexpected success creating service")
			}
			createdSvc := createdObj.(*api.Service)

			if createdSvc.Spec.ClusterIP != "None" {
				t.Errorf("expected clusterIP \"None\", got %q", createdSvc.Spec.ClusterIP)
			}
			if !reflect.DeepEqual(createdSvc.Spec.ClusterIPs, []string{"None"}) {
				t.Errorf("expected clusterIPs [\"None\"], got %q", createdSvc.Spec.ClusterIPs)
			}
		})
	}
}

// Prove that a dry-run create doesn't actually allocate IPs or ports.
func TestCreateDryRun(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		svc             *api.Service
	}{{
		name:            "singlestack:v4_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v4_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetClusterIPs("10.0.0.1")),
	}, {
		name:            "singlestack:v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo"),
	}, {
		name:            "singlestack:v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetClusterIPs("2000::1")),
	}, {
		name:            "dualstack:v4v6_clusterip:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
	}, {
		name:            "dualstack:v4v6_clusterip:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack), svctest.SetClusterIPs("10.0.0.1", "2000::1")),
	}, {
		name:            "singlestack:v4_type:NodePort_nodeport:unset",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeNodePort),
	}, {
		name:            "singlestack:v4_type:LoadBalancer_nodePort:set",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeLoadBalancer, svctest.SetUniqueNodePorts),
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			// Ensure IPs were assigned
			if netutils.ParseIPSloppy(createdSvc.Spec.ClusterIP) == nil {
				t.Errorf("expected valid clusterIP: %q", createdSvc.Spec.ClusterIP)
			}
			for _, ip := range createdSvc.Spec.ClusterIPs {
				if netutils.ParseIPSloppy(ip) == nil {
					t.Errorf("expected valid clusterIP: %q", createdSvc.Spec.ClusterIP)
				}
			}

			// Ensure the allocators are clean.
			proveClusterIPsDeallocated(t, storage, createdSvc, nil)
			if tc.svc.Spec.Type != api.ServiceTypeClusterIP {
				proveNodePortsDeallocated(t, storage, createdSvc, nil)
			}
		})
	}
}

func TestDeleteWithFinalizer(t *testing.T) {
	svcName := "foo"

	storage, _, server := newStorage(t, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	// This will allocate cluster IPs, NodePort, and HealthCheckNodePort.
	svc := svctest.MakeService(svcName, svctest.SetTypeLoadBalancer,
		svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
		func(s *api.Service) {
			s.Finalizers = []string{"example.com/test"}
		})

	ctx := genericapirequest.NewDefaultContext()

	// Create it with finalizer.
	obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error creating service: %v", err)
	}
	createdSvc := obj.(*api.Service)

	// Prove everything was allocated.
	obj, err = storage.Get(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error getting service: %v", err)
	}
	if !cmp.Equal(createdSvc, obj) {
		t.Errorf("expected the result of Create() and Get() to match: %v", cmp.Diff(createdSvc, obj))
	}
	proveClusterIPsAllocated(t, storage, svc, createdSvc)
	proveNodePortsAllocated(t, storage, svc, createdSvc)
	proveHealthCheckNodePortAllocated(t, storage, svc, createdSvc)

	// Try to delete it, but it should be blocked by the finalizer.
	obj, deleted, err := storage.Delete(ctx, svcName, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})
	if err != nil {
		t.Fatalf("unexpected error deleting service: %v", err)
	}
	if deleted {
		t.Fatalf("expected service to not be deleted")
	}
	deletedSvc := obj.(*api.Service)

	// Prove everything is still allocated.
	_, err = storage.Get(ctx, svcName, &metav1.GetOptions{})
	if err != nil {
		t.Fatalf("unexpected error getting service: %v", err)
	}
	proveClusterIPsAllocated(t, storage, svc, createdSvc)
	proveNodePortsAllocated(t, storage, svc, createdSvc)
	proveHealthCheckNodePortAllocated(t, storage, svc, createdSvc)

	// Clear the finalizer - should delete.
	deletedSvc.Finalizers = nil
	_, _, err = storage.Update(ctx, svcName,
		rest.DefaultUpdatedObjectInfo(deletedSvc), rest.ValidateAllObjectFunc,
		rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
	if err != nil {
		t.Fatalf("unexpected error updating service: %v", err)
	}

	// Prove everything is deallocated.
	_, err = storage.Get(ctx, svcName, &metav1.GetOptions{})
	if err == nil {
		t.Fatalf("unexpected success getting service")
	}
	proveClusterIPsDeallocated(t, storage, createdSvc, nil)
	proveNodePortsDeallocated(t, storage, createdSvc, nil)
	proveHealthCheckNodePortDeallocated(t, storage, createdSvc, nil)
}

// Prove that a dry-run delete doesn't actually deallocate IPs or ports.
func TestDeleteDryRun(t *testing.T) {
	testCases := []struct {
		name string
		svc  *api.Service
	}{
		{
			name: "v4",
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		},
		{
			name: "v4v6",
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {

			storage, _, server := newStorage(t, tc.svc.Spec.IPFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			createdObj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := createdObj.(*api.Service)

			// Ensure IPs and ports were allocated
			proveClusterIPsAllocated(t, storage, tc.svc, createdSvc)
			proveNodePortsAllocated(t, storage, tc.svc, createdSvc)
			proveHealthCheckNodePortAllocated(t, storage, tc.svc, createdSvc)

			_, _, err = storage.Delete(ctx, tc.svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("unexpected error deleting service: %v", err)
			}

			// Ensure they are still allocated.
			proveClusterIPsAllocated(t, storage, tc.svc, createdSvc)
			proveNodePortsAllocated(t, storage, tc.svc, createdSvc)
			proveHealthCheckNodePortAllocated(t, storage, tc.svc, createdSvc)
		})
	}
}

// Prove that a dry-run update doesn't actually allocate or deallocate IPs or ports.
func TestUpdateDryRun(t *testing.T) {
	testCases := []struct {
		name            string
		clusterFamilies []api.IPFamily
		svc             *api.Service
		update          *api.Service
		verifyDryAllocs bool
	}{{
		name:            "singlestack:v4_NoAllocs-Allocs",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeExternalName),
		update: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		verifyDryAllocs: true, // make sure values were not allocated.
	}, {
		name:            "singlestack:v4_Allocs-NoAllocs",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol},
		svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		update:          svctest.MakeService("foo", svctest.SetTypeExternalName),
		verifyDryAllocs: false, // make sure values were not released.
	}, {
		name:            "singlestack:v6_NoAllocs-Allocs",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeExternalName),
		update: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		verifyDryAllocs: true, // make sure values were not allocated.
	}, {
		name:            "singlestack:v6_Allocs-NoAllocs",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol},
		svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		update:          svctest.MakeService("foo", svctest.SetTypeExternalName),
		verifyDryAllocs: false, // make sure values were not released.
	}, {
		name:            "dualstack:v4v6_NoAllocs-Allocs",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeExternalName),
		update: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		verifyDryAllocs: true, // make sure values were not allocated.
	}, {
		name:            "dualstack:v4v6_Allocs-NoAllocs",
		clusterFamilies: []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol},
		svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		update:          svctest.MakeService("foo", svctest.SetTypeExternalName),
		verifyDryAllocs: false, // make sure values were not released.
	}, {
		name:            "dualstack:v6v4_NoAllocs-Allocs",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc:             svctest.MakeService("foo", svctest.SetTypeExternalName),
		update: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		verifyDryAllocs: true, // make sure values were not allocated.
	}, {
		name:            "dualstack:v6v4_Allocs-NoAllocs",
		clusterFamilies: []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol},
		svc: svctest.MakeService("foo", svctest.SetTypeLoadBalancer,
			svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
		update:          svctest.MakeService("foo", svctest.SetTypeExternalName),
		verifyDryAllocs: false, // make sure values were not released.
	}}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			storage, _, server := newStorage(t, tc.clusterFamilies)
			defer server.Terminate(t)
			defer storage.Store.DestroyFunc()

			ctx := genericapirequest.NewDefaultContext()
			obj, err := storage.Create(ctx, tc.svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Fatalf("unexpected error creating service: %v", err)
			}
			createdSvc := obj.(*api.Service)

			if tc.verifyDryAllocs {
				// Dry allocs means no allocs on create.  Ensure values were
				// NOT allocated.
				proveClusterIPsDeallocated(t, storage, nil, createdSvc)
			} else {
				// Ensure IPs were allocated
				proveClusterIPsAllocated(t, storage, nil, createdSvc)
			}

			// Update the object to the new state and check the results.
			obj, _, err = storage.Update(ctx, tc.update.Name,
				rest.DefaultUpdatedObjectInfo(tc.update), rest.ValidateAllObjectFunc,
				rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{DryRun: []string{metav1.DryRunAll}})
			if err != nil {
				t.Fatalf("unexpected error updating service: %v", err)
			}
			updatedSvc := obj.(*api.Service)

			if tc.verifyDryAllocs {
				// Dry allocs means the values are assigned but not
				// allocated.
				if netutils.ParseIPSloppy(updatedSvc.Spec.ClusterIP) == nil {
					t.Errorf("expected valid clusterIP: %q", updatedSvc.Spec.ClusterIP)
				}
				for _, ip := range updatedSvc.Spec.ClusterIPs {
					if netutils.ParseIPSloppy(ip) == nil {
						t.Errorf("expected valid clusterIP: %q", updatedSvc.Spec.ClusterIP)
					}
				}
				for i, fam := range updatedSvc.Spec.IPFamilies {
					if ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[fam], updatedSvc.Spec.ClusterIPs[i]) {
						t.Errorf("expected IP to not be allocated: %q", updatedSvc.Spec.ClusterIPs[i])
					}
				}

				for _, p := range updatedSvc.Spec.Ports {
					if p.NodePort == 0 {
						t.Errorf("expected nodePort to be assigned: %d", p.NodePort)
					}
					if portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
						t.Errorf("expected nodePort to not be allocated: %d", p.NodePort)
					}
				}

				if updatedSvc.Spec.HealthCheckNodePort == 0 {
					t.Errorf("expected HCNP to be assigned: %d", updatedSvc.Spec.HealthCheckNodePort)
				}
				if portIsAllocated(t, storage.alloc.serviceNodePorts, updatedSvc.Spec.HealthCheckNodePort) {
					t.Errorf("expected HCNP to not be allocated: %d", updatedSvc.Spec.HealthCheckNodePort)
				}
			} else {
				// Ensure IPs were unassigned but not deallocated.
				if updatedSvc.Spec.ClusterIP != "" {
					t.Errorf("expected clusterIP to be unset: %q", updatedSvc.Spec.ClusterIP)
				}
				if len(updatedSvc.Spec.ClusterIPs) != 0 {
					t.Errorf("expected clusterIPs to be unset: %q", updatedSvc.Spec.ClusterIPs)
				}
				for i, fam := range createdSvc.Spec.IPFamilies {
					if !ipIsAllocated(t, storage.alloc.serviceIPAllocatorsByFamily[fam], createdSvc.Spec.ClusterIPs[i]) {
						t.Errorf("expected IP to still be allocated: %q", createdSvc.Spec.ClusterIPs[i])
					}
				}

				for _, p := range updatedSvc.Spec.Ports {
					if p.NodePort != 0 {
						t.Errorf("expected nodePort to be unset: %d", p.NodePort)
					}
				}
				for _, p := range createdSvc.Spec.Ports {
					if !portIsAllocated(t, storage.alloc.serviceNodePorts, p.NodePort) {
						t.Errorf("expected nodePort to still be allocated: %d", p.NodePort)
					}
				}

				if updatedSvc.Spec.HealthCheckNodePort != 0 {
					t.Errorf("expected HCNP to be unset: %d", updatedSvc.Spec.HealthCheckNodePort)
				}
				if !portIsAllocated(t, storage.alloc.serviceNodePorts, createdSvc.Spec.HealthCheckNodePort) {
					t.Errorf("expected HCNP to still be allocated: %d", createdSvc.Spec.HealthCheckNodePort)
				}
			}
		})
	}
}

func TestUpdatePatchAllocatedValues(t *testing.T) {
	prove := func(proofs ...svcTestProof) []svcTestProof {
		return proofs
	}
	proveClusterIP := func(idx int, ip string) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			if want, got := ip, after.Spec.ClusterIPs[idx]; want != got {
				t.Errorf("wrong ClusterIPs[%d]: want %q, got %q", idx, want, got)
			}
		}
	}
	proveNodePort := func(idx int, port int32) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			got := after.Spec.Ports[idx].NodePort
			if port > 0 && got != port {
				t.Errorf("wrong Ports[%d].NodePort: want %d, got %d", idx, port, got)
			} else if port < 0 && got == -port {
				t.Errorf("wrong Ports[%d].NodePort: wanted anything but %d", idx, got)
			}
		}
	}
	proveHCNP := func(port int32) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			got := after.Spec.HealthCheckNodePort
			if port > 0 && got != port {
				t.Errorf("wrong HealthCheckNodePort: want %d, got %d", port, got)
			} else if port < 0 && got == -port {
				t.Errorf("wrong HealthCheckNodePort: wanted anything but %d", got)
			}
		}
	}

	// each create needs clusterIP, NodePort, and HealthCheckNodePort allocated
	// each update needs clusterIP, NodePort, and/or HealthCheckNodePort blank
	testCases := []cudTestCase{{
		name: "single-ip_single-port",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetClusterIPs("10.0.0.1"),
				svctest.SetNodePorts(30093),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
			prove: prove(
				proveClusterIP(0, "10.0.0.1"),
				proveNodePort(0, 30093),
				proveHCNP(30118)),
		},
	}, {
		name: "multi-ip_multi-port",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1"),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP))),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
			prove: prove(
				proveClusterIP(0, "10.0.0.1"),
				proveClusterIP(1, "2000::1"),
				proveNodePort(0, 30093),
				proveNodePort(1, 30076),
				proveHCNP(30118)),
		},
	}, {
		name: "multi-ip_partial",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1"),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "multi-port_partial",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 0)), // provide just 1 value
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
			prove: prove(
				proveNodePort(0, 30093),
				proveNodePort(1, 30076),
				proveHCNP(30118)),
		},
	}, {
		name: "swap-ports",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					// swapped from above
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP),
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP))),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
			prove: prove(
				proveNodePort(0, 30076),
				proveNodePort(1, 30093),
				proveHCNP(30118)),
		},
	}, {
		name: "partial-swap-ports",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30076, 0), // set [0] to [1]'s value, omit [1]
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
			prove: prove(
				proveNodePort(0, 30076),
				proveNodePort(1, -30076),
				proveHCNP(30118)),
		},
	}, {
		name: "swap-port-with-hcnp",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30076, 30118)), // set [0] to HCNP's value
			expectError: true,
		},
	}, {
		name: "partial-swap-port-with-hcnp",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30118, 0)), // set [0] to HCNP's value, omit [1]
			expectError: true,
		},
	}, {
		name: "update-hcnp",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30118)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetPorts(
					svctest.MakeServicePort("p", 867, intstr.FromInt32(867), api.ProtocolTCP),
					svctest.MakeServicePort("q", 5309, intstr.FromInt32(5309), api.ProtocolTCP)),
				svctest.SetNodePorts(30093, 30076),
				svctest.SetHealthCheckNodePort(30111)),
			expectError: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

// Proves that updates from single-stack work.
func TestUpdateIPsFromSingleStack(t *testing.T) {
	prove := func(proofs ...svcTestProof) []svcTestProof {
		return proofs
	}
	proveNumFamilies := func(n int) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			t.Helper()
			if got := len(after.Spec.IPFamilies); got != n {
				t.Errorf("wrong number of ipFamilies: expected %d, got %d", n, got)
			}
		}
	}

	// Single-stack cases as control.
	testCasesV4 := []cudTestCase{{
		name: "single-single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "single-dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "single-dual_policy",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectError: true,
		},
	}, {
		name: "single-dual_families",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		},
	}}

	t.Run("singlestack:v4", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesV4, []api.IPFamily{api.IPv4Protocol})
	})

	// Dual-stack v4,v6 cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	//    ips={nil, single, dual}
	testCasesV4V6 := []cudTestCase{{
		name: "policy:nil_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "single-dual_wrong_order_families",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "single-dual_wrong_order_ips",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "single-dual_ip_in_use",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		beforeUpdate: func(t *testing.T, storage *wrapperRESTForTests) {
			alloc := storage.alloc.serviceIPAllocatorsByFamily[api.IPv6Protocol]
			ip := "2000::1"
			if err := alloc.Allocate(netutils.ParseIPSloppy(ip)); err != nil {
				t.Fatalf("test is incorrect, unable to preallocate IP %q: %v", ip, err)
			}
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}}

	t.Run("dualstack:v4v6", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesV4V6, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	})

	// Dual-stack v6,v4 cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	//    ips={nil, single, dual}
	testCasesV6V4 := []cudTestCase{{
		name: "policy:nil_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "single-dual_wrong_order_families",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "single-dual_wrong_order_ips",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "single-dual_ip_in_use",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
		beforeUpdate: func(t *testing.T, storage *wrapperRESTForTests) {
			alloc := storage.alloc.serviceIPAllocatorsByFamily[api.IPv4Protocol]
			ip := "10.0.0.1"
			if err := alloc.Allocate(netutils.ParseIPSloppy(ip)); err != nil {
				t.Fatalf("test is incorrect, unable to preallocate IP %q: %v", ip, err)
			}
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}}

	t.Run("dualstack:v6v4", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesV6V4, []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol})
	})

	// Headless cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	testCasesHeadless := []cudTestCase{{
		name: "policy:nil_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies("IPv4")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:nil_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectError: true,
		},
	}, {
		name: "policy:single_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies("IPv4")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies("IPv4")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies("IPv4")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}}

	t.Run("headless", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesHeadless, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	})
}

// Proves that updates from dual-stack.
func TestUpdateIPsFromDualStack(t *testing.T) {
	prove := func(proofs ...svcTestProof) []svcTestProof {
		return proofs
	}
	proveNumFamilies := func(n int) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			t.Helper()
			if got := len(after.Spec.IPFamilies); got != n {
				t.Errorf("wrong number of ipFamilies: expected %d, got %d", n, got)
			}
		}
	}

	// Dual-stack v4,v6 cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	//    ips={nil, single, dual}
	testCasesV4V6 := []cudTestCase{{
		name: "policy:nil_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:single_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:require_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv4Protocol, api.IPv6Protocol),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "dual-single_wrong_order_families",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "dual-single_wrong_order_ips",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("10.0.0.1", "2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}}

	t.Run("dualstack:v4v6", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesV4V6, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	})

	// Dual-stack v6,v4 cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	//    ips={nil, single, dual}
	testCasesV6V4 := []cudTestCase{{
		name: "policy:nil_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:single_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs:     true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:nil_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol)),
			expectError: true,
		},
	}, {
		name: "policy:require_families:single_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:single_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:dual_ips:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol)),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:dual_ips:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:dual_ips:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies(api.IPv6Protocol, api.IPv4Protocol),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
	}, {
		name: "dual-single_wrong_order_families",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies(api.IPv4Protocol)),
			expectError: true,
		},
	}, {
		name: "dual-single_wrong_order_ips",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs("2000::1", "10.0.0.1")),
			expectClusterIPs: true,
			prove:            prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetClusterIPs("10.0.0.1")),
			expectError: true,
		},
	}}

	t.Run("dualstack:v6v4", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesV6V4, []api.IPFamily{api.IPv6Protocol, api.IPv4Protocol})
	})

	// Headless cases: Covers the full matrix of:
	//    policy={nil, single, prefer, require}
	//    families={nil, single, dual}
	testCasesHeadless := []cudTestCase{{
		name: "policy:nil_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"})),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:nil_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies("IPv4")),
			expectError: true,
		},
	}, {
		name: "policy:nil_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:single_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies("IPv4")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:single_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicySingleStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless:       true,
			expectStackDowngrade: true,
			prove:                prove(proveNumFamilies(1)),
		},
	}, {
		name: "policy:prefer_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:prefer_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies("IPv4")),
			expectError: true,
		},
	}, {
		name: "policy:prefer_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:nil",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}, {
		name: "policy:require_families:single",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies("IPv4")),
			expectError: true,
		},
	}, {
		name: "policy:require_families:dual",
		line: line(),
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetClusterIPs(api.ClusterIPNone)),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSelector(map[string]string{"k2": "v2"}),
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyRequireDualStack),
				svctest.SetIPFamilies("IPv4", "IPv6")),
			expectHeadless: true,
			prove:          prove(proveNumFamilies(2)),
		},
	}}

	t.Run("headless", func(t *testing.T) {
		helpTestCreateUpdateDeleteWithFamilies(t, testCasesHeadless, []api.IPFamily{api.IPv4Protocol, api.IPv6Protocol})
	})
}

func TestFeatureExternalName(t *testing.T) {
	testCases := []cudTestCase{{
		name: "valid-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName, svctest.SetExternalName("updated.example.com")),
		},
	}, {
		name: "valid-blank",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc:         svctest.MakeService("foo", svctest.SetTypeExternalName, svctest.SetExternalName("")),
			expectError: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureSelector(t *testing.T) {
	testCases := []cudTestCase{{
		name: "valid-valid",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					s.Spec.Selector = map[string]string{"updated": "value"}
				}),
			expectClusterIPs: true,
		},
	}, {
		name: "valid-nil",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					s.Spec.Selector = nil
				}),
			expectClusterIPs: true,
		},
	}, {
		name: "valid-empty",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					s.Spec.Selector = map[string]string{}
				}),
			expectClusterIPs: true,
		},
	}, {
		name: "nil-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					s.Spec.Selector = nil
				}),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
	}, {
		name: "empty-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					s.Spec.Selector = map[string]string{}
				}),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureClusterIPs(t *testing.T) {
	testCases := []cudTestCase{{
		name: "clusterIP:valid-headless",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetHeadless),
			expectError: true,
		},
	}, {
		name: "clusterIP:headless-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetHeadless),
			expectHeadless: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetClusterIP("10.0.0.93")),
			expectError: true,
		},
	}, {
		name: "clusterIP:valid-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetClusterIP("10.0.0.93")),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetClusterIP("10.0.0.76")),
			expectError: true,
		},
	}, {
		name: "clusterIPs:valid-valid",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.93", "2000::93")),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetIPFamilyPolicy(api.IPFamilyPolicyPreferDualStack),
				svctest.SetClusterIPs("10.0.0.76", "2000::76")),
			expectError: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeaturePorts(t *testing.T) {
	testCases := []cudTestCase{{
		name: "add_port",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
	}, {
		name: "add_port_ClusterIP-NodePort",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "add_port_NodePort-ClusterIP",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
	}, {
		name: "remove_port",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
	}, {
		name: "remove_port_ClusterIP-NodePort",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "remove_port_NodePort-ClusterIP",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
		},
	}, {
		name: "swap_ports",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP),
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "modify_ports",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 8080, intstr.FromInt32(8080), api.ProtocolTCP),
					svctest.MakeServicePort("q", 8443, intstr.FromInt32(8443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "modify_protos",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolUDP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolUDP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "modify_ports_and_protos",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("r", 53, intstr.FromInt32(53), api.ProtocolTCP),
					svctest.MakeServicePort("s", 53, intstr.FromInt32(53), api.ProtocolUDP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "add_alt_proto",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 53, intstr.FromInt32(53), api.ProtocolTCP),
					svctest.MakeServicePort("q", 53, intstr.FromInt32(53), api.ProtocolUDP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "wipe_all",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts(
					svctest.MakeServicePort("p", 80, intstr.FromInt32(80), api.ProtocolTCP),
					svctest.MakeServicePort("q", 443, intstr.FromInt32(443), api.ProtocolTCP))),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeNodePort,
				svctest.SetPorts()),
			expectError:     true,
			expectNodePorts: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureSessionAffinity(t *testing.T) {
	testCases := []cudTestCase{{
		name: "None-ClientIPNoConfig",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityNone)),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				func(s *api.Service) {
					// Set it without setting the config
					s.Spec.SessionAffinity = api.ServiceAffinityClientIP
				}),
			expectError: true,
		},
	}, {
		name: "None-ClientIP",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityNone)),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityClientIP)),
			expectClusterIPs: true,
		},
	}, {
		name: "ClientIP-NoneWithConfig",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityClientIP)),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityClientIP),
				func(s *api.Service) {
					// Set it without wiping the config
					s.Spec.SessionAffinity = api.ServiceAffinityNone
				}),
			expectError: true,
		},
	}, {
		name: "ClientIP-None",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityClientIP)),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeClusterIP,
				svctest.SetSessionAffinity(api.ServiceAffinityNone),
				func(s *api.Service) {
					s.Spec.SessionAffinityConfig = nil
				}),
			expectClusterIPs: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureType(t *testing.T) {
	testCases := []cudTestCase{{
		name: "ExternalName-ClusterIP",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
	}, {
		name: "ClusterIP-ExternalName",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
	}, {
		name: "ExternalName-NodePort",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "NodePort-ExternalName",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
	}, {
		name: "ExternalName-LoadBalancer",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "LoadBalancer-ExternalName",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
	}, {
		name: "ClusterIP-NodePort",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "NodePort-ClusterIP",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
	}, {
		name: "ClusterIP-LoadBalancer",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "LoadBalancer-ClusterIP",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeClusterIP),
			expectClusterIPs: true,
		},
	}, {
		name: "NodePort-LoadBalancer",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "LoadBalancer-NodePort",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
	}, {
		name: "Headless-ExternalName",
		create: svcTestCase{
			svc:            svctest.MakeService("foo", svctest.SetHeadless),
			expectHeadless: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
	}, {
		name: "ExternalName-Headless",
		create: svcTestCase{
			svc: svctest.MakeService("foo", svctest.SetTypeExternalName),
		},
		update: svcTestCase{
			svc:            svctest.MakeService("foo", svctest.SetHeadless),
			expectHeadless: true,
		},
	}, {
		name: "Headless-NodePort",
		create: svcTestCase{
			svc:            svctest.MakeService("foo", svctest.SetHeadless),
			expectHeadless: true,
		},
		update: svcTestCase{
			svc:         svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectError: true,
		},
	}, {
		name: "NodePort-Headless",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:         svctest.MakeService("foo", svctest.SetHeadless),
			expectError: true,
		},
	}, {
		name: "Headless-LoadBalancer",
		create: svcTestCase{
			svc:            svctest.MakeService("foo", svctest.SetHeadless),
			expectHeadless: true,
		},
		update: svcTestCase{
			svc:         svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectError: true,
		},
	}, {
		name: "LoadBalancer-Headless",
		create: svcTestCase{
			svc:              svctest.MakeService("foo", svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
		},
		update: svcTestCase{
			svc:         svctest.MakeService("foo", svctest.SetHeadless),
			expectError: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureExternalTrafficPolicy(t *testing.T) {
	testCases := []cudTestCase{{
		name: "ExternalName_policy:none_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName,
				svctest.SetExternalTrafficPolicy(""),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "ExternalName_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectError: true,
		},
	}, {
		name: "ExternalName_policy:Cluster_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "ExternalName_policy:Local_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectError: true,
		},
	}, {
		name: "ExternalName_policy:Local_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:none_hcnp:none_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy("")),
			expectClusterIPs: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:none_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(""),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:Cluster_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:Local_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectError: true,
		},
	}, {
		name: "ClusterIP_policy:Local_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "NodePort_policy:none_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy("")),
			expectError: true,
		},
	}, {
		name: "NodePort_policy:none_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(""),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "NodePort_policy:Cluster_hcnp:none_policy:Local_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
	}, {
		name: "NodePort_policy:Cluster_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "NodePort_policy:Local_hcnp:none_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
	}, {
		name: "NodePort_policy:Local_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:none_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy("")),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:none_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(""),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:Cluster_hcnp:none_policy:Local_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
	}, {
		name: "LoadBalancer_policy:Cluster_hcnp:none_policy:Local_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
	}, {
		name: "LoadBalancer_policy:Cluster_hcnp:specified",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster),
				svctest.SetHealthCheckNodePort(30000)),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:Local_hcnp:none_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
	}, {
		name: "LoadBalancer_policy:Local_hcnp:specified_policy:Cluster_hcnp:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyCluster)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: false,
		},
	}, {
		name: "LoadBalancer_policy:Local_hcnp:specified_policy:Cluster_hcnp:different",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30000)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(30001)),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:Local_hcnp:none_policy:Inalid",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal)),
			expectClusterIPs:          true,
			expectNodePorts:           true,
			expectHealthCheckNodePort: true,
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy("Invalid")),
			expectError: true,
		},
	}, {
		name: "LoadBalancer_policy:Local_hcnp:negative",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetExternalTrafficPolicy(api.ServiceExternalTrafficPolicyLocal),
				svctest.SetHealthCheckNodePort(-1)),
			expectError: true,
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

func TestFeatureInternalTrafficPolicy(t *testing.T) {
	prove := func(proofs ...svcTestProof) []svcTestProof {
		return proofs
	}
	proveITP := func(want api.ServiceInternalTrafficPolicy) svcTestProof {
		return func(t *testing.T, storage *wrapperRESTForTests, before, after *api.Service) {
			t.Helper()
			if got := after.Spec.InternalTrafficPolicy; got == nil {
				if want != "" {
					t.Errorf("internalTrafficPolicy was nil")
				}
			} else if *got != want {
				if want == "" {
					want = "nil"
				}
				t.Errorf("wrong internalTrafficPoilcy: expected %s, got %s", want, *got)
			}
		}
	}

	testCases := []cudTestCase{{
		name: "ExternalName_policy:none-ExternalName_policy:none",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName),
			prove: prove(proveITP("")),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeExternalName),
			prove: prove(proveITP("")),
		},
	}, {
		name: "ClusterIP_policy:none-ClusterIP_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP),
			expectClusterIPs: true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "ClusterIP_policy:Cluster-ClusterIP_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster)),
			expectClusterIPs: true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeClusterIP,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "NodePort_policy:none-NodePort_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "NodePort_policy:Cluster-NodePort_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeNodePort,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "LoadBalancer_policy:none-LoadBalancer_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "LoadBalancer_policy:Cluster-LoadBalancer_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetTypeLoadBalancer,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectClusterIPs: true,
			expectNodePorts:  true,
			prove:            prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "Headless_policy:none-Headless_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetHeadless),
			expectHeadless: true,
			prove:          prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectHeadless: true,
			prove:          prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}, {
		name: "Headless_policy:Cluster-Headless_policy:Local",
		create: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyCluster)),
			expectHeadless: true,
			prove:          prove(proveITP(api.ServiceInternalTrafficPolicyCluster)),
		},
		update: svcTestCase{
			svc: svctest.MakeService("foo",
				svctest.SetHeadless,
				svctest.SetInternalTrafficPolicy(api.ServiceInternalTrafficPolicyLocal)),
			expectHeadless: true,
			prove:          prove(proveITP(api.ServiceInternalTrafficPolicyLocal)),
		},
	}}

	helpTestCreateUpdateDelete(t, testCases)
}

// TODO(thockin): We need to look at feature-tests for:
//   externalIPs, lbip, lbsourceranges, externalname, PublishNotReadyAddresses, AllocateLoadBalancerNodePorts, LoadBalancerClass, status

// this is local because it's not fully fleshed out enough for general use.
func makePod(name string, ips ...string) api.Pod {
	p := api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: metav1.NamespaceDefault,
		},
		Spec: api.PodSpec{
			RestartPolicy: api.RestartPolicyAlways,
			DNSPolicy:     api.DNSDefault,
			Containers:    []api.Container{{Name: "ctr", Image: "img", ImagePullPolicy: api.PullIfNotPresent, TerminationMessagePolicy: api.TerminationMessageReadFile}},
		},
		Status: api.PodStatus{
			PodIPs: []api.PodIP{},
		},
	}

	for _, ip := range ips {
		p.Status.PodIPs = append(p.Status.PodIPs, api.PodIP{IP: ip})
	}

	return p
}

func TestServiceRegistryResourceLocation(t *testing.T) {
	pods := []api.Pod{
		makePod("unnamed", "1.2.3.4", "1.2.3.5"),
		makePod("named", "1.2.3.6", "1.2.3.7"),
		makePod("no-endpoints", "9.9.9.9"), // to prove this does not get chosen
	}

	endpoints := []*api.Endpoints{
		epstest.MakeEndpoints("unnamed",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.4", "unnamed"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("", 80),
			}),
		epstest.MakeEndpoints("unnamed2",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.5", "unnamed"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("", 80),
			}),
		epstest.MakeEndpoints("named",
			[]api.EndpointAddress{
				epstest.MakeEndpointAddress("1.2.3.6", "named"),
			},
			[]api.EndpointPort{
				epstest.MakeEndpointPort("p", 80),
				epstest.MakeEndpointPort("q", 81),
			}),
		epstest.MakeEndpoints("no-endpoints", nil, nil), // to prove this does not get chosen
	}

	storage, _, server := newStorageWithPods(t, []api.IPFamily{api.IPv4Protocol}, pods, endpoints)
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()

	ctx := genericapirequest.NewDefaultContext()
	for _, name := range []string{"unnamed", "unnamed2", "no-endpoints"} {
		_, err := storage.Create(ctx,
			svctest.MakeService(name,
				svctest.SetPorts(
					svctest.MakeServicePort("", 93, intstr.FromInt32(80), api.ProtocolTCP))),
			rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
		if err != nil {
			t.Fatalf("unexpected error creating service %q: %v", name, err)
		}

	}
	_, err := storage.Create(ctx,
		svctest.MakeService("named",
			svctest.SetPorts(
				svctest.MakeServicePort("p", 93, intstr.FromInt32(80), api.ProtocolTCP),
				svctest.MakeServicePort("q", 76, intstr.FromInt32(81), api.ProtocolTCP))),
		rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
	if err != nil {
		t.Fatalf("unexpected error creating service %q: %v", "named", err)
	}
	redirector := rest.Redirector(storage)

	cases := []struct {
		query  string
		err    bool
		expect string
	}{{
		query:  "unnamed",
		expect: "//1.2.3.4:80",
	}, {
		query:  "unnamed:",
		expect: "//1.2.3.4:80",
	}, {
		query:  "unnamed:93",
		expect: "//1.2.3.4:80",
	}, {
		query:  "http:unnamed:",
		expect: "http://1.2.3.4:80",
	}, {
		query:  "http:unnamed:93",
		expect: "http://1.2.3.4:80",
	}, {
		query: "unnamed:80",
		err:   true,
	}, {
		query:  "unnamed2",
		expect: "//1.2.3.5:80",
	}, {
		query:  "named:p",
		expect: "//1.2.3.6:80",
	}, {
		query:  "named:q",
		expect: "//1.2.3.6:81",
	}, {
		query:  "named:93",
		expect: "//1.2.3.6:80",
	}, {
		query:  "named:76",
		expect: "//1.2.3.6:81",
	}, {
		query:  "http:named:p",
		expect: "http://1.2.3.6:80",
	}, {
		query:  "http:named:q",
		expect: "http://1.2.3.6:81",
	}, {
		query: "named:bad",
		err:   true,
	}, {
		query: "no-endpoints",
		err:   true,
	}, {
		query: "non-existent",
		err:   true,
	}}
	for _, tc := range cases {
		t.Run(tc.query, func(t *testing.T) {
			location, _, err := redirector.ResourceLocation(ctx, tc.query)
			if tc.err == false && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if tc.err && err == nil {
				t.Fatalf("unexpected success")
			}
			if !tc.err {
				if location == nil {
					t.Errorf("unexpected location: %v", location)
				}
				if e, a := tc.expect, location.String(); e != a {
					t.Errorf("expected %q, but got %q", e, a)
				}
			}
		})
	}
}

func TestUpdateServiceLoadBalancerStatus(t *testing.T) {
	storage, statusStorage, server := newStorage(t, []api.IPFamily{api.IPv4Protocol})
	defer server.Terminate(t)
	defer storage.Store.DestroyFunc()
	defer statusStorage.store.DestroyFunc()

	ipModeVIP := api.LoadBalancerIPModeVIP
	ipModeProxy := api.LoadBalancerIPModeProxy
	ipModeDummy := api.LoadBalancerIPMode("dummy")

	testCases := []struct {
		name                   string
		ipModeEnabled          bool
		statusBeforeUpdate     api.ServiceStatus
		newStatus              api.ServiceStatus
		expectedStatus         api.ServiceStatus
		expectErr              bool
		expectedReasonForError metav1.StatusReason
	}{
		/*LoadBalancerIPMode disabled*/
		{
			name:               "LoadBalancerIPMode disabled, ipMode not used in old, not used in new",
			ipModeEnabled:      false,
			statusBeforeUpdate: api.ServiceStatus{},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectErr: false,
		}, {
			name:          "LoadBalancerIPMode disabled, ipMode used in old and in new",
			ipModeEnabled: false,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeVIP,
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			expectErr: false,
		}, {
			name:          "LoadBalancerIPMode disabled, ipMode not used in old, used in new",
			ipModeEnabled: false,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectErr: false,
		}, {
			name:          "LoadBalancerIPMode disabled, ipMode used in old, not used in new",
			ipModeEnabled: false,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeVIP,
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectErr: false,
		},
		/*LoadBalancerIPMode enabled*/
		{
			name:               "LoadBalancerIPMode enabled, ipMode not used in old, not used in new",
			ipModeEnabled:      true,
			statusBeforeUpdate: api.ServiceStatus{},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectedStatus:         api.ServiceStatus{},
			expectErr:              true,
			expectedReasonForError: metav1.StatusReasonInvalid,
		}, {
			name:          "LoadBalancerIPMode enabled, ipMode used in old and in new",
			ipModeEnabled: true,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeVIP,
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeVIP,
					}},
				},
			},
			expectErr: false,
		}, {
			name:          "LoadBalancerIPMode enabled, ipMode not used in old, used in new",
			ipModeEnabled: true,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			expectedStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeProxy,
					}},
				},
			},
			expectErr: false,
		}, {
			name:          "LoadBalancerIPMode enabled, ipMode used in old, not used in new",
			ipModeEnabled: true,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeVIP,
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			expectedStatus:         api.ServiceStatus{},
			expectErr:              true,
			expectedReasonForError: metav1.StatusReasonInvalid,
		}, {
			name:          "LoadBalancerIPMode enabled, ipMode not used in old, invalid value used in new",
			ipModeEnabled: true,
			statusBeforeUpdate: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP: "1.2.3.4",
					}},
				},
			},
			newStatus: api.ServiceStatus{
				LoadBalancer: api.LoadBalancerStatus{
					Ingress: []api.LoadBalancerIngress{{
						IP:     "1.2.3.4",
						IPMode: &ipModeDummy,
					}},
				},
			},
			expectedStatus:         api.ServiceStatus{},
			expectErr:              true,
			expectedReasonForError: metav1.StatusReasonInvalid,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {

			svc := svctest.MakeService("foo", svctest.SetTypeLoadBalancer)
			ctx := genericapirequest.NewDefaultContext()
			obj, err := storage.Create(ctx, svc, rest.ValidateAllObjectFunc, &metav1.CreateOptions{})
			if err != nil {
				t.Errorf("created svc: %s", err)
			}
			defer storage.Delete(ctx, svc.Name, rest.ValidateAllObjectFunc, &metav1.DeleteOptions{})

			// prepare status
			// Test here is negative, because starting with v1.30 the feature gate is enabled by default, so we should
			// now disable it to do the proper test
			if !loadbalancerIPModeInUse(tc.statusBeforeUpdate) {
				featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, false)
			}
			oldSvc := obj.(*api.Service).DeepCopy()
			oldSvc.Status = tc.statusBeforeUpdate
			obj, _, err = statusStorage.Update(ctx, oldSvc.Name, rest.DefaultUpdatedObjectInfo(oldSvc), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
			if err != nil {
				t.Errorf("updated status: %s", err)
			}

			featuregatetesting.SetFeatureGateDuringTest(t, utilfeature.DefaultFeatureGate, features.LoadBalancerIPMode, tc.ipModeEnabled)
			newSvc := obj.(*api.Service).DeepCopy()
			newSvc.Status = tc.newStatus
			obj, _, err = statusStorage.Update(ctx, newSvc.Name, rest.DefaultUpdatedObjectInfo(newSvc), rest.ValidateAllObjectFunc, rest.ValidateAllObjectUpdateFunc, false, &metav1.UpdateOptions{})
			if err != nil {
				if tc.expectErr && tc.expectedReasonForError == errors.ReasonForError(err) {
					return
				}
				t.Errorf("updated status: %s", err)
			}

			updated := obj.(*api.Service)
			if !reflect.DeepEqual(tc.expectedStatus, updated.Status) {
				t.Errorf("%v: unexpected svc status: %v", tc.name, cmp.Diff(tc.expectedStatus, updated.Status))
			}
		})
	}
}

func loadbalancerIPModeInUse(status api.ServiceStatus) bool {
	for _, ing := range status.LoadBalancer.Ingress {
		if ing.IPMode != nil {
			return true
		}
	}
	return false
}
