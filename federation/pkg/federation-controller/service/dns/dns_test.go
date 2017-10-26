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

package dns

import (
	"fmt"
	"io"
	"reflect"
	"sort"
	"strconv"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/apimachinery/pkg/util/sets"
	"k8s.io/apimachinery/pkg/util/wait"
	corelisters "k8s.io/client-go/listers/core/v1"
	"k8s.io/client-go/tools/cache"
	"k8s.io/kubernetes/federation/apis/federation/v1beta1"
	fakefedclientset "k8s.io/kubernetes/federation/client/clientset_generated/federation_clientset/fake"
	"k8s.io/kubernetes/federation/pkg/dnsprovider"
	"k8s.io/kubernetes/federation/pkg/dnsprovider/providers/google/clouddns" // Only for unit testing purposes.
	"k8s.io/kubernetes/federation/pkg/federation-controller/service/ingress"
	. "k8s.io/kubernetes/federation/pkg/federation-controller/util/test"

	"github.com/golang/glog"
	"github.com/stretchr/testify/require"
)

const (
	services string = "services"

	dnsZone      = "example.com"
	fedName      = "ufp"
	svcName      = "nginx"
	svcNamespace = "test"
	recordA      = "A"
	recordCNAME  = "CNAME"
	DNSTTL       = "180"

	retryInterval = 100 * time.Millisecond

	OP_ADD    = "ADD"
	OP_UPDATE = "UPDATE"
	OP_DELETE = "DELETE"
)

var instanceCounter uint64 = 0

// NewClusterWithRegionZone builds a new cluster object with given region and zone attributes.
func NewClusterWithRegionZone(name string, readyStatus v1.ConditionStatus, region, zone string) *v1beta1.Cluster {
	cluster := NewCluster(name, readyStatus)
	cluster.Status.Zones = []string{zone}
	cluster.Status.Region = region
	return cluster
}

func NewService(name, namespace string, serviceType v1.ServiceType, port int32) *v1.Service {
	return &v1.Service{
		ObjectMeta: metav1.ObjectMeta{
			Name:      name,
			Namespace: namespace,
			SelfLink:  "/api/v1/namespaces/" + namespace + "/services/" + name,
			Labels:    map[string]string{"app": name},
		},
		Spec: v1.ServiceSpec{
			Ports: []v1.ServicePort{{Port: port}},
			Type:  serviceType,
		},
	}
}

func init() {
	dnsprovider.RegisterDnsProvider("fake-clouddns", func(config io.Reader) (dnsprovider.Interface, error) {
		return clouddns.NewFakeInterface([]string{dnsZone})
	})
}

func TestServiceDNSController(t *testing.T) {
	cluster1Name := "c1"
	cluster2Name := "c2"
	cluster1 := NewClusterWithRegionZone(cluster1Name, v1.ConditionTrue, "fooregion", "foozone")
	cluster2 := NewClusterWithRegionZone(cluster2Name, v1.ConditionTrue, "barregion", "barzone")
	globalDNSName := strings.Join([]string{svcName, svcNamespace, fedName, "svc", dnsZone}, ".")
	fooRegionDNSName := strings.Join([]string{svcName, svcNamespace, fedName, "svc", "fooregion", dnsZone}, ".")
	fooZoneDNSName := strings.Join([]string{svcName, svcNamespace, fedName, "svc", "foozone", "fooregion", dnsZone}, ".")
	barRegionDNSName := strings.Join([]string{svcName, svcNamespace, fedName, "svc", "barregion", dnsZone}, ".")
	barZoneDNSName := strings.Join([]string{svcName, svcNamespace, fedName, "svc", "barzone", "barregion", dnsZone}, ".")

	type step struct {
		operation string
		ingress   string
		expected  sets.String
	}
	tests := map[string]struct {
		steps []step
	}{
		"ServiceWithNoIngress": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}}},

		"ServiceWithSingleLBIngress": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		},
			{
				operation: OP_UPDATE,
				ingress: ingress.NewFederatedServiceIngress().
					AddEndpoints(cluster1Name, []string{"198.51.100.1"}).
					AddEndpoints(cluster2Name, []string{}).
					String(),
				expected: sets.NewString(
					strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
					strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
					strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
					strings.Join([]string{dnsZone, barRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
					strings.Join([]string{dnsZone, barZoneDNSName, recordCNAME, DNSTTL, "[" + barRegionDNSName + "]"}, ":"),
				),
			}}},

		/* This test case is dependent on system. there is no way to mock net.LookupHost
		// This test case however can be run locally as we are using 'localhost'
		"ServiceWithHostnameAsIngress": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"localhost"}).
				AddEndpoints(cluster2Name, []string{}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[127.0.0.1]"}, ":"),
				strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[127.0.0.1]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[127.0.0.1]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordCNAME, DNSTTL, "[" + barRegionDNSName + "]"}, ":"),
			),
		}}},
		*/

		"ServiceWithNoLBIngress": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{}).
				AddEndpoints(cluster2Name, []string{}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, fooRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordCNAME, DNSTTL, "[" + fooRegionDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordCNAME, DNSTTL, "[" + barRegionDNSName + "]"}, ":"),
			),
		}}},

		"ServiceWithMultipleLBIngress": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"198.51.100.1"}).
				AddEndpoints(cluster2Name, []string{"198.51.200.1"}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[198.51.100.1 198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
			),
		}}},

		"ServiceWithLBIngressAndServiceDeleted": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"198.51.100.1"}).
				AddEndpoints(cluster2Name, []string{"198.51.200.1"}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[198.51.100.1 198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
			),
		}, {
			operation: OP_DELETE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"198.51.100.1"}).
				AddEndpoints(cluster2Name, []string{"198.51.200.1"}).
				String(),
			expected: sets.NewString(
				// TODO: Ideally we should expect that there are no DNS records when federated service is deleted. Need to remove these leaks in future
				strings.Join([]string{dnsZone, fooRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordCNAME, DNSTTL, "[" + fooRegionDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordCNAME, DNSTTL, "[" + globalDNSName + "]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordCNAME, DNSTTL, "[" + barRegionDNSName + "]"}, ":"),
			),
		}}},

		"ServiceWithLBIngressAndLBIngressModifiedOvertime": {steps: []step{{
			operation: OP_ADD,
			expected:  sets.NewString(),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"198.51.100.1"}).
				AddEndpoints(cluster2Name, []string{"198.51.200.1"}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[198.51.100.1 198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[198.51.100.1]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
			),
		}, {
			operation: OP_UPDATE,
			ingress: ingress.NewFederatedServiceIngress().
				AddEndpoints(cluster1Name, []string{"198.51.150.1"}).
				AddEndpoints(cluster2Name, []string{"198.51.200.1"}).
				String(),
			expected: sets.NewString(
				strings.Join([]string{dnsZone, globalDNSName, recordA, DNSTTL, "[198.51.150.1 198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, fooRegionDNSName, recordA, DNSTTL, "[198.51.150.1]"}, ":"),
				strings.Join([]string{dnsZone, fooZoneDNSName, recordA, DNSTTL, "[198.51.150.1]"}, ":"),
				strings.Join([]string{dnsZone, barRegionDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
				strings.Join([]string{dnsZone, barZoneDNSName, recordA, DNSTTL, "[198.51.200.1]"}, ":"),
			),
		}}},
	}
	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			fakeClient := &fakefedclientset.Clientset{}
			RegisterFakeClusterGet(&fakeClient.Fake, &v1beta1.ClusterList{Items: []v1beta1.Cluster{*cluster1, *cluster2}})
			RegisterFakeList(services, &fakeClient.Fake, &v1.ServiceList{Items: []v1.Service{}})
			fedServiceWatch := RegisterFakeWatch(services, &fakeClient.Fake)
			RegisterFakeOnCreate(services, &fakeClient.Fake, fedServiceWatch)
			RegisterFakeOnUpdate(services, &fakeClient.Fake, fedServiceWatch)

			dc, err := NewServiceDNSController(fakeClient, "fake-clouddns", "", fedName, "", dnsZone, "")
			if err != nil {
				t.Errorf("error initializing dns controller: %v", err)
			}
			stop := make(chan struct{})
			glog.Infof("Running Service DNS Controller")
			go dc.DNSControllerRun(5, stop)

			service := NewService(svcName, svcNamespace, v1.ServiceTypeLoadBalancer, 80)
			key := types.NamespacedName{Namespace: service.Namespace, Name: service.Name}.String()
			for _, step := range test.steps {
				switch step.operation {
				case OP_ADD:
					fedServiceWatch.Add(service)
					require.NoError(t, WaitForFederatedServiceUpdate(t, dc.serviceStore,
						key, service, serviceCompare, wait.ForeverTestTimeout))
				case OP_UPDATE:
					service.Annotations = map[string]string{
						ingress.FederatedServiceIngressAnnotation: step.ingress}
					fedServiceWatch.Modify(service)
					require.NoError(t, WaitForFederatedServiceUpdate(t, dc.serviceStore,
						key, service, serviceCompare, wait.ForeverTestTimeout))
				case OP_DELETE:
					service.ObjectMeta.Finalizers = append(service.ObjectMeta.Finalizers, metav1.FinalizerOrphanDependents)
					service.DeletionTimestamp = &metav1.Time{Time: time.Now()}
					fedServiceWatch.Delete(service)
					require.NoError(t, WaitForFederatedServiceDelete(t, dc.serviceStore,
						key, wait.ForeverTestTimeout))
				}

				waitForDNSRecords(t, dc, step.expected)
			}
			close(stop)
		})
	}
}

// waitForDNSRecords waits for DNS records in fakedns to match expected DNS records
func waitForDNSRecords(t *testing.T, d *ServiceDNSController, expectedDNSRecords sets.String) {
	fakednsZones, ok := d.dns.Zones()
	if !ok {
		t.Error("Unable to fetch zones")
	}
	zones, err := fakednsZones.List()
	if err != nil {
		t.Errorf("error querying zones: %v", err)
	}

	// Dump every record to a testable-by-string-comparison form
	availableDNSRecords := sets.NewString()
	err = wait.PollImmediate(retryInterval, 5*time.Second, func() (bool, error) {
		for _, z := range zones {
			zoneName := z.Name()

			rrs, ok := z.ResourceRecordSets()
			if !ok {
				t.Errorf("cannot get rrs for zone %q", zoneName)
			}

			rrList, err := rrs.List()
			if err != nil {
				t.Errorf("error querying rr for zone %q: %v", zoneName, err)
			}
			availableDNSRecords = sets.NewString()
			for _, rr := range rrList {
				rrdatas := rr.Rrdatas()

				// Put in consistent (testable-by-string-comparison) order
				sort.Strings(rrdatas)
				availableDNSRecords.Insert(fmt.Sprintf("%s:%s:%s:%d:%s", zoneName, rr.Name(), rr.Type(), rr.Ttl(), rrdatas))
			}
		}

		if !availableDNSRecords.Equal(expectedDNSRecords) {
			return false, nil
		}
		return true, nil
	})
	if err != nil {
		t.Errorf("Actual DNS records does not match expected. Actual=%v, Expected=%v", availableDNSRecords, expectedDNSRecords)
	}
}

func TestNewServiceDNSController(t *testing.T) {
	tests := map[string]struct {
		registeredProvider string
		useProvider        string
		registeredZones    []string
		useZone            string
		federationName     string
		serviceDNSSuffix   string
		zoneId             string
		expectError        bool
	}{
		"AllValidParams": {
			federationName: "ufp",
			expectError:    false,
		},
		"EmptyFederationName": {
			federationName: "",
			expectError:    true,
		},
		"NoneExistingDNSProvider": {
			federationName: "ufp",
			useProvider:    "non-existent",
			expectError:    true,
		},
		"MultipleRegisteredZonesWithDifferentNames": {
			federationName:  "ufp",
			registeredZones: []string{"abc.com", "xyz.com"},
			expectError:     false,
		},
		"MultipleRegisteredZonesWithSameNames": {
			federationName:  "ufp",
			registeredZones: []string{"abc.com", "abc.com"},
			expectError:     true,
		},

		"MultipleRegisteredZonesWithSameNamesUseZoneId": {
			federationName:  "ufp",
			registeredZones: []string{"abc.com", "abc.com"},
			useZone:         "abc.com",
			zoneId:          "1",
			expectError:     true, // TODO: "google-clouddns" does not support multiple managed zones with same names
		},

		"UseNonExistentZone": {
			federationName:  "ufp",
			registeredZones: []string{"abc.com", "xyz.com"},
			useZone:         "example.com",
			expectError:     false,
		},
		"WithServiceDNSSuffix": {
			federationName:   "ufp",
			serviceDNSSuffix: "federation.example.com",
			expectError:      false,
		},
	}
	for testName, test := range tests {
		t.Run(testName, func(t *testing.T) {
			if test.registeredProvider == "" {
				test.registeredProvider = "fake-dns-" + testName + strconv.FormatUint(instanceCounter, 10)
				atomic.AddUint64(&instanceCounter, 1)
			}
			if test.useProvider == "" {
				test.useProvider = test.registeredProvider
			}
			if test.registeredZones == nil {
				test.registeredZones = append(test.registeredZones, "example.com")
			}
			if test.useZone == "" {
				test.useZone = test.registeredZones[0]
			}

			dnsprovider.RegisterDnsProvider(test.registeredProvider, func(config io.Reader) (dnsprovider.Interface, error) {
				return clouddns.NewFakeInterface(test.registeredZones)
			})

			_, err := NewServiceDNSController(&fakefedclientset.Clientset{},
				test.useProvider,
				"",
				test.federationName,
				test.serviceDNSSuffix,
				test.useZone,
				test.zoneId)
			if err != nil {
				if !test.expectError {
					t.Errorf("expected to succeed but got error: %v", err)
				}
			} else {
				if test.expectError {
					t.Errorf("expected to return error but succeeded")
				}
			}
		})
	}
}

type compare func(current, desired *v1.Service) (match bool)

func serviceCompare(s1, s2 *v1.Service) bool {
	return s1.Name == s2.Name && s1.Namespace == s2.Namespace &&
		(reflect.DeepEqual(s1.Annotations, s2.Annotations) || (len(s1.Annotations) == 0 && len(s2.Annotations) == 0)) &&
		(reflect.DeepEqual(s1.Labels, s2.Labels) || (len(s1.Labels) == 0 && len(s2.Labels) == 0)) &&
		reflect.DeepEqual(s1.Spec, s2.Spec) &&
		s1.DeletionTimestamp == s2.DeletionTimestamp
}

// WaitForFederatedServiceUpdate waits for federated service updates to match the desiredService.
func WaitForFederatedServiceUpdate(t *testing.T, store corelisters.ServiceLister, key string, desiredService *v1.Service, match compare, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			return false, err
		}
		service, err := store.Services(namespace).Get(name)
		switch {
		case errors.IsNotFound(err):
			return false, nil
		case err != nil:
			return false, err
		case !match(service, desiredService):
			return false, nil
		default:
			return true, nil
		}
	})
	return err
}

// WaitForFederatedServiceDelete waits for federated service to be deleted.
func WaitForFederatedServiceDelete(t *testing.T, store corelisters.ServiceLister, key string, timeout time.Duration) error {
	err := wait.PollImmediate(retryInterval, timeout, func() (bool, error) {
		namespace, name, err := cache.SplitMetaNamespaceKey(key)
		if err != nil {
			return false, err
		}
		_, err = store.Services(namespace).Get(name)
		if errors.IsNotFound(err) {
			return true, nil
		}
		return false, nil
	})
	return err
}
