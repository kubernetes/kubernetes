package util

import (
	"testing"

	"github.com/google/go-cmp/cmp"
	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	mcsv1alpha1 "k8s.io/mcs-api/pkg/apis/multicluster/v1alpha1"
)

func TestServiceFromImport(t *testing.T) {
	httpProtocol := "http"
	testSessionTimeout := int32(60)
	tests := []struct {
		name     string
		input    mcsv1alpha1.ServiceImport
		expected v1.Service
	}{
		{
			name: "SuperclusterIP no affinity",
			input: mcsv1alpha1.ServiceImport{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testnamespace",
				},
				Spec: mcsv1alpha1.ServiceImportSpec{
					Type: mcsv1alpha1.SuperclusterIP,
					IP:   "10.42.42.42",
					Ports: []mcsv1alpha1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
					},
				},
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceImportPrefix + "test",
					Namespace: "testnamespace",
				},
				Spec: v1.ServiceSpec{
					Type:      v1.ServiceTypeClusterIP,
					ClusterIP: "10.42.42.42",
					Ports: []v1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
					},
				},
			},
		},
		{
			name: "Headless no affinity",
			input: mcsv1alpha1.ServiceImport{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testnamespace",
				},
				Spec: mcsv1alpha1.ServiceImportSpec{
					Type: mcsv1alpha1.Headless,
					Ports: []mcsv1alpha1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
					},
				},
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceImportPrefix + "test",
					Namespace: "testnamespace",
				},
				Spec: v1.ServiceSpec{
					Type:      v1.ServiceTypeClusterIP,
					ClusterIP: "none",
					Ports: []v1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
					},
				},
			},
		},
		{
			name: "SuperclusterIP multiple ports and affinity",
			input: mcsv1alpha1.ServiceImport{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testnamespace",
				},
				Spec: mcsv1alpha1.ServiceImportSpec{
					Type: mcsv1alpha1.SuperclusterIP,
					IP:   "10.42.42.42",
					Ports: []mcsv1alpha1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
						{Name: "health", Port: 8080, Protocol: v1.ProtocolTCP},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
					SessionAffinityConfig: &v1.SessionAffinityConfig{
						ClientIP: &v1.ClientIPConfig{
							TimeoutSeconds: &testSessionTimeout,
						},
					},
				},
			},
			expected: v1.Service{
				ObjectMeta: metav1.ObjectMeta{
					Name:      ServiceImportPrefix + "test",
					Namespace: "testnamespace",
				},
				Spec: v1.ServiceSpec{
					Type:      v1.ServiceTypeClusterIP,
					ClusterIP: "10.42.42.42",
					Ports: []v1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
						{Name: "health", Port: 8080, Protocol: v1.ProtocolTCP},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
					SessionAffinityConfig: &v1.SessionAffinityConfig{
						ClientIP: &v1.ClientIPConfig{
							TimeoutSeconds: &testSessionTimeout,
						},
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			got := ServiceFromImport(&tc.input)
			if diff := cmp.Diff(&tc.expected, got); diff != "" {
				t.Errorf("ServiceFromImport() returned unexpected result, got (+/-):\n%s", diff)
			}
		})
	}
}

func toUnstructured(t *testing.T, obj interface{}) *unstructured.Unstructured {
	u, err := runtime.DefaultUnstructuredConverter.ToUnstructured(obj)
	if err != nil {
		t.Fatalf("Failed to create unstructured: %v", err)
	}
	return &unstructured.Unstructured{
		Object: u,
	}
}

func TestServiceImportFromInformer(t *testing.T) {
	httpProtocol := "http"
	testSessionTimeout := int32(60)
	tests := []struct {
		name           string
		input          interface{}
		expectedImport *mcsv1alpha1.ServiceImport
		expectError    bool
	}{
		{
			name:        "not unstructured",
			input:       &mcsv1alpha1.ServiceImport{},
			expectError: true,
		},
		{
			name: "valid import",
			input: toUnstructured(t, &mcsv1alpha1.ServiceImport{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testnamespace",
				},
				Spec: mcsv1alpha1.ServiceImportSpec{
					Type: mcsv1alpha1.SuperclusterIP,
					IP:   "10.42.42.42",
					Ports: []mcsv1alpha1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
						{Name: "health", Port: 8080, Protocol: v1.ProtocolTCP},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
					SessionAffinityConfig: &v1.SessionAffinityConfig{
						ClientIP: &v1.ClientIPConfig{
							TimeoutSeconds: &testSessionTimeout,
						},
					},
				},
			}),
			expectedImport: &mcsv1alpha1.ServiceImport{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "test",
					Namespace: "testnamespace",
				},
				Spec: mcsv1alpha1.ServiceImportSpec{
					Type: mcsv1alpha1.SuperclusterIP,
					IP:   "10.42.42.42",
					Ports: []mcsv1alpha1.ServicePort{
						{Name: "http", Port: 80, Protocol: v1.ProtocolTCP, AppProtocol: &httpProtocol},
						{Name: "health", Port: 8080, Protocol: v1.ProtocolTCP},
					},
					SessionAffinity: v1.ServiceAffinityClientIP,
					SessionAffinityConfig: &v1.SessionAffinityConfig{
						ClientIP: &v1.ClientIPConfig{
							TimeoutSeconds: &testSessionTimeout,
						},
					},
				},
			},
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			si, err := ServiceImportFromInformer(tc.input)
			if tc.expectError {
				if err == nil {
					t.Errorf("ServiceImportFromInformer() returned unexpected result. Expected error, got %#v", si)
				}
				return
			}
			if err != nil {
				t.Fatalf("ServiceImportFromInformer() returned unexpected error, got %v", err)
			}
			if diff := cmp.Diff(tc.expectedImport, si); diff != "" {
				t.Errorf("ServiceImportFromInformer() returned unexpected result, got (+/1):\n%s", diff)
			}
		})
	}
}
