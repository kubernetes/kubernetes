package benchmark

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/api/meta"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/apis/meta/v1/unstructured"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	discofake "k8s.io/client-go/discovery/fake"
	dyfake "k8s.io/client-go/dynamic/fake"
	fakeclientset "k8s.io/client-go/kubernetes/fake"
	"k8s.io/kubernetes/test/utils/ktesting"
	"k8s.io/utils/ptr"
)

func TestIsValid(t *testing.T) {
	testCases := []struct {
		desc string
		c    *createAny
		want error
	}{
		{
			desc: "TemplatePath must be set",
			c:    &createAny{},
			want: fmt.Errorf("TemplatePath must be set"),
		},
		{
			desc: "valid createAny",
			c: &createAny{
				TemplatePath: "hoge",
			},
			want: nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			got := tc.c.isValid(false)
			if (got == nil) != (tc.want == nil) {
				t.Fatalf("Got %q, want %q", got, tc.want)
			}
		})
	}
}

func TestCollectsMetrics(t *testing.T) {
	testCases := []struct {
		desc string
		want bool
	}{
		{
			desc: "return false",
			want: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			c := &createAny{}
			got := c.collectsMetrics()
			if got != tc.want {
				t.Fatalf("Got %t, want %t", got, tc.want)
			}
		})
	}
}

func TestPatchParams(t *testing.T) {
	testCases := []struct {
		desc      string
		c         *createAny
		w         *workload
		wantCount *int
		wantErr   bool
	}{
		{
			desc: "empty CountParam",
			c: &createAny{
				TemplatePath: "template.path",
			},
			w: &workload{
				Params: params{
					params: map[string]any{"numPods": 10},
				},
			},
			wantCount: nil,
			wantErr:   false,
		},
		{
			desc: "valid CountParam",
			c: &createAny{
				CountParam:   "$numPods",
				TemplatePath: "template.path",
			},
			w: &workload{
				Params: params{
					params: map[string]any{"numPods": 10.0},
					isUsed: map[string]bool{},
				},
			},
			wantCount: ptr.To(10),
			wantErr:   false,
		},
		{
			desc: "CountParam not present",
			c: &createAny{
				CountParam:   "$notFound",
				TemplatePath: "template.path",
			},
			w: &workload{
				Params: params{
					params: map[string]any{"numPods": 10},
					isUsed: map[string]bool{},
				},
			},
			wantCount: nil,
			wantErr:   true,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			op, err := tc.c.patchParams(tc.w)

			if (err != nil) != tc.wantErr {
				t.Fatalf("patchParams() error = %v, wantErr = %v", err, tc.wantErr)
			}
			if err != nil {
				return
			}

			got := op.(*createAny)
			switch {
			case got.Count == nil && tc.wantCount != nil:
				t.Fatalf("got.Count = nil, want %d", *tc.wantCount)
			case got.Count != nil && tc.wantCount == nil:
				t.Fatalf("got.Count = %d, want nil", *got.Count)
			case got.Count != nil && tc.wantCount != nil:
				if *got.Count != *tc.wantCount {
					t.Errorf("got.Count = %d, want %d", *got.Count, *tc.wantCount)
				}
			}
		})
	}
}

func TestRequiredNamespaces(t *testing.T) {
	testCases := []struct {
		desc string
		c    *createAny
		want []string
	}{
		{
			desc: "createAny has NameSpace",
			c: &createAny{
				Namespace: "hoge",
			},
			want: []string{"hoge"},
		},
		{
			desc: "createAny doesn't have NameSpace",
			c:    &createAny{},
			want: nil,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			got := tc.c.requiredNamespaces()
			if !reflect.DeepEqual(got, tc.want) {
				t.Errorf("requiredNamespaces() = %v, want %v", got, tc.want)
			}
		})
	}
}

func TestCreate(t *testing.T) {
	// モック用の動的クライアントを初期化
	dynamicClient := dyfake.NewSimpleDynamicClient(runtime.NewScheme())

	// モック用の TContext を初期化
	tCtx := ktesting.Init(t)
	tCtx = ktesting.WithClients(tCtx, nil, nil, nil, dynamicClient, nil)

	// テストケース定義
	testCases := []struct {
		desc             string
		templatePath     string
		namespace        string
		env              map[string]any
		getSpecFunc      func(string, map[string]any, interface{}) error
		restMappingFunc  func(ktesting.TContext, *unstructured.Unstructured) (*meta.RESTMapping, error)
		expectedErrorMsg string
	}{
		{
			desc:         "getSpecFromTextTemplateFile fails",
			templatePath: "invalid-template.yaml",
			getSpecFunc: func(path string, env map[string]any, spec interface{}) error {
				return fmt.Errorf("failed to parse template: %s", path)
			},
			restMappingFunc: func(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
				return nil, nil
			},
			expectedErrorMsg: "parsing failed",
		},
		{
			desc:         "restMappingFromUnstructuredObj fails",
			templatePath: "hoge.yaml",
			getSpecFunc: func(path string, env map[string]any, spec interface{}) error {
				// spec をチェックして初期化
				u, ok := spec.(*unstructured.Unstructured)
				if !ok {
					return fmt.Errorf("spec is not of type *unstructured.Unstructured")
				}
				*u = unstructured.Unstructured{} // 明示的に初期化
				u.SetAPIVersion("v1")
				u.SetKind("Pod")
				u.SetName("test-pod")
				u.SetNamespace("default")
				return nil
			},
			restMappingFunc: func(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
				return nil, fmt.Errorf("failed to map object: %s", obj.GetKind())
			},
			expectedErrorMsg: "failed to map object",
		},
		{
			desc:         "namespace set, scope is root (error case)",
			templatePath: "valid-template.yaml",
			namespace:    "default",
			getSpecFunc: func(path string, env map[string]any, spec interface{}) error {
				obj := &unstructured.Unstructured{}
				obj.SetAPIVersion("v1")
				obj.SetKind("Pod")
				obj.SetName("test-pod")
				if u, ok := spec.(*unstructured.Unstructured); ok {
					*u = *obj
				}
				return nil
			},
			restMappingFunc: func(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
				return &meta.RESTMapping{
					GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
					Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
					Scope:            meta.RESTScopeRoot,
				}, nil
			},
			expectedErrorMsg: "namespace \"default\" set for \"valid-template.yaml\", but \"/v1, Kind=Pod\" has scope \"root\"",
		},
		{
			desc:         "namespace not set, scope is namespace (error case)",
			templatePath: "valid-template.yaml",
			namespace:    "",
			getSpecFunc: func(path string, env map[string]any, spec interface{}) error {
				obj := &unstructured.Unstructured{}
				obj.SetAPIVersion("v1")
				obj.SetKind("Pod")
				obj.SetName("test-pod")
				if u, ok := spec.(*unstructured.Unstructured); ok {
					*u = *obj
				}
				return nil
			},
			restMappingFunc: func(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
				return &meta.RESTMapping{
					GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
					Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
					Scope:            meta.RESTScopeNamespace,
				}, nil
			},
			expectedErrorMsg: "namespace not set for \"valid-template.yaml\", but \"/v1, Kind=Pod\" has scope \"namespace\"",
		},
		{
			desc:         "success",
			templatePath: "valid-template.yaml",
			namespace:    "default",
			getSpecFunc: func(path string, env map[string]any, spec interface{}) error {
				obj := &unstructured.Unstructured{}
				obj.SetAPIVersion("v1")
				obj.SetKind("Pod")
				obj.SetName("test-pod")
				if u, ok := spec.(*unstructured.Unstructured); ok {
					*u = *obj
				}
				return nil
			},
			restMappingFunc: func(tCtx ktesting.TContext, obj *unstructured.Unstructured) (*meta.RESTMapping, error) {
				return &meta.RESTMapping{
					GroupVersionKind: schema.GroupVersionKind{Group: "", Version: "v1", Kind: "Pod"},
					Resource:         schema.GroupVersionResource{Group: "", Version: "v1", Resource: "pods"},
					Scope:            meta.RESTScopeNamespace,
				}, nil
			},
			expectedErrorMsg: "",
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			op := &createAny{
				TemplatePath: tc.templatePath,
				Namespace:    tc.namespace,
			}

			err := op.create(tCtx, tc.env, tc.getSpecFunc, tc.restMappingFunc)

			if tc.expectedErrorMsg != "" {
				if err == nil || !strings.Contains(err.Error(), tc.expectedErrorMsg) {
					t.Errorf("expected error message %q, but got %v", tc.expectedErrorMsg, err)
				}
			} else if err != nil {
				t.Errorf("unexpected error: %v", err)
			}
		})
	}
}

func TestGetSpecFromTextTemplateFile(t *testing.T) {
	testCases := []struct {
		desc       string
		path       string
		wantErr    bool
		wantErrMsg string
	}{
		{
			desc:       "not exist file",
			path:       "notFound",
			wantErr:    true,
			wantErrMsg: "no such file or directory",
		},
		{
			desc:       "template parse error",
			path:       "./testfiles/parse-error.yaml",
			wantErr:    true,
			wantErrMsg: "function \"operator\" not defined",
		},
		{
			desc:       "template execute error",
			path:       "./testfiles/execute-error.yaml",
			wantErr:    true,
			wantErrMsg: "template: object:7:23: executing \"object\" at <div .Index 0>: error calling div: runtime error: integer divide by zero",
		},
		{
			desc:       "unmarshal error",
			path:       "./testfiles/unmarshal-error.yaml",
			wantErr:    true,
			wantErrMsg: "error unmarshaling JSON: while decoding JSON: json: unknown field \"hoge\"",
		},
		{
			desc:    "success to get templatefile",
			path:    "./testfiles/success-template.yaml",
			wantErr: false,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.desc, func(t *testing.T) {
			env := map[string]any{
				"Index": 1,
			}
			spec := &corev1.Pod{}

			err := getSpecFromTextTemplateFile(tc.path, env, spec)

			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error but got none (path=%s)", tc.path)
				}
				if tc.wantErrMsg != "" && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Errorf("expected error message to contain %q, but got %q", tc.wantErrMsg, err.Error())
				}
			} else {
				if err != nil {
					t.Fatalf("unexpected error: %v (path=%s)", err, tc.path)
				}
			}
		})
	}
}

func TestRestMappingFromUnstructuredObj(t *testing.T) {
	successResources := []*metav1.APIResourceList{
		{
			GroupVersion: "v1",
			APIResources: []metav1.APIResource{
				{
					Kind:       "Pod",
					Name:       "pods",
					Namespaced: true,
				},
			},
		},
	}

	clientset := fakeclientset.NewSimpleClientset()

	fakeDisco, ok := clientset.Discovery().(*discofake.FakeDiscovery)
	if !ok {
		t.Fatalf("unable to cast Discovery() to *discofake.FakeDiscovery")
	}
	fakeDisco.Fake.Resources = successResources

	tCtx := ktesting.Init(t)
	tCtx = ktesting.WithClients(tCtx, nil, nil, clientset, nil, nil)

	// テストケース定義
	testCases := []struct {
		name        string
		obj         *unstructured.Unstructured
		wantErr     bool
		wantErrMsg  string
		wantMapping *schema.GroupVersionResource
	}{
		{
			name: "Invalid API version",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "hoge/hoge/hoge",
					"kind":       "Pod",
				},
			},
			wantErr:    true,
			wantErrMsg: "extract group+version",
		},
		{
			name: "Unknown GVK",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "apps/v1",
					"kind":       "Deployment",
				},
			},
			wantErr:    true,
			wantErrMsg: "failed mapping",
		},
		{
			name: "Success mapping for Pod",
			obj: &unstructured.Unstructured{
				Object: map[string]interface{}{
					"apiVersion": "v1",
					"kind":       "Pod",
				},
			},
			wantErr: false,
			wantMapping: &schema.GroupVersionResource{
				Group:    "",
				Version:  "v1",
				Resource: "pods",
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			mapping, err := restMappingFromUnstructuredObj(tCtx, tc.obj)
			if tc.wantErr {
				if err == nil {
					t.Fatalf("expected error but got none (name=%s)", tc.name)
				}
				if tc.wantErrMsg != "" && !strings.Contains(err.Error(), tc.wantErrMsg) {
					t.Errorf("expected error message to contain %q, but got %q", tc.wantErrMsg, err.Error())
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v (name=%s)", err, tc.name)
			}

			if tc.wantMapping != nil {
				if !reflect.DeepEqual(mapping.Resource, *tc.wantMapping) {
					t.Errorf("got mapping.Resource = %v, want %v", mapping.Resource, *tc.wantMapping)
				}
			}
		})
	}
}
