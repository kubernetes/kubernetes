package benchmark

import (
	"fmt"
	"reflect"
	"strings"
	"testing"

	corev1 "k8s.io/api/core/v1"
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

// func TestRun(t *testing.T) {
// 	testCases := []strunct {
// 		desc: string,
// 		want:
// 	}
// }

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
