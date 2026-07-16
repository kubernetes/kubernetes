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

package rest

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"net/url"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	v1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apimachinery/pkg/types"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/kubernetes/pkg/kubelet/client"

	_ "k8s.io/kubernetes/pkg/apis/core/install"
)

type fakeConnectionInfoGetter struct {
	info *client.ConnectionInfo
	err  error
}

func (f *fakeConnectionInfoGetter) GetConnectionInfo(ctx context.Context, nodeName types.NodeName) (*client.ConnectionInfo, error) {
	return f.info, f.err
}

func TestAllocatedRESTGet(t *testing.T) {
	testPod := &api.Pod{
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
			UID:       "uid-foo",
		},
		Spec: api.PodSpec{
			NodeName: "test-node",
		},
	}

	versionedPod := &v1.Pod{
		TypeMeta: metav1.TypeMeta{
			Kind:       "Pod",
			APIVersion: "v1",
		},
		ObjectMeta: metav1.ObjectMeta{
			Name:      "foo",
			Namespace: "default",
			UID:       "uid-foo",
		},
		Spec: v1.PodSpec{
			Containers: []v1.Container{
				{Name: "c1", Image: "img1"},
			},
		},
		Status: v1.PodStatus{},
	}

	tests := []struct {
		name          string
		pod           *api.Pod
		podGetErr     error
		kubeletStatus int
		kubeletResp   interface{}
		connInfoErr   error
		expectedErr   error
		expectedPod   *api.Pod
	}{
		{
			name:          "happy path",
			pod:           testPod,
			kubeletStatus: http.StatusOK,
			kubeletResp:   versionedPod,
			expectedPod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{
					Name:      "foo",
					Namespace: "default",
					UID:       "uid-foo",
				},
				Spec: api.PodSpec{
					Containers: []api.Container{
						{Name: "c1", Image: "img1"},
					},
				},
			},
		},
		{
			name:        "pod not found in registry",
			podGetErr:   apierrors.NewNotFound(api.Resource("pods"), "foo"),
			expectedErr: apierrors.NewNotFound(api.Resource("pods"), "foo"),
		},
		{
			name: "pod has no host assigned",
			pod: &api.Pod{
				ObjectMeta: metav1.ObjectMeta{Name: "foo", Namespace: "default"},
				Spec:       api.PodSpec{NodeName: ""},
			},
			expectedErr: apierrors.NewBadRequest("pod foo does not have a host assigned"),
		},
		{
			name:        "connection info error",
			pod:         testPod,
			connInfoErr: fmt.Errorf("node not found"),
			expectedErr: fmt.Errorf("node not found"),
		},
		{
			name:          "kubelet returns 404",
			pod:           testPod,
			kubeletStatus: http.StatusNotFound,
			expectedErr:   apierrors.NewNotFound(schema.GroupResource{Resource: "pods/allocated"}, "foo"),
		},
		{
			name:          "kubelet returns 500",
			pod:           testPod,
			kubeletStatus: http.StatusInternalServerError,
			kubeletResp:   "internal error",
			expectedErr:   apierrors.NewGenericServerResponse(http.StatusInternalServerError, "get", schema.GroupResource{Resource: "pods/allocated"}, "foo", "Kubelet returned status 500: internal error", 0, true),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Start mock Kubelet server
			var ts *httptest.Server
			if tt.kubeletStatus != 0 {
				ts = httptest.NewTLSServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
					assert.Equal(t, fmt.Sprintf("/allocatedPods/%s", testPod.UID), r.URL.Path)
					w.WriteHeader(tt.kubeletStatus)
					if tt.kubeletResp != nil {
						switch v := tt.kubeletResp.(type) {
						case string:
							if _, err := w.Write([]byte(v)); err != nil {
								panic(fmt.Errorf("failed to write response: %w", err)) // Can't call t.Error
							}
						default:
							if err := json.NewEncoder(w).Encode(v); err != nil {
								panic(fmt.Errorf("failed to encode response: %w", err)) // Can't call t.Error
							}
						}
					}
				}))
				defer ts.Close()
			}

			// Configure fake connection info getter
			var connInfo *client.ConnectionInfo
			if ts != nil {
				u, err := url.Parse(ts.URL)
				require.NoError(t, err)
				host, port, err := net.SplitHostPort(u.Host)
				require.NoError(t, err)

				connInfo = &client.ConnectionInfo{
					Scheme:    "https",
					Hostname:  host,
					Port:      port,
					Transport: ts.Client().Transport,
				}
			}
			kubeletConn := &fakeConnectionInfoGetter{
				info: connInfo,
				err:  tt.connInfoErr,
			}

			// Configure fake pod getter (uses fakePodGetter from subresources_test.go)
			podGetter := &fakePodGetter{
				pod: tt.pod,
				err: tt.podGetErr,
			}

			r := &AllocatedREST{
				Store:       podGetter,
				KubeletConn: kubeletConn,
			}

			obj, err := r.Get(context.Background(), "foo", &metav1.GetOptions{})
			if tt.expectedErr != nil {
				require.Error(t, err)
				if statusErr, ok := err.(apierrors.APIStatus); ok {
					if expectedStatusErr, ok := tt.expectedErr.(apierrors.APIStatus); ok {
						assert.Equal(t, expectedStatusErr.Status().Code, statusErr.Status().Code)
						assert.Equal(t, expectedStatusErr.Status().Message, statusErr.Status().Message)
					} else {
						t.Errorf("expected error %v, got APIStatus %v", tt.expectedErr, statusErr)
					}
				} else {
					assert.Equal(t, tt.expectedErr.Error(), err.Error())
				}
				return
			}

			require.NoError(t, err)
			require.NotNil(t, obj)

			pod, ok := obj.(*api.Pod)
			require.True(t, ok)

			assert.Equal(t, tt.expectedPod.Name, pod.Name)
			assert.Equal(t, defaultPod(tt.expectedPod).Spec, pod.Spec)
		})
	}
}

func defaultPod(pod *api.Pod) *api.Pod {
	if pod == nil {
		return nil
	}
	if len(pod.Spec.RestartPolicy) == 0 {
		pod.Spec.RestartPolicy = api.RestartPolicyAlways
	}
	if pod.Spec.TerminationGracePeriodSeconds == nil {
		var period int64 = 30
		pod.Spec.TerminationGracePeriodSeconds = &period
	}
	if len(pod.Spec.DNSPolicy) == 0 {
		pod.Spec.DNSPolicy = api.DNSClusterFirst
	}
	if pod.Spec.SecurityContext == nil {
		pod.Spec.SecurityContext = &api.PodSecurityContext{}
	}
	if len(pod.Spec.SchedulerName) == 0 {
		pod.Spec.SchedulerName = "default-scheduler"
	}
	if pod.Spec.EnableServiceLinks == nil {
		enable := true
		pod.Spec.EnableServiceLinks = &enable
	}
	for i := range pod.Spec.Containers {
		c := &pod.Spec.Containers[i]
		if len(c.TerminationMessagePath) == 0 {
			c.TerminationMessagePath = "/dev/termination-log"
		}
		if len(c.TerminationMessagePolicy) == 0 {
			c.TerminationMessagePolicy = api.TerminationMessageReadFile
		}
		if len(c.ImagePullPolicy) == 0 {
			c.ImagePullPolicy = api.PullAlways
		}
	}
	return pod
}
