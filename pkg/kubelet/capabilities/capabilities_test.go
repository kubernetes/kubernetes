/*
Copyright 2019 The Kubernetes Authors.

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

package capabilities

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/types"
	"k8s.io/kubernetes/pkg/kubelet/lifecycle"
	kubetypes "k8s.io/kubernetes/pkg/kubelet/types"
)

var testuid = types.UID("71f0f627-1565-11e9-9b0c-52540088d919")

func Test_capabilities_Admit(t *testing.T) {

	tests := []struct {
		name string
		args *lifecycle.PodAdmitAttributes
		cap  *Capabilities
		want lifecycle.PodAdmitResult
	}{
		{
			name: "privileged container not allowed",
			args: &lifecycle.PodAdmitAttributes{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: testuid},
					Spec: v1.PodSpec{
						Containers: []v1.Container{getContainer("test", true)},
					},
				},
			},
			cap:  getCapabilities(withAllowedPrivilege(false)),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified privileged container, but is disallowed", testuid)),
		},
		{
			name: "privileged init container not allowed",
			args: &lifecycle.PodAdmitAttributes{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: testuid},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{getContainer("test-init", true)},
					},
				},
			},
			cap:  getCapabilities(withAllowedPrivilege(false)),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified privileged init container, but is disallowed", testuid)),
		},
		{
			name: "no privileged containers",
			args: &lifecycle.PodAdmitAttributes{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: testuid},
					Spec: v1.PodSpec{
						InitContainers: []v1.Container{getContainer("test-init", false)},
						Containers:     []v1.Container{getContainer("test", false)},
					},
				},
			},
			cap:  getCapabilities(withAllowedPrivilege(false)),
			want: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "privileged containers allowed",
			args: &lifecycle.PodAdmitAttributes{
				Pod: &v1.Pod{
					ObjectMeta: metav1.ObjectMeta{UID: testuid},
					Spec: v1.PodSpec{
						Containers: []v1.Container{getContainer("test", true)},
					},
				},
			},
			cap:  getCapabilities(withAllowedPrivilege(true)),
			want: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "allowed host network",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostNetwork(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostNetworkSources(kubetypes.FileSource)),
			want: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "wrong host network source",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostNetwork(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, "ftp"),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host networking, but is disallowed", testuid)),
		},
		{
			name: "wrong host network annotation in pod",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostNetwork(),
					withAnnotation("foobar", kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host networking, but is disallowed", testuid)),
		},
		{
			name: "One host network source allowed and other provided",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostNetwork(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostNetworkSources(kubetypes.HTTPSource, kubetypes.ApiserverSource)),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host networking, but is disallowed", testuid)),
		},
		{
			name: "allowed host PID",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostPID(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostPIDSources(kubetypes.FileSource)),
			want: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "wrong host PID source",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostPID(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, "ftp"),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host PID, but is disallowed", testuid)),
		},
		{
			name: "wrong host PID annotation in pod",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostPID(),
					withAnnotation("foobar", kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host PID, but is disallowed", testuid)),
		},
		{
			name: "One host PID source allowed and other provided",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostPID(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostPIDSources(kubetypes.HTTPSource, kubetypes.ApiserverSource)),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host PID, but is disallowed", testuid)),
		},
		{
			name: "allowed host IPC",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostIPC(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostIPCSources(kubetypes.FileSource)),
			want: lifecycle.PodAdmitResult{Admit: true},
		},
		{
			name: "wrong host IPC source",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostIPC(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, "ftp"),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host IPC, but is disallowed", testuid)),
		},
		{
			name: "wrong host IPC annotation in pod",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostIPC(),
					withAnnotation("foobar", kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host IPC, but is disallowed", testuid)),
		},
		{
			name: "One host IPC source allowed and other provided",
			args: &lifecycle.PodAdmitAttributes{
				Pod: getPod(withHostIPC(),
					withAnnotation(kubetypes.ConfigSourceAnnotationKey, kubetypes.FileSource),
				),
			},
			cap:  getCapabilities(withHostIPCSources(kubetypes.HTTPSource, kubetypes.ApiserverSource)),
			want: rejectPod(fmt.Sprintf("pod with UID %q specified host IPC, but is disallowed", testuid)),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			c := tt.cap
			if got := c.Admit(tt.args); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("capabilities.Admit() = %v, want %v", got, tt.want)
			}
		})
	}
}

func boolToPtr(b bool) *bool {
	return &b
}

func getPod(fns ...applyPod) *v1.Pod {
	pod := &v1.Pod{
		ObjectMeta: metav1.ObjectMeta{UID: testuid},
	}
	for _, fn := range fns {
		fn(pod)
	}
	return pod
}

type applyPod func(*v1.Pod)

func withAnnotation(k, v string) applyPod {
	return func(pod *v1.Pod) {
		pod.Annotations = map[string]string{k: v}
	}
}

func withHostNetwork() applyPod {
	return func(pod *v1.Pod) {
		pod.Spec.HostNetwork = true
	}
}

func withHostPID() applyPod {
	return func(pod *v1.Pod) {
		pod.Spec.HostPID = true
	}
}

func withHostIPC() applyPod {
	return func(pod *v1.Pod) {
		pod.Spec.HostIPC = true
	}
}

func getCapabilities(fns ...applyCapabilities) *Capabilities {
	caps := &Capabilities{}
	for _, fn := range fns {
		fn(caps)
	}
	return caps
}

type applyCapabilities func(*Capabilities)

func withAllowedPrivilege(allow bool) applyCapabilities {
	return func(caps *Capabilities) {
		caps.Capabilities.AllowPrivileged = allow
	}
}

func withHostNetworkSources(sources ...string) applyCapabilities {
	return func(caps *Capabilities) {
		caps.Capabilities.PrivilegedSources.HostNetworkSources = sources
	}
}

func withHostPIDSources(sources ...string) applyCapabilities {
	return func(caps *Capabilities) {
		caps.Capabilities.PrivilegedSources.HostPIDSources = sources
	}
}

func withHostIPCSources(sources ...string) applyCapabilities {
	return func(caps *Capabilities) {
		caps.Capabilities.PrivilegedSources.HostIPCSources = sources
	}
}

func getContainer(name string, privileged bool) v1.Container {
	return v1.Container{
		Name: name,
		SecurityContext: &v1.SecurityContext{
			Privileged: boolToPtr(privileged),
		},
	}
}
