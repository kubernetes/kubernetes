/*
Copyright 2022 The Kubernetes Authors.

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

package probe

import (
	"testing"

	v1 "k8s.io/api/core/v1"
	"k8s.io/apimachinery/pkg/util/intstr"
)

func TestFindPortByName(t *testing.T) {
	t.Parallel()
	type args struct {
		container *v1.Container
		portName  string
	}
	tests := []struct {
		name    string
		args    args
		want    int
		wantErr bool
	}{
		{
			name: "get port from exist port name",
			args: args{
				container: &v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
						{
							Name:          "bar",
							ContainerPort: 9000,
						},
					},
				},
				portName: "foo",
			},
			want:    8080,
			wantErr: false,
		},
		{
			name: "get port from not exist port name",
			args: args{
				container: &v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
						{
							Name:          "bar",
							ContainerPort: 9000,
						},
					},
				},
				portName: "http",
			},
			want:    0,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := findPortByName(tt.args.container, tt.args.portName)
			if (err != nil) != tt.wantErr {
				t.Errorf("findPortByName() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("findPortByName() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestResolveContainerPort(t *testing.T) {
	t.Parallel()
	type args struct {
		param     intstr.IntOrString
		container *v1.Container
	}
	tests := []struct {
		name    string
		args    args
		want    int
		wantErr bool
	}{
		{
			name: "get port by int type",
			args: args{
				param:     intstr.IntOrString{Type: 0, IntVal: 443},
				container: &v1.Container{},
			},
			want:    443,
			wantErr: false,
		},
		{
			name: "invalid port",
			args: args{
				param:     intstr.IntOrString{Type: 0, IntVal: 66666},
				container: &v1.Container{},
			},
			want:    66666,
			wantErr: true,
		},
		{
			name: "get port by port name",
			args: args{
				param: intstr.IntOrString{Type: 1, StrVal: "foo"},
				container: &v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
						{
							Name:          "bar",
							ContainerPort: 9000,
						},
					},
				},
			},
			want:    8080,
			wantErr: false,
		},
		{
			name: "no port name",
			args: args{
				param: intstr.IntOrString{Type: 1, StrVal: "foo"},
				container: &v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "bar",
							ContainerPort: 9000,
						},
					},
				},
			},
			want:    0,
			wantErr: true,
		},
		{
			name: "invalid param type",
			args: args{
				param: intstr.IntOrString{Type: 2, StrVal: "foo"},
				container: &v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
						{
							Name:          "bar",
							ContainerPort: 9000,
						},
					},
				},
			},
			want:    -1,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Parallel()
			got, err := ResolveContainerPort(tt.args.param, tt.args.container)
			if (err != nil) != tt.wantErr {
				t.Errorf("ResolveContainerPort() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("ResolveContainerPort() = %v, want %v", got, tt.want)
			}
		})
	}
}
