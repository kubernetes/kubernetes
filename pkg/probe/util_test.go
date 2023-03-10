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
	container := v1.Container{
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
	}
	want := 8080
	got, err := findPortByName(&container, "foo")
	if got != want || err != nil {
		t.Errorf("Expected %v, got %v, err: %v", want, got, err)
	}
}

func TestResolveContainerPort(t *testing.T) {
	type args struct {
		param     intstr.IntOrString
		container v1.Container
	}
	tests := []struct {
		name    string
		args    args
		want    int
		wantErr bool
	}{
		{
			name: "get port by int val ",
			args: args{
				param: intstr.IntOrString{
					Type:   0,
					IntVal: 80,
					StrVal: "foo",
				},
				container: v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
					},
				},
			},
			want:    80,
			wantErr: false,
		},
		{
			name: "get port by string val",
			args: args{
				param: intstr.IntOrString{
					Type:   1,
					IntVal: 80,
					StrVal: "foo",
				},
				container: v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
					},
				},
			},
			want:    8080,
			wantErr: false,
		},
		{
			name: "get port by invalid type",
			args: args{
				param: intstr.IntOrString{
					Type:   20,
					IntVal: 80,
					StrVal: "foo",
				},
				container: v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 8080,
						},
					},
				},
			},
			want:    -1,
			wantErr: true,
		},
		{
			name: "get invalid container port",
			args: args{
				param: intstr.IntOrString{
					Type:   1,
					StrVal: "foo",
				},
				container: v1.Container{
					Ports: []v1.ContainerPort{
						{
							Name:          "foo",
							ContainerPort: 80800,
						},
					},
				},
			},
			want:    80800,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ResolveContainerPort(tt.args.param, &tt.args.container)
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
