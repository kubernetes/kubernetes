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

package cm

import (
	"fmt"
	"reflect"
	"testing"

	v1 "k8s.io/api/core/v1"
	podresourcesapi "k8s.io/kubelet/pkg/apis/podresources/v1"
	"k8s.io/kubernetes/pkg/kubelet/cm/devicemanager"
)

func Test_parsePercentage(t *testing.T) {

	// Part 1 of the test, iterate through all values 0-100%
	t.Run("validTestswithPercent", func(t *testing.T) {
		for i := 0; i < 100; i++ {
			value := fmt.Sprintf("%d", i) + "%"
			if p, e := parsePercentage(value); p != int64(i) || e != nil {
				t.Errorf("parsePercentage() failed for value %d, error: %v", i, e)
			}
		}
	})

	// Part 2 give a few error values
	tests := []struct {
		name    string
		v       string
		want    int64
		wantErr bool
	}{
		{
			name:    "invalid1",
			v:       "105%",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid2",
			v:       "-87",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid3",
			v:       "258",
			want:    0,
			wantErr: true,
		},
		{
			name:    "invalid4",
			v:       "-38%",
			want:    0,
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parsePercentage(tt.v)
			if (err != nil) != tt.wantErr {
				t.Errorf("parsePercentage() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("parsePercentage() = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestParseQOSReserved(t *testing.T) {
	type args struct {
		m map[string]string
	}
	tests := []struct {
		name    string
		args    args
		want    *map[v1.ResourceName]int64
		wantErr bool
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := ParseQOSReserved(tt.args.m)
			if (err != nil) != tt.wantErr {
				t.Errorf("ParseQOSReserved() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("ParseQOSReserved() = %v, want %v", got, tt.want)
			}
		})
	}
}

func Test_containerDevicesFromResourceDeviceInstances(t *testing.T) {
	type args struct {
		devs devicemanager.ResourceDeviceInstances
	}
	tests := []struct {
		name string
		args args
		want []*podresourcesapi.ContainerDevices
	}{
		// TODO: Add test cases.
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := containerDevicesFromResourceDeviceInstances(tt.args.devs); !reflect.DeepEqual(got, tt.want) {
				t.Errorf("containerDevicesFromResourceDeviceInstances() = %v, want %v", got, tt.want)
			}
		})
	}
}
