/*
Copyright 2020 The Kubernetes Authors.

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

package config

import (
	"io/ioutil"
	"os"
	"reflect"
	"testing"

	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	internal "k8s.io/controller-manager/config"
)

func TestReadLeaderMigrationConfiguration(t *testing.T) {
	testCases := []struct {
		name      string
		content   string
		expected  *internal.LeaderMigrationConfiguration
		expectErr bool
	}{
		{
			name:      "empty",
			content:   "",
			expected:  nil,
			expectErr: true,
		},
		{
			name: "wrong type",
			content: `
apiVersion: kubelet.config.k8s.io/v1beta1
kind: KubeletConfiguration
`,
			expected:  nil,
			expectErr: true,
		},
		{
			name: "basic",
			content: `
apiVersion: controllermanager.config.k8s.io/v1alpha1
kind: LeaderMigrationConfiguration
leaderName: migration-120-to-121
resourceLock: leases
controllerLeaders: []
`,
			expected: &internal.LeaderMigrationConfiguration{
				TypeMeta:          metav1.TypeMeta{},
				LeaderName:        "migration-120-to-121",
				ResourceLock:      "leases",
				ControllerLeaders: []internal.ControllerLeaderConfiguration{},
			},
			expectErr: false,
		},
		{
			name: "endpoints",
			content: `
apiVersion: controllermanager.config.k8s.io/v1alpha1
kind: LeaderMigrationConfiguration
leaderName: migration-120-to-121
resourceLock: endpoints
controllerLeaders: []
`,
			expected: &internal.LeaderMigrationConfiguration{
				TypeMeta:          metav1.TypeMeta{},
				LeaderName:        "migration-120-to-121",
				ResourceLock:      "endpoints",
				ControllerLeaders: []internal.ControllerLeaderConfiguration{},
			},
		},
		{
			name: "withLeaders",
			content: `
apiVersion: controllermanager.config.k8s.io/v1alpha1
kind: LeaderMigrationConfiguration
leaderName: migration-120-to-121
resourceLock: endpoints
controllerLeaders: 
  - name: route-controller
    component: kube-controller-manager
  - name: service-controller
    component: kube-controller-manager
`,
			expected: &internal.LeaderMigrationConfiguration{
				TypeMeta:     metav1.TypeMeta{},
				LeaderName:   "migration-120-to-121",
				ResourceLock: "endpoints",
				ControllerLeaders: []internal.ControllerLeaderConfiguration{
					{
						Name:      "route-controller",
						Component: "kube-controller-manager",
					},
					{
						Name:      "service-controller",
						Component: "kube-controller-manager",
					},
				},
			},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			configFile, err := ioutil.TempFile("", tc.name)
			if err != nil {
				t.Fatal(err)
			}
			defer os.Remove(configFile.Name())
			err = ioutil.WriteFile(configFile.Name(), []byte(tc.content), os.FileMode(0755))
			if err != nil {
				t.Fatal(err)
			}
			result, err := ReadLeaderMigrationConfiguration(configFile.Name())
			if tc.expectErr && err == nil {
				t.Errorf("unexpected no error for %s", tc.name)
			} else if !tc.expectErr && err != nil {
				t.Errorf("get error from ReadLeaderElectionConfiguration: %#v", err)
			} else if !reflect.DeepEqual(result, tc.expected) {
				t.Errorf("result not matching expected, got %#v, expected %#v", result, tc.expected)
			}
		})
	}
}

func TestValidateLeaderMigrationConfiguration(t *testing.T) {
	testCases := []struct {
		name      string
		config    *internal.LeaderMigrationConfiguration
		expectErr bool
	}{
		{
			name: "empty name",
			config: &internal.LeaderMigrationConfiguration{
				LeaderName:        "",
				ResourceLock:      ResourceLockLeases,
				ControllerLeaders: []internal.ControllerLeaderConfiguration{},
			},
			expectErr: true,
		},
		{
			name: "invalid resourceLock",
			config: &internal.LeaderMigrationConfiguration{
				LeaderName:        "test",
				ResourceLock:      "invalid",
				ControllerLeaders: []internal.ControllerLeaderConfiguration{},
			},
			expectErr: true,
		},
		{
			name: "empty controllerLeaders (valid)",
			config: &internal.LeaderMigrationConfiguration{
				LeaderName:        "test",
				ResourceLock:      ResourceLockLeases,
				ControllerLeaders: []internal.ControllerLeaderConfiguration{},
			},
			expectErr: false,
		},
		{
			name: "endpoints",
			config: &internal.LeaderMigrationConfiguration{
				TypeMeta:     metav1.TypeMeta{},
				LeaderName:   "migration-120-to-121",
				ResourceLock: ResourceLockEndpoints,
				ControllerLeaders: []internal.ControllerLeaderConfiguration{
					{
						Name:      "route-controller",
						Component: "kube-controller-manager",
					},
				},
			},
			expectErr: false,
		},
	}
	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			errs := ValidateLeaderMigrationConfiguration(tc.config)
			if tc.expectErr && len(errs) == 0 {
				t.Errorf("calling ValidateLeaderMigrationConfiguration expected errors but got no error")
			}
			if !tc.expectErr && len(errs) != 0 {
				t.Errorf("calling ValidateLeaderMigrationConfiguration expected no error but got %v", errs)
			}
		})
	}
}
