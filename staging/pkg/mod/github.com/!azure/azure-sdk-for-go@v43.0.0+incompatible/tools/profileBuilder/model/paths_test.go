// +build go1.9

// Copyright 2018 Microsoft Corporation and contributors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package model

import (
	"path/filepath"
	"reflect"
	"testing"
)

func TestDeconstructPath(t *testing.T) {
	type testcase struct {
		name string
		path string
		pi   PathInfo
	}
	testcases := []testcase{
		{
			name: "arm1",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "storage", "mgmt", "2016-01-01", "storage"),
			pi: PathInfo{
				IsArm:    true,
				Provider: "storage",
				Version:  "2016-01-01",
				Group:    "storage",
			},
		},
		{
			name: "arm2",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "datalake", "analytics", "mgmt", "2016-11-01", "account"),
			pi: PathInfo{
				IsArm:    true,
				Provider: filepath.Join("datalake", "analytics"),
				Version:  "2016-11-01",
				Group:    "account",
			},
		},
		{
			name: "arm3",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "datalake", "analytics", "mgmt", "2016-11-01", "account", "v2"),
			pi: PathInfo{
				IsArm:    true,
				Provider: filepath.Join("datalake", "analytics"),
				Version:  "2016-11-01",
				Group:    "account",
				ModVer:   "v2",
			},
		},
		{
			name: "arm4",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "datalake", "analytics", "mgmt", "2016-11-01", "account", "v2", "accountapi"),
			pi: PathInfo{
				IsArm:    true,
				Provider: filepath.Join("datalake", "analytics"),
				Version:  "2016-11-01",
				Group:    "account",
				ModVer:   "v2",
				APIPkg:   "accountapi",
			},
		},
		{
			name: "arm5",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "storage", "mgmt", "2016-01-01", "storage", "v10"),
			pi: PathInfo{
				IsArm:    true,
				Provider: "storage",
				Version:  "2016-01-01",
				Group:    "storage",
				ModVer:   "v10",
			},
		},
		{
			name: "arm6",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "storage", "mgmt", "2016-01-01", "storage", "v10", "storageapi"),
			pi: PathInfo{
				IsArm:    true,
				Provider: "storage",
				Version:  "2016-01-01",
				Group:    "storage",
				ModVer:   "v10",
				APIPkg:   "storageapi",
			},
		},
		{
			name: "arm7",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "storage", "mgmt", "2016-01-01", "storage", "storageapi"),
			pi: PathInfo{
				IsArm:    true,
				Provider: "storage",
				Version:  "2016-01-01",
				Group:    "storage",
				APIPkg:   "storageapi",
			},
		},
		{
			name: "arm8",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "datalake", "analytics", "mgmt", "2016-11-01", "account", "accountapi"),
			pi: PathInfo{
				IsArm:    true,
				Provider: filepath.Join("datalake", "analytics"),
				Version:  "2016-11-01",
				Group:    "account",
				APIPkg:   "accountapi",
			},
		},
		{
			name: "dataplane1",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "cognitiveservices", "v2.0", "luis", "authoring"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "cognitiveservices",
				Version:  "v2.0",
				Group:    filepath.Join("luis", "authoring"),
			},
		},
		{
			name: "dataplane2",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "datalake", "analytics", "2016-11-01", "job"),
			pi: PathInfo{
				IsArm:    false,
				Provider: filepath.Join("datalake", "analytics"),
				Version:  "2016-11-01",
				Group:    "job",
			},
		},
		{
			name: "dataplane3",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "appinsights", "v1", "insights"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "appinsights",
				Version:  "v1",
				Group:    "insights",
			},
		},
		{
			name: "dataplane4",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "cognitiveservices", "v2.0", "luis", "authoring", "v2"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "cognitiveservices",
				Version:  "v2.0",
				Group:    filepath.Join("luis", "authoring"),
				ModVer:   "v2",
			},
		},
		{
			name: "dataplane5",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "cognitiveservices", "v2.0", "luis", "authoring", "v2", "authoringapi"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "cognitiveservices",
				Version:  "v2.0",
				Group:    filepath.Join("luis", "authoring"),
				ModVer:   "v2",
				APIPkg:   "authoringapi",
			},
		},
		{
			name: "dataplane6",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "appinsights", "v1", "insights", "v4"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "appinsights",
				Version:  "v1",
				Group:    "insights",
				ModVer:   "v4",
			},
		},
		{
			name: "dataplane7",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "appinsights", "v1", "insights", "insightsapi"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "appinsights",
				Version:  "v1",
				Group:    "insights",
				APIPkg:   "insightsapi",
			},
		},
		{
			name: "dataplane8",
			path: filepath.Join("work", "src", "github.com", "Azure", "azure-sdk-for-go", "services", "appinsights", "v1", "insights", "v4", "insightsapi"),
			pi: PathInfo{
				IsArm:    false,
				Provider: "appinsights",
				Version:  "v1",
				Group:    "insights",
				ModVer:   "v4",
				APIPkg:   "insightsapi",
			},
		},
	}
	for _, tc := range testcases {
		t.Run(tc.name, func(t *testing.T) {
			p, err := DeconstructPath(tc.path)
			if err != nil {
				t.Fatalf("failed to deconstruct path: %v", err)
			}
			if !reflect.DeepEqual(p, tc.pi) {
				t.Fatalf("expected %+v, got %+v", tc.pi, p)
			}
		})
	}
}
