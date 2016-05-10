/*
Copyright 2016 The Kubernetes Authors All rights reserved.

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

// Package install installs the batch.v1 API group, making it available as
// an option to all of the API encoding/decoding machinery.
package install

import (
	"k8s.io/kubernetes/pkg/apimachinery/registered"
	"k8s.io/kubernetes/pkg/apis/batch/v1"
)

func init() {
	if err := registered.AnnounceGroupVersion(&registered.GroupVersionFactory{
		GroupName:   "batch",
		VersionName: "v1",
		AddToScheme: v1.AddToScheme,
	}); err != nil {
		panic(err)
	}
}
