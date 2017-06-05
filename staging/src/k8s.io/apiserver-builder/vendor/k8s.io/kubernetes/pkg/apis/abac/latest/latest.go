/*
Copyright 2014 The Kubernetes Authors.

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

package latest

import (
	_ "k8s.io/kubernetes/pkg/apis/abac"
	_ "k8s.io/kubernetes/pkg/apis/abac/v0"
	_ "k8s.io/kubernetes/pkg/apis/abac/v1beta1"
)

// TODO: this file is totally wrong, it should look like other latest files.
// lavalamp is in the middle of fixing this code, so wait for the new way of doing things..
