/*
Copyright 2016 The Kubernetes Authors.

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

package federatedtypes

import (
	"k8s.io/apimachinery/pkg/runtime"
)

// Definition of the hooks usable in the reconcile process of sync adapters.
// Controllers which need additional logic in reconcile can use these hooks to implement the same.
// Reconcile of sync controller will use them as an add on if they are present on an adapter.
type ScheduleHook func(runtime.Object, map[string]runtime.Object) (map[string]runtime.Object, error)
type UpdateFedSpecHook func(runtime.Object, map[string]runtime.Object) (runtime.Object, error)
