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

package master

// These imports are the API groups the API server will support.
import (
	_ "k8s.io/internal-api/apis/admission/install"
	_ "k8s.io/internal-api/apis/admissionregistration/install"
	_ "k8s.io/internal-api/apis/apps/install"
	_ "k8s.io/internal-api/apis/auditregistration/install"
	_ "k8s.io/internal-api/apis/authentication/install"
	_ "k8s.io/internal-api/apis/authorization/install"
	_ "k8s.io/internal-api/apis/autoscaling/install"
	_ "k8s.io/internal-api/apis/batch/install"
	_ "k8s.io/internal-api/apis/certificates/install"
	_ "k8s.io/internal-api/apis/coordination/install"
	_ "k8s.io/internal-api/apis/core/install"
	_ "k8s.io/internal-api/apis/discovery/install"
	_ "k8s.io/internal-api/apis/events/install"
	_ "k8s.io/internal-api/apis/extensions/install"
	_ "k8s.io/internal-api/apis/imagepolicy/install"
	_ "k8s.io/internal-api/apis/networking/install"
	_ "k8s.io/internal-api/apis/node/install"
	_ "k8s.io/internal-api/apis/policy/install"
	_ "k8s.io/internal-api/apis/rbac/install"
	_ "k8s.io/internal-api/apis/scheduling/install"
	_ "k8s.io/internal-api/apis/settings/install"
	_ "k8s.io/internal-api/apis/storage/install"
)
