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

package rest

import (
	"k8s.io/apimachinery/pkg/runtime"
	genericapirequest "k8s.io/apiserver/pkg/endpoints/request"
)

// RESTExportStrategy is the interface that defines how to export a Kubernetes
// object.  An exported object is stripped of non-user-settable fields and
// optionally, the identifying information related to the object's identity in
// the cluster so that it can be loaded into a different namespace or entirely
// different cluster without conflict.
type RESTExportStrategy interface {
	// Export strips fields that can not be set by the user.  If 'exact' is false
	// fields specific to the cluster are also stripped
	Export(ctx genericapirequest.Context, obj runtime.Object, exact bool) error
}
