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

package filters

import (
	"errors"
	"fmt"
	"net/http"

	"k8s.io/apimachinery/pkg/runtime/schema"
	"k8s.io/apiserver/pkg/endpoints/handlers/responsewriters"
	"k8s.io/apiserver/pkg/endpoints/request"
	"k8s.io/apiserver/pkg/storageversion"
	_ "k8s.io/component-base/metrics/prometheus/workqueue" // for workqueue metric registration
	"k8s.io/klog/v2"
)

// WithStorageVersionPrecondition checks if the storage version barrier has
// completed, if not, it only passes the following API requests:
// 1. non-resource requests,
// 2. read requests,
// 3. write requests to the storageversion API,
// 4. resources whose StorageVersion is not pending update, including non-persisted resources.
func WithStorageVersionPrecondition(handler http.Handler, svm storageversion.Manager) http.Handler {
	if svm == nil {
		// TODO(roycaihw): switch to warning after the feature graduate to beta/GA
		klog.V(2).Infof("Storage Version barrier is disabled")
		return handler
	}
	return http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		if svm.Completed() {
			handler.ServeHTTP(w, req)
			return
		}
		ctx := req.Context()
		requestInfo, found := request.RequestInfoFrom(ctx)
		if !found {
			responsewriters.InternalError(w, req, errors.New("no RequestInfo found in the context"))
			return
		}
		// Allow non-resource requests
		if !requestInfo.IsResourceRequest {
			handler.ServeHTTP(w, req)
			return
		}
		// Allow read requests
		if requestInfo.Verb == "get" || requestInfo.Verb == "list" || requestInfo.Verb == "watch" {
			handler.ServeHTTP(w, req)
			return
		}
		// Allow writes to the storage version API
		if requestInfo.APIGroup == "internal.apiserver.k8s.io" && requestInfo.Resource == "storageversions" {
			handler.ServeHTTP(w, req)
			return
		}
		// If the resource's StorageVersion is not in the to-be-updated list, let it pass.
		// Non-persisted resources are not in the to-be-updated list, so they will pass.
		gr := schema.GroupResource{requestInfo.APIGroup, requestInfo.Resource}
		if !svm.PendingUpdate(gr) {
			handler.ServeHTTP(w, req)
			return
		}

		responsewriters.ServiceUnavailabeError(w, req, errors.New(fmt.Sprintf("wait for storage version registration to complete for resource: %v, last seen error: %v", gr, svm.LastUpdateError(gr))))
	})
}
