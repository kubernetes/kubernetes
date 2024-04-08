/*
Copyright 2024 The Kubernetes Authors.

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

package restproxy

import (
	"fmt"
	"net/http"

	"github.com/go-logr/logr"

	restproxyapi "k8s.io/dynamic-resource-allocation/apis/restproxy/v1alpha1"
	"k8s.io/klog/v2"
	"k8s.io/utils/ptr"
)

type logRequest struct {
	logger  *klog.Logger
	request *restproxyapi.Request
}

var _ logr.Marshaler = logRequest{}

func (r logRequest) MarshalLog() any {
	// Strip down the request depending on verbosity.
	// Without any logger, only the ID gets logged.
	if r.logger == nil {
		return map[string]int64{"id": r.request.Id}
	}

	request := *r.request
	if !r.logger.V(6).Enabled() {
		request.Body = []byte(fmt.Sprintf("<%d bytes truncated>", len(r.request.Body)))
	}
	// Pointer to pointer "hides" the String implementation.
	return ptr.To(ptr.To(request))
}

type logResponse struct {
	logger   *klog.Logger
	response *http.Response
}

var _ logr.Marshaler = logResponse{}

func (r logResponse) MarshalLog() any {
	// Strip down the response depending on verbosity.
	// Without any logger, only the content length gets logged.
	if r.logger == nil {
		return map[string]int64{"ContentLength": r.response.ContentLength}
	}

	response := *r.response
	response.Body = nil
	response.Request = nil
	if !r.logger.V(6).Enabled() {
		response.TLS = nil
	}
	return response
}
