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

package warning

import (
	restful "github.com/emicklei/go-restful"

	"k8s.io/apiserver/pkg/warning"
)

// AddWarningsHandler returns a handler that adds the provided warnings to all requests,
// then delegates to the provided handler.
func AddWarningsHandler(handler restful.RouteFunction, warnings []string) restful.RouteFunction {
	if len(warnings) == 0 {
		return handler
	}

	return func(req *restful.Request, res *restful.Response) {
		ctx := req.Request.Context()
		for _, msg := range warnings {
			warning.AddWarning(ctx, "", msg)
		}
		handler(req, res)
	}
}
