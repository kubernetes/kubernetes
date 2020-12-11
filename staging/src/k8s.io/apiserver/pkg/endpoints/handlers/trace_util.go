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

package handlers

import (
	"net/http"

	utiltrace "k8s.io/utils/trace"
)

func traceFields(req *http.Request) []utiltrace.Field {
	return []utiltrace.Field{
		{Key: "url", Value: req.URL.Path},
		{Key: "user-agent", Value: &lazyTruncatedUserAgent{req: req}},
		{Key: "client", Value: &lazyClientIP{req: req}},
		{Key: "accept", Value: &lazyAccept{req: req}},
		{Key: "protocol", Value: req.Proto}}
}
