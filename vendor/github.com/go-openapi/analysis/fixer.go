// Copyright 2015 go-swagger maintainers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package analysis

import "github.com/go-openapi/spec"

// FixEmptyResponseDescriptions replaces empty ("") response
// descriptions in the input with "(empty)" to ensure that the
// resulting Swagger is stays valid.  The problem appears to arise
// from reading in valid specs that have a explicit response
// description of "" (valid, response.description is required), but
// due to zero values being omitted upon re-serializing (omitempty) we
// lose them unless we stick some chars in there.
func FixEmptyResponseDescriptions(s *spec.Swagger) {
	if s.Paths != nil {
		for _, v := range s.Paths.Paths {
			if v.Get != nil {
				FixEmptyDescs(v.Get.Responses)
			}
			if v.Put != nil {
				FixEmptyDescs(v.Put.Responses)
			}
			if v.Post != nil {
				FixEmptyDescs(v.Post.Responses)
			}
			if v.Delete != nil {
				FixEmptyDescs(v.Delete.Responses)
			}
			if v.Options != nil {
				FixEmptyDescs(v.Options.Responses)
			}
			if v.Head != nil {
				FixEmptyDescs(v.Head.Responses)
			}
			if v.Patch != nil {
				FixEmptyDescs(v.Patch.Responses)
			}
		}
	}
	for k, v := range s.Responses {
		FixEmptyDesc(&v)
		s.Responses[k] = v
	}
}

// FixEmptyDescs adds "(empty)" as the description for any Response in
// the given Responses object that doesn't already have one.
func FixEmptyDescs(rs *spec.Responses) {
	FixEmptyDesc(rs.Default)
	for k, v := range rs.StatusCodeResponses {
		FixEmptyDesc(&v)
		rs.StatusCodeResponses[k] = v
	}
}

// FixEmptyDesc adds "(empty)" as the description to the given
// Response object if it doesn't already have one and isn't a
// ref. No-op on nil input.
func FixEmptyDesc(rs *spec.Response) {
	if rs == nil || rs.Description != "" || rs.Ref.Ref.GetURL() != nil {
		return
	}
	rs.Description = "(empty)"
}
