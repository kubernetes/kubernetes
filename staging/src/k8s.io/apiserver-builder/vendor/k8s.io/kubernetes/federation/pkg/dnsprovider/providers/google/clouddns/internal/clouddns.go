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

package internal

// Implementation of internal/interfaces/* on top of Google Cloud DNS API.
// See https://godoc.org/google.golang.org/api/dns/v1 for details
// This facilitates stubbing out Google Cloud DNS for unit testing.
// Only the parts of the API that we use are included.
// Others can be added as needed.

import dns "google.golang.org/api/dns/v1"

type (
	Project struct{ impl *dns.Project }

	ProjectsGetCall struct{ impl *dns.ProjectsGetCall }

	ProjectsService struct{ impl *dns.ProjectsService }

	Quota struct{ impl *dns.Quota }
)
