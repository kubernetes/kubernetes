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

package common

import (
	"k8s.io/kubernetes/pkg/util/sets"
)

type Suite string

const (
	E2E     Suite = "e2e"
	NodeE2E Suite = "node e2e"
)

var CurrentSuite Suite

// CommonImageWhiteList is the list of images used in common test. These images should be prepulled
// before a tests starts, so that the tests won't fail due image pulling flakes. Currently, this is
// only used by node e2e test.
// TODO(random-liu): Change the image puller pod to use similar mechanism.
var CommonImageWhiteList = sets.NewString(
	"gcr.io/google_containers/busybox:1.24",
	"gcr.io/google_containers/eptest:0.1",
	"gcr.io/google_containers/liveness:e2e",
	"gcr.io/google_containers/mounttest:0.7",
	"gcr.io/google_containers/mounttest-user:0.3",
	"gcr.io/google_containers/netexec:1.4",
	"gcr.io/google_containers/netexec:1.5",
	"gcr.io/google_containers/nginx-slim:0.7",
	"gcr.io/google_containers/serve_hostname:v1.4",
	"gcr.io/google_containers/test-webserver:e2e",
	"gcr.io/google_containers/hostexec:1.2",
	"gcr.io/google_containers/volume-nfs:0.8",
	"gcr.io/google_containers/volume-gluster:0.2",
)
