/*
Copyright 2017 The Kubernetes Authors.

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

package apps

import (
	"k8s.io/apimachinery/pkg/runtime/schema"
	batchv2alpha1 "k8s.io/kubernetes/pkg/apis/batch/v2alpha1"
)

const (
	NautilusImage  = "gcr.io/google_containers/update-demo:nautilus"
	KittenImage    = "gcr.io/google_containers/update-demo:kitten"
	NginxImage     = "gcr.io/google_containers/nginx-slim:0.7"
	NginxImageName = "nginx"
	RedisImage     = "gcr.io/k8s-testimages/redis:e2e"
	RedisImageName = "redis"
	NewNginxImage  = "gcr.io/google_containers/nginx-slim:0.8"
)

var (
	CronJobGroupVersionResource = schema.GroupVersionResource{Group: batchv2alpha1.GroupName, Version: "v2alpha1", Resource: "cronjobs"}
)
