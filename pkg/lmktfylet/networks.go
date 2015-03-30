/*
Copyright 2014 Google Inc. All rights reserved.

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

package lmktfylet

import (
	"github.com/GoogleCloudPlatform/lmktfy/pkg/api"
	"github.com/GoogleCloudPlatform/lmktfy/pkg/client"
)

// This just exports required functions from lmktfylet proper, for use by network
// plugins.
type networkHost struct {
	lmktfylet *LMKTFYlet
}

func (nh *networkHost) GetPodByName(name, namespace string) (*api.Pod, bool) {
	return nh.lmktfylet.GetPodByName(name, namespace)
}

func (nh *networkHost) GetLMKTFYClient() client.Interface {
	return nh.lmktfylet.lmktfyClient
}
