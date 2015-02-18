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

package v1beta2

import (
	"strings"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/util"
	"github.com/golang/glog"
)

func init() {
	api.Scheme.AddDefaultingFuncs(
		func(obj *Volume) {
			if util.AllPtrFieldsNil(&obj.Source) {
				glog.Errorf("Defaulting volume source for %v", obj)
				obj.Source = VolumeSource{
					EmptyDir: &EmptyDir{},
				}
			}
		},
		func(obj *Port) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
		},
		func(obj *Container) {
			if obj.ImagePullPolicy == "" {
				// TODO(dchen1107): Move ParseImageName code to pkg/util
				parts := strings.Split(obj.Image, ":")
				// Check image tag
				if parts[len(parts)-1] == "latest" {
					obj.ImagePullPolicy = PullAlways
				} else {
					obj.ImagePullPolicy = PullIfNotPresent
				}
			}
			if obj.TerminationMessagePath == "" {
				obj.TerminationMessagePath = TerminationMessagePathDefault
			}
		},
		func(obj *RestartPolicy) {
			if util.AllPtrFieldsNil(obj) {
				obj.Always = &RestartPolicyAlways{}
			}
		},
		func(obj *Service) {
			if obj.Protocol == "" {
				obj.Protocol = ProtocolTCP
			}
			if obj.SessionAffinity == "" {
				obj.SessionAffinity = AffinityTypeNone
			}
		},
		func(obj *PodSpec) {
			if obj.DNSPolicy == "" {
				obj.DNSPolicy = DNSClusterFirst
			}
		},
		func(obj *ContainerManifest) {
			if obj.DNSPolicy == "" {
				obj.DNSPolicy = DNSClusterFirst
			}
		},
		func(obj *LivenessProbe) {
			if obj.TimeoutSeconds == 0 {
				obj.TimeoutSeconds = 1
			}
		},
		func(obj *Secret) {
			if obj.Type == "" {
				obj.Type = SecretTypeOpaque
			}
		},
		func(obj *Endpoints) {
			if obj.Protocol == "" {
				obj.Protocol = "TCP"
			}
		},
	)
}
