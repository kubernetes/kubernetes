/*
Copyright 2018 The Kubernetes Authors.

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

package fuzzer

import (
	fuzz "github.com/google/gofuzz"

	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/auditregistration"
)

// Funcs returns the fuzzer functions for the auditregistration api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(obj *auditregistration.AuditSink, c fuzz.Continue) {
			c.FuzzNoCustom(obj)
			v := int64(1)
			obj.Spec.Webhook.Throttle = &auditregistration.WebhookThrottleConfig{
				QPS:   &v,
				Burst: &v,
			}
		},
	}
}
