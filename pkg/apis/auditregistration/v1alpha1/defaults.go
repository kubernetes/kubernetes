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

package v1alpha1

import (
	auditregistrationv1alpha1 "k8s.io/api/auditregistration/v1alpha1"
	"k8s.io/apimachinery/pkg/runtime"
	utilpointer "k8s.io/utils/pointer"
)

const (
	// DefaultQPS is the default QPS value
	DefaultQPS = int64(10)
	// DefaultBurst is the default burst value
	DefaultBurst = int64(15)
)

// DefaultThrottle is a default throttle config
func DefaultThrottle() *auditregistrationv1alpha1.WebhookThrottleConfig {
	return &auditregistrationv1alpha1.WebhookThrottleConfig{
		QPS:   utilpointer.Int64Ptr(DefaultQPS),
		Burst: utilpointer.Int64Ptr(DefaultBurst),
	}
}

func addDefaultingFuncs(scheme *runtime.Scheme) error {
	return RegisterDefaults(scheme)
}

// SetDefaults_AuditSink sets defaults if the audit sink isn't present
func SetDefaults_AuditSink(obj *auditregistrationv1alpha1.AuditSink) {
	if obj.Spec.Webhook.Throttle != nil {
		if obj.Spec.Webhook.Throttle.QPS == nil {
			obj.Spec.Webhook.Throttle.QPS = utilpointer.Int64Ptr(DefaultQPS)
		}
		if obj.Spec.Webhook.Throttle.Burst == nil {
			obj.Spec.Webhook.Throttle.Burst = utilpointer.Int64Ptr(DefaultBurst)
		}
	} else {
		obj.Spec.Webhook.Throttle = DefaultThrottle()
	}
}

// SetDefaults_ServiceReference sets defaults for AuditSync Webhook's ServiceReference
func SetDefaults_ServiceReference(obj *auditregistrationv1alpha1.ServiceReference) {
	if obj.Port == nil {
		obj.Port = utilpointer.Int32Ptr(443)
	}
}
