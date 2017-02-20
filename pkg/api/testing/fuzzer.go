/*
Copyright 2015 The Kubernetes Authors.

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

package testing

import (
	"fmt"
	"reflect"
	"strconv"

	"github.com/google/gofuzz"

	"k8s.io/apimachinery/pkg/api/resource"
	apitesting "k8s.io/apimachinery/pkg/api/testing"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/fields"
	"k8s.io/apimachinery/pkg/labels"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/apimachinery/pkg/util/intstr"
	"k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm"
	"k8s.io/kubernetes/pkg/api"
	"k8s.io/kubernetes/pkg/api/testapi"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/apis/autoscaling"
	"k8s.io/kubernetes/pkg/apis/batch"
	"k8s.io/kubernetes/pkg/apis/certificates"
	"k8s.io/kubernetes/pkg/apis/extensions"
	extensionsv1beta1 "k8s.io/kubernetes/pkg/apis/extensions/v1beta1"
	"k8s.io/kubernetes/pkg/apis/policy"
	"k8s.io/kubernetes/pkg/apis/rbac"
)

// overrideGenericFuncs override some generic fuzzer funcs from k8s.io/apiserver in order to have more realistic
// values in a Kubernetes context.
func overrideGenericFuncs(t apitesting.TestingCommon, codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(j *runtime.Object, c fuzz.Continue) {
			// TODO: uncomment when round trip starts from a versioned object
			if true { //c.RandBool() {
				*j = &runtime.Unknown{
					// We do not set TypeMeta here because it is not carried through a round trip
					Raw:         []byte(`{"apiVersion":"unknown.group/unknown","kind":"Something","someKey":"someValue"}`),
					ContentType: runtime.ContentTypeJSON,
				}
			} else {
				types := []runtime.Object{&api.Pod{}, &api.ReplicationController{}}
				t := types[c.Rand.Intn(len(types))]
				c.Fuzz(t)
				*j = t
			}
		},
		func(r *runtime.RawExtension, c fuzz.Continue) {
			// Pick an arbitrary type and fuzz it
			types := []runtime.Object{&api.Pod{}, &extensions.Deployment{}, &api.Service{}}
			obj := types[c.Rand.Intn(len(types))]
			c.Fuzz(obj)

			var codec runtime.Codec
			switch obj.(type) {
			case *extensions.Deployment:
				codec = apitesting.TestCodec(codecs, extensionsv1beta1.SchemeGroupVersion)
			default:
				codec = apitesting.TestCodec(codecs, v1.SchemeGroupVersion)
			}

			// Convert the object to raw bytes
			bytes, err := runtime.Encode(codec, obj)
			if err != nil {
				t.Errorf("Failed to encode object: %v", err)
				return
			}

			// Set the bytes field on the RawExtension
			r.Raw = bytes
		},
	}
}

func extensionFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(j *extensions.DeploymentStrategy, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			// Ensure that strategyType is one of valid values.
			strategyTypes := []extensions.DeploymentStrategyType{extensions.RecreateDeploymentStrategyType, extensions.RollingUpdateDeploymentStrategyType}
			j.Type = strategyTypes[c.Rand.Intn(len(strategyTypes))]
			if j.Type != extensions.RollingUpdateDeploymentStrategyType {
				j.RollingUpdate = nil
			} else {
				rollingUpdate := extensions.RollingUpdateDeployment{}
				if c.RandBool() {
					rollingUpdate.MaxUnavailable = intstr.FromInt(int(c.Rand.Int31()))
					rollingUpdate.MaxSurge = intstr.FromInt(int(c.Rand.Int31()))
				} else {
					rollingUpdate.MaxSurge = intstr.FromString(fmt.Sprintf("%d%%", c.Rand.Int31()))
				}
				j.RollingUpdate = &rollingUpdate
			}
		},
		func(psp *extensions.PodSecurityPolicySpec, c fuzz.Continue) {
			c.FuzzNoCustom(psp) // fuzz self without calling this function again
			runAsUserRules := []extensions.RunAsUserStrategy{extensions.RunAsUserStrategyMustRunAsNonRoot, extensions.RunAsUserStrategyMustRunAs, extensions.RunAsUserStrategyRunAsAny}
			psp.RunAsUser.Rule = runAsUserRules[c.Rand.Intn(len(runAsUserRules))]
			seLinuxRules := []extensions.SELinuxStrategy{extensions.SELinuxStrategyRunAsAny, extensions.SELinuxStrategyMustRunAs}
			psp.SELinux.Rule = seLinuxRules[c.Rand.Intn(len(seLinuxRules))]
		},
		func(s *extensions.Scale, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			// TODO: Implement a fuzzer to generate valid keys, values and operators for
			// selector requirements.
			if s.Status.Selector != nil {
				s.Status.Selector = &metav1.LabelSelector{
					MatchLabels: map[string]string{
						"testlabelkey": "testlabelval",
					},
					MatchExpressions: []metav1.LabelSelectorRequirement{
						{
							Key:      "testkey",
							Operator: metav1.LabelSelectorOpIn,
							Values:   []string{"val1", "val2", "val3"},
						},
					},
				}
			}
		},
	}
}

func batchFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(j *batch.JobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			completions := int32(c.Rand.Int31())
			parallelism := int32(c.Rand.Int31())
			j.Completions = &completions
			j.Parallelism = &parallelism
			if c.Rand.Int31()%2 == 0 {
				j.ManualSelector = newBool(true)
			} else {
				j.ManualSelector = nil
			}
		},
		func(sj *batch.CronJobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(sj)
			suspend := c.RandBool()
			sj.Suspend = &suspend
			sds := int64(c.RandUint64())
			sj.StartingDeadlineSeconds = &sds
			sj.Schedule = c.RandString()
		},
		func(cp *batch.ConcurrencyPolicy, c fuzz.Continue) {
			policies := []batch.ConcurrencyPolicy{batch.AllowConcurrent, batch.ForbidConcurrent, batch.ReplaceConcurrent}
			*cp = policies[c.Rand.Intn(len(policies))]
		},
	}
}

func autoscalingFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(s *autoscaling.HorizontalPodAutoscalerSpec, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			minReplicas := int32(c.Rand.Int31())
			s.MinReplicas = &minReplicas

			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}

			targetUtilization := int32(c.RandUint64())
			s.Metrics = []autoscaling.MetricSpec{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricSource{
						MetricName:         c.RandString(),
						TargetAverageValue: randomQuantity(),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricSource{
						Name: api.ResourceCPU,
						TargetAverageUtilization: &targetUtilization,
					},
				},
			}
		},
		func(s *autoscaling.HorizontalPodAutoscalerStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			randomQuantity := func() resource.Quantity {
				var q resource.Quantity
				c.Fuzz(&q)
				// precalc the string for benchmarking purposes
				_ = q.String()
				return q
			}
			currentUtilization := int32(c.RandUint64())
			s.CurrentMetrics = []autoscaling.MetricStatus{
				{
					Type: autoscaling.PodsMetricSourceType,
					Pods: &autoscaling.PodsMetricStatus{
						MetricName:          c.RandString(),
						CurrentAverageValue: randomQuantity(),
					},
				},
				{
					Type: autoscaling.ResourceMetricSourceType,
					Resource: &autoscaling.ResourceMetricStatus{
						Name: api.ResourceCPU,
						CurrentAverageUtilization: &currentUtilization,
					},
				},
			}
		},
	}
}

func rbacFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(r *rbac.RoleRef, c fuzz.Continue) {
			c.FuzzNoCustom(r) // fuzz self without calling this function again

			// match defaulter
			if len(r.APIGroup) == 0 {
				r.APIGroup = rbac.GroupName
			}
		},
		func(r *rbac.Subject, c fuzz.Continue) {
			switch c.Int31n(3) {
			case 0:
				r.Kind = rbac.ServiceAccountKind
				r.APIGroup = ""
				c.FuzzNoCustom(&r.Name)
				c.FuzzNoCustom(&r.Namespace)
			case 1:
				r.Kind = rbac.UserKind
				r.APIGroup = rbac.GroupName
				c.FuzzNoCustom(&r.Name)
			case 2:
				r.Kind = rbac.GroupKind
				r.APIGroup = rbac.GroupName
				c.FuzzNoCustom(&r.Name)
			}
		},
	}
}

func policyFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(s *policy.PodDisruptionBudgetStatus, c fuzz.Continue) {
			c.FuzzNoCustom(s) // fuzz self without calling this function again
			s.PodDisruptionsAllowed = int32(c.Rand.Intn(2))
		},
	}
}

func certificateFuncs(t apitesting.TestingCommon) []interface{} {
	return []interface{}{
		func(obj *certificates.CertificateSigningRequestSpec, c fuzz.Continue) {
			c.FuzzNoCustom(obj) // fuzz self without calling this function again
			obj.Usages = []certificates.KeyUsage{certificates.UsageKeyEncipherment}
		},
	}
}

func FuzzerFuncs(t apitesting.TestingCommon, codecs runtimeserializer.CodecFactory) []interface{} {
	return apitesting.MergeFuzzerFuncs(t,
		apitesting.GenericFuzzerFuncs(t, codecs),
		overrideGenericFuncs(t, codecs),
		CoreFuzzerFuncs(t),
		extensionFuncs(t),
		batchFuncs(t),
		autoscalingFuncs(t),
		rbacFuncs(t),
		kubeadm.KubeadmFuzzerFuncs(t),
		policyFuncs(t),
		certificateFuncs(t),
	)
}

func newBool(val bool) *bool {
	p := new(bool)
	*p = val
	return p
}
