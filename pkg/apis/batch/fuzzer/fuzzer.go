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

package fuzzer

import (
	"math"

	fuzz "github.com/google/gofuzz"
	runtimeserializer "k8s.io/apimachinery/pkg/runtime/serializer"
	"k8s.io/kubernetes/pkg/apis/batch"
	api "k8s.io/kubernetes/pkg/apis/core"
	"k8s.io/utils/pointer"
)

// Funcs returns the fuzzer functions for the batch api group.
var Funcs = func(codecs runtimeserializer.CodecFactory) []interface{} {
	return []interface{}{
		func(j *batch.Job, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again

			// match defaulting
			if len(j.Labels) == 0 {
				j.Labels = j.Spec.Template.Labels
			}
		},
		func(j *batch.JobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(j) // fuzz self without calling this function again
			completions := int32(c.Rand.Int31())
			parallelism := int32(c.Rand.Int31())
			backoffLimit := int32(c.Rand.Int31())
			j.Completions = &completions
			j.Parallelism = &parallelism
			j.BackoffLimit = &backoffLimit
			j.ManualSelector = pointer.Bool(c.RandBool())
			mode := batch.NonIndexedCompletion
			if c.RandBool() {
				mode = batch.IndexedCompletion
				j.BackoffLimitPerIndex = pointer.Int32(c.Rand.Int31())
				j.MaxFailedIndexes = pointer.Int32(c.Rand.Int31())
			}
			if c.RandBool() {
				j.BackoffLimit = pointer.Int32(math.MaxInt32)
			}
			j.CompletionMode = &mode
			// We're fuzzing the internal JobSpec type, not the v1 type, so we don't
			// need to fuzz the nil value.
			j.Suspend = pointer.Bool(c.RandBool())
			podReplacementPolicy := batch.TerminatingOrFailed
			if c.RandBool() {
				podReplacementPolicy = batch.Failed
			}
			j.PodReplacementPolicy = &podReplacementPolicy
			if c.RandBool() {
				c.Fuzz(j.ManagedBy)
			}
		},
		func(sj *batch.CronJobSpec, c fuzz.Continue) {
			c.FuzzNoCustom(sj)
			suspend := c.RandBool()
			sj.Suspend = &suspend
			sds := int64(c.RandUint64())
			sj.StartingDeadlineSeconds = &sds
			sj.Schedule = c.RandString()
			successfulJobsHistoryLimit := int32(c.Rand.Int31())
			sj.SuccessfulJobsHistoryLimit = &successfulJobsHistoryLimit
			failedJobsHistoryLimit := int32(c.Rand.Int31())
			sj.FailedJobsHistoryLimit = &failedJobsHistoryLimit
		},
		func(cp *batch.ConcurrencyPolicy, c fuzz.Continue) {
			policies := []batch.ConcurrencyPolicy{batch.AllowConcurrent, batch.ForbidConcurrent, batch.ReplaceConcurrent}
			*cp = policies[c.Rand.Intn(len(policies))]
		},
		func(p *batch.PodFailurePolicyOnPodConditionsPattern, c fuzz.Continue) {
			c.FuzzNoCustom(p)
			if p.Status == "" {
				p.Status = api.ConditionTrue
			}
		},
	}
}
