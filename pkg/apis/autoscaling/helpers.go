/*
Copyright 2020 The Kubernetes Authors.

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

package autoscaling

// DropRoundTripHorizontalPodAutoscalerAnnotations removes any annotations used to serialize round-tripped fields from later API versions,
// and returns false if no changes were made and the original input object was returned.
// It should always be called when converting internal -> external versions, prior
// to setting any of the custom annotations:
//
//     annotations, copiedAnnotations := DropRoundTripHorizontalPodAutoscalerAnnotations(externalObj.Annotations)
//     externalObj.Annotations = annotations
//
//     if internal.SomeField != nil {
//       if !copiedAnnotations {
//         externalObj.Annotations = DeepCopyStringMap(externalObj.Annotations)
//         copiedAnnotations = true
//       }
//       externalObj.Annotations[...] = json.Marshal(...)
//     }
func DropRoundTripHorizontalPodAutoscalerAnnotations(in map[string]string) (out map[string]string, copied bool) {
	_, hasMetricsSpecs := in[MetricSpecsAnnotation]
	_, hasBehaviorSpecs := in[BehaviorSpecsAnnotation]
	_, hasMetricsStatuses := in[MetricStatusesAnnotation]
	_, hasConditions := in[HorizontalPodAutoscalerConditionsAnnotation]
	if hasMetricsSpecs || hasBehaviorSpecs || hasMetricsStatuses || hasConditions {
		out = DeepCopyStringMap(in)
		delete(out, MetricSpecsAnnotation)
		delete(out, BehaviorSpecsAnnotation)
		delete(out, MetricStatusesAnnotation)
		delete(out, HorizontalPodAutoscalerConditionsAnnotation)
		return out, true
	}
	return in, false
}

// DeepCopyStringMap returns a copy of the input map.
// If input is nil, an empty map is returned.
func DeepCopyStringMap(in map[string]string) map[string]string {
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}
