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

package bandwidth

import (
	"errors"
	"fmt"

	"k8s.io/apimachinery/pkg/api/resource"
)

const (
	directionIngress = "ingress"
	directionEgress  = "egress"
)

var (
	minBandwidth = resource.MustParse("1k")
	maxBandwidth = resource.MustParse("1P")

	minBurst = resource.MustParse("1k")
	maxBurst = resource.MustParse("4G") // comply with bandwidth plugin limit
)

func validateBandwidthIsReasonable(rsrc *resource.Quantity) error {
	if rsrc.Value() < minBandwidth.Value() {
		return fmt.Errorf("bandwidth is unreasonably small (< 1kbit)")
	}
	if rsrc.Value() > maxBandwidth.Value() {
		return fmt.Errorf("bandwidth is unreasonably large (> 1Pbit)")
	}
	return nil
}

func validateBurstIsReasonable(rsrc *resource.Quantity) error {
	if rsrc.Value() < minBurst.Value() {
		return fmt.Errorf("burst is unreasonably small (< 1kbit)")
	}
	if rsrc.Value() > maxBurst.Value() {
		return fmt.Errorf("burst is unreasonably large (> 4GB)")
	}
	return nil

}

// Limit provides a mapping to cniBandwidthEntry.
// Rate: The speed knob.
// Burst: Also known as buffer or maxburst. Size of the bucket.
// See tc(8) for details.
type Limit struct {
	Rate, Burst *resource.Quantity
}

func extractPodBandwidthResources(direction string, podAnnotations map[string]string) (limit *Limit, err error) {
	if podAnnotations == nil {
		return
	}

	limit = new(Limit)
	prefix := fmt.Sprintf("kubernetes.io/%s-", direction)

	str, rateFound := podAnnotations[prefix+"bandwidth"]
	if rateFound {
		rateValue, err := resource.ParseQuantity(str)
		if err != nil {
			return nil, err
		}
		if err := validateBandwidthIsReasonable(&rateValue); err != nil {
			return nil, err
		}
		limit.Rate = &rateValue
	}

	str, burstFound := podAnnotations[prefix+"burst"]
	if burstFound {
		if !rateFound {
			return nil, errors.New("rate must be set if burst is set")
		}
		burstValue, err := resource.ParseQuantity(str)
		if err != nil {
			return nil, err
		}
		if err := validateBurstIsReasonable(&burstValue); err != nil {
			return nil, err
		}
		limit.Burst = &burstValue
	}

	if !rateFound {
		limit = nil
	}

	return
}

// ExtractPodBandwidthResources extracts the ingress and egress from the given pod annotations
func ExtractPodBandwidthResources(podAnnotations map[string]string) (ingress, egress *Limit, err error) {
	if podAnnotations == nil {
		return nil, nil, nil
	}
	ingress, err = extractPodBandwidthResources(directionIngress, podAnnotations)
	if err != nil {
		return nil, nil, err
	}
	egress, err = extractPodBandwidthResources(directionEgress, podAnnotations)
	if err != nil {
		return nil, nil, err
	}

	return ingress, egress, nil
}
