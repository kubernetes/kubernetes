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

package dynamic

import (
	"time"

	bufferedplugin "k8s.io/apiserver/plugin/pkg/audit/buffered"
)

const (
	// Default configuration values for ModeBatch when applied to a dynamic plugin
	defaultBatchBufferSize    = 5000             // Buffer up to 5000 events before starting discarding.
	defaultBatchMaxSize       = 400              // Only send up to 400 events at a time.
	defaultBatchMaxWait       = 30 * time.Second // Send events at least twice a minute.
	defaultBatchThrottleQPS   = 10               // Limit the send rate by 10 QPS.
	defaultBatchThrottleBurst = 15               // Allow up to 15 QPS burst.
)

// NewDefaultWebhookBatchConfig returns new Batch Config objects populated by default values
// for dynamic webhooks
func NewDefaultWebhookBatchConfig() *bufferedplugin.BatchConfig {
	return &bufferedplugin.BatchConfig{
		BufferSize:     defaultBatchBufferSize,
		MaxBatchSize:   defaultBatchMaxSize,
		MaxBatchWait:   defaultBatchMaxWait,
		ThrottleEnable: true,
		ThrottleQPS:    defaultBatchThrottleQPS,
		ThrottleBurst:  defaultBatchThrottleBurst,
		AsyncDelegate:  true,
	}
}
