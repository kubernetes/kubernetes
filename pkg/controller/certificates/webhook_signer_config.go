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

package certificates

import (
	"fmt"
	"os"
	"time"

	"github.com/golang/glog"

	"k8s.io/apimachinery/pkg/util/yaml"
)

const (
	defaultRetryBackoff = time.Duration(500) * time.Millisecond
	useDefault          = time.Duration(0) // missing/zero value indicates to use default
)

type webhookSignerConfig struct {
	KubeConfigFile string        `json:"kubeConfigFile"`
	RetryBackoff   time.Duration `json:"retryBackoff"` // in milliseconds
}

func NewWebhookSignerFromConfigFile(configFile string) (*WebhookSigner, error) {
	var config webhookSignerConfig

	f, err := os.Open(configFile)
	if err != nil {
		return nil, fmt.Errorf("could not open file %s: %s", configFile, err)
	}
	defer f.Close()

	d := yaml.NewYAMLOrJSONDecoder(f, 4096)
	err = d.Decode(&config)
	if err != nil {
		return nil, fmt.Errorf("could not decode file %s: %s", configFile, err)
	}

	if config.RetryBackoff == useDefault {
		glog.V(1).Infof("webhook signer using default retry backoff: %s", defaultRetryBackoff)
		config.RetryBackoff = defaultRetryBackoff
	} else {
		// Unmarshalling gives us nanoseconds, so scale to milliseconds
		config.RetryBackoff *= time.Millisecond
	}

	return NewWebhookSigner(config.KubeConfigFile, config.RetryBackoff)
}
