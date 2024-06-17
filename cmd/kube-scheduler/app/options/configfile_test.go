/*
Copyright 2023 The Kubernetes Authors.

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

package options

import (
	"context"
	"fmt"
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/stretchr/testify/assert"
	"k8s.io/klog/v2"
	"k8s.io/kubernetes/pkg/scheduler/apis/config"
)

const (
	apiVersionMissing = "'apiVersion' is missing"
	apiVersionTooOld  = "no kind \"KubeSchedulerConfiguration\" is registered for" +
		" version \"kubescheduler.config.k8s.io/v1alpha1\""

	// schedulerConfigMinimalCorrect is the minimal
	// correct scheduler config
	schedulerConfigMinimalCorrect = `
apiVersion: kubescheduler.config.k8s.io/v1
kind: KubeSchedulerConfiguration`

	// schedulerConfigDecodeErr is the scheduler config
	// which throws decoding error when we try to load it
	schedulerConfigDecodeErr = `
kind: KubeSchedulerConfiguration`

	// schedulerConfigVersionTooOld is the scheduler config
	// which throws error because the config version 'v1alpha1'
	// is too old
	schedulerConfigVersionTooOld = `
apiVersion: kubescheduler.config.k8s.io/v1alpha1
kind: KubeSchedulerConfiguration
`
)

func TestLoadConfigFromFile(t *testing.T) {
	tmpDir, err := os.MkdirTemp("", "scheduler-configs")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmpDir)

	correctConfigFile := filepath.Join(tmpDir, "correct_config.yaml")
	if err := os.WriteFile(correctConfigFile,
		[]byte(schedulerConfigMinimalCorrect),
		os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	decodeErrConfigFile := filepath.Join(tmpDir, "decode_err_no_version_config.yaml")
	if err := os.WriteFile(decodeErrConfigFile,
		[]byte(schedulerConfigDecodeErr),
		os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	versionTooOldConfigFile := filepath.Join(tmpDir, "version_too_old_config.yaml")
	if err := os.WriteFile(versionTooOldConfigFile,
		[]byte(schedulerConfigVersionTooOld),
		os.FileMode(0600)); err != nil {
		t.Fatal(err)
	}

	tests := []struct {
		name           string
		path           string
		expectedErr    error
		expectedConfig *config.KubeSchedulerConfiguration
	}{
		{
			name:           "Empty scheduler config file path",
			path:           "",
			expectedErr:    syscall.Errno(syscall.ENOENT),
			expectedConfig: nil,
		},
		{
			name:           "Correct scheduler config",
			path:           correctConfigFile,
			expectedErr:    nil,
			expectedConfig: &config.KubeSchedulerConfiguration{},
		},
		{
			name:           "Scheduler config with decode error",
			path:           decodeErrConfigFile,
			expectedErr:    fmt.Errorf(apiVersionMissing),
			expectedConfig: nil,
		},
		{
			name:           "Scheduler config version too old",
			path:           versionTooOldConfigFile,
			expectedErr:    fmt.Errorf(apiVersionTooOld),
			expectedConfig: nil,
		},
	}

	logger := klog.FromContext(context.Background())

	for i, test := range tests {
		t.Run(fmt.Sprintf("case_%d: %s", i, test.name), func(t *testing.T) {
			cfg, err := LoadConfigFromFile(logger, test.path)
			if test.expectedConfig == nil {
				assert.Nil(t, cfg)
			} else {
				assert.NotNil(t, cfg)
			}

			if test.expectedErr == nil {
				assert.NoError(t, err)
			} else {
				assert.ErrorContains(t, err, test.expectedErr.Error())
			}
		})

	}
}
