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

package remote

import (
	"fmt"
)

var _ Runner = (*SSHRunner)(nil)

type SSHRunner struct {
	cfg Config
}

func (s *SSHRunner) StartTests(suite TestSuite, archivePath string, results chan *TestResult) (numTests int) {
	for _, host := range s.cfg.Hosts {
		fmt.Printf("Initializing e2e tests using host %s.\n", host)
		numTests++
		go func(host string, junitFileName string) {
			output, exitOk, err := RunRemote(RunRemoteConfig{
				Suite:          suite,
				Archive:        archivePath,
				Host:           host,
				Cleanup:        s.cfg.Cleanup,
				ImageDesc:      "",
				JunitFileName:  junitFileName,
				TestArgs:       s.cfg.TestArgs,
				GinkgoArgs:     s.cfg.GinkgoFlags,
				SystemSpecName: s.cfg.SystemSpecName,
				ExtraEnvs:      s.cfg.ExtraEnvs,
				RuntimeConfig:  s.cfg.RuntimeConfig,
			})
			results <- &TestResult{
				Output: output,
				Err:    err,
				Host:   host,
				ExitOK: exitOk,
			}
		}(host, host)
	}
	return
}

func NewSSHRunner(cfg Config) Runner {
	return &SSHRunner{
		cfg: cfg,
	}
}

func (s *SSHRunner) Validate() error {
	if len(s.cfg.Hosts) == 0 {
		return fmt.Errorf("must specify --hosts when running ssh")
	}
	if s.cfg.ImageConfigFile != "" {
		return fmt.Errorf("must not specify --image-config-file when running ssh")
	}
	if len(s.cfg.Images) > 0 {
		return fmt.Errorf("must not specify --images when running ssh")
	}
	return nil
}
