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

type Runner interface {
	Validate() error
	StartTests(suite TestSuite, archivePath string, results chan *TestResult) (numTests int)
}

type Config struct {
	InstanceNamePrefix string
	ImageConfigFile    string
	Images             []string
	ImageConfigDir     string
	GinkgoFlags        string
	DeleteInstances    bool
	Cleanup            bool
	TestArgs           string
	ExtraEnvs          string
	RuntimeConfig      string
	SystemSpecName     string
	Hosts              []string
}

// TestResult contains some information about the test results.
type TestResult struct {
	Output string
	Err    error
	Host   string
	ExitOK bool
}
