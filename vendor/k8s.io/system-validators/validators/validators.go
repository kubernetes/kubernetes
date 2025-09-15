/*
Copyright 2016 The Kubernetes Authors.

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

package system

import (
	"fmt"
	"runtime"
)

// Validator is the interface for all validators.
type Validator interface {
	// Name is the name of the validator.
	Name() string
	// Validate is the validate function.
	Validate(SysSpec) ([]error, []error)
}

// Reporter is the interface for the reporters for the validators.
type Reporter interface {
	// Report reports the results of the system verification
	Report(string, string, ValidationResultType) error
}

// Validate uses validators to validate the system and returns a warning or error.
func Validate(spec SysSpec, validators []Validator) ([]error, []error) {
	var errs []error
	var warns []error

	for _, v := range validators {
		fmt.Printf("Validating %s...\n", v.Name())
		warn, err := v.Validate(spec)
		if len(err) != 0 {
			errs = append(errs, err...)
		}
		if len(warn) != 0 {
			warns = append(warns, warn...)
		}
	}
	return warns, errs
}

// ValidateSpec uses all default validators to validate the system and writes to stdout.
func ValidateSpec(spec SysSpec, containerRuntime string) ([]error, []error) {
	// OS-level validators.
	var osValidators = []Validator{
		&OSValidator{Reporter: DefaultReporter},
		&KernelValidator{Reporter: DefaultReporter},
	}

	// Docker-specific validators.
	var dockerValidators = []Validator{
		&DockerValidator{Reporter: DefaultReporter},
	}

	validators := osValidators
	switch containerRuntime {
	case "docker":
		validators = append(validators, dockerValidators...)
	}

	// Linux-specific validators.
	if runtime.GOOS == "linux" {
		validators = append(validators,
			&CgroupsValidator{Reporter: DefaultReporter},
			&packageValidator{reporter: DefaultReporter},
		)
	}

	return Validate(spec, validators)
}
