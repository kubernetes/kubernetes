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
	"github.com/golang/glog"
	"k8s.io/apimachinery/pkg/util/errors"
)

// Validator is the interface for all validators.
type Validator interface {
	// Name is the name of the validator.
	Name() string
	// Validate is the validate function.
	Validate(SysSpec) error
}

// Reporter is the interface for the reporters for the validators.
type Reporter interface {
	// Report reports the results of the system verification
	Report(string, string, ValidationResultType) error
}

// Validate uses all validators to validate the system.
func Validate(spec SysSpec, report Reporter) error {
	var errs []error
	// validators are all the validators.
	var validators = []Validator{
		&OSValidator{Reporter: report},
		&KernelValidator{Reporter: report},
		&CgroupsValidator{Reporter: report},
		&DockerValidator{Reporter: report},
	}

	for _, v := range validators {
		glog.Infof("Validating %s...", v.Name())
		errs = append(errs, v.Validate(spec))
	}
	return errors.NewAggregate(errs)
}

// ValidateDefault uses all default validators to validate the system and writes to stdout.
func ValidateDefault() error {
	return Validate(DefaultSysSpec, DefaultReporter)
}
