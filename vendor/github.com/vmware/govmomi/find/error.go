/*
Copyright (c) 2015 VMware, Inc. All Rights Reserved.

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

package find

import "fmt"

type NotFoundError struct {
	kind string
	path string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("%s '%s' not found", e.kind, e.path)
}

type MultipleFoundError struct {
	kind string
	path string
}

func (e *MultipleFoundError) Error() string {
	return fmt.Sprintf("path '%s' resolves to multiple %ss", e.path, e.kind)
}

type DefaultNotFoundError struct {
	kind string
}

func (e *DefaultNotFoundError) Error() string {
	return fmt.Sprintf("no default %s found", e.kind)
}

type DefaultMultipleFoundError struct {
	kind string
}

func (e DefaultMultipleFoundError) Error() string {
	return fmt.Sprintf("default %s resolves to multiple instances, please specify", e.kind)
}

func toDefaultError(err error) error {
	switch e := err.(type) {
	case *NotFoundError:
		return &DefaultNotFoundError{e.kind}
	case *MultipleFoundError:
		return &DefaultMultipleFoundError{e.kind}
	default:
		return err
	}
}
