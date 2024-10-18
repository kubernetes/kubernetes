/*
Copyright 2024 The Kubernetes Authors.

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

package library

import "github.com/google/cel-go/cel"

func AllowMultipleMessageExpression() cel.EnvOption {
	return cel.Lib(allowMultipleMessageExpressionLib)
}

var allowMultipleMessageExpressionLib = &allowMultipleMessageExpression{}

var AllowMultipleMessageExpressionName = "AllowMultipleMessageExpression"

type allowMultipleMessageExpression struct{}

func (*allowMultipleMessageExpression) LibraryName() string {
	return AllowMultipleMessageExpressionName
}

func (*allowMultipleMessageExpression) CompileOptions() []cel.EnvOption {
	return nil
}

func (*allowMultipleMessageExpression) ProgramOptions() []cel.ProgramOption {
	return nil
}
