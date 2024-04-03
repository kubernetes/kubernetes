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

package execution

import "fmt"

func (v *Vars) Print(i ...any) {
	_, _ = fmt.Fprint(v.Out, i...)
}

func (v *Vars) Println(i ...any) {
	v.Print(fmt.Sprintln(i...))
}

func (v *Vars) Printf(format string, i ...any) {
	v.Print(fmt.Sprintf(format, i...))
}
