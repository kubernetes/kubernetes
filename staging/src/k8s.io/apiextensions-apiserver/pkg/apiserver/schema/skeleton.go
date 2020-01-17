/*
Copyright 2019 The Kubernetes Authors.

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

package schema

// StripDefaults returns a copy without defaults.
func (s *Structural) StripDefaults() *Structural {
	s = s.DeepCopy()
	v := Visitor{
		Structural: func(s *Structural) bool {
			changed := false
			if s.Default.Object != nil {
				s.Default.Object = nil
				changed = true
			}
			return changed
		},
	}
	v.Visit(s)
	return s
}

// StripValueValidations returns a copy without value validations.
func (s *Structural) StripValueValidations() *Structural {
	s = s.DeepCopy()
	v := Visitor{
		Structural: func(s *Structural) bool {
			changed := false
			if s.ValueValidation != nil {
				s.ValueValidation = nil
				changed = true
			}
			return changed
		},
	}
	v.Visit(s)
	return s
}
