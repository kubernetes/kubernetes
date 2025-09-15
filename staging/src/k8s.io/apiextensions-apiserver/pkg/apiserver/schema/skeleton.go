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

// StripNullable returns a copy without nullable.
func (s *Structural) StripNullable() *Structural {
	s = s.DeepCopy()
	v := Visitor{
		Structural: func(s *Structural) bool {
			changed := s.Nullable
			s.Nullable = false
			return changed
		},
	}
	v.Visit(s)
	return s
}
