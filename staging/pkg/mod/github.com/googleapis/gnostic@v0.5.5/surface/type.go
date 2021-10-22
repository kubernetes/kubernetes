// Copyright 2017 Google LLC. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package surface_v1

func (t *Type) addField(f *Field) {
	t.Fields = append(t.Fields, f)
}

func (s *Type) HasFieldWithName(name string) bool {
	return s.FieldWithName(name) != nil
}

func (s *Type) FieldWithName(name string) *Field {
	if s == nil || s.Fields == nil || name == "" {
		return nil
	}
	// Compares Go specific field names.
	for _, f := range s.Fields {
		if f.FieldName == name {
			return f
		}
	}

	// Compares names as specified in the OpenAPI description.
	for _, f := range s.Fields {
		if f.Name == name {
			return f
		}
	}

	return nil
}

func (s *Type) HasFieldWithPosition(position Position) bool {
	return s.FieldWithPosition(position) != nil
}

func (s *Type) FieldWithPosition(position Position) *Field {
	if s == nil || s.Fields == nil {
		return nil
	}
	for _, f := range s.Fields {
		if f.Position == position {
			return f
		}
	}
	return nil
}
