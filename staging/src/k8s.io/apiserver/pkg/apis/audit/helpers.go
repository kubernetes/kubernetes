/*
Copyright 2017 The Kubernetes Authors.

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

package audit

func ordLevel(l Level) int {
	switch l {
	case LevelMetadata:
		return 1
	case LevelRequest:
		return 2
	case LevelRequestResponse:
		return 3
	default:
		return 0
	}
}

func (a Level) Less(b Level) bool {
	return ordLevel(a) < ordLevel(b)
}

func (a Level) GreaterOrEqual(b Level) bool {
	return ordLevel(a) >= ordLevel(b)
}
