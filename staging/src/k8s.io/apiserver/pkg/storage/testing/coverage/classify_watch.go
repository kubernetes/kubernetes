/*
Copyright The Kubernetes Authors.

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

package coverage

import (
	"k8s.io/apiserver/pkg/storage"
)

// ClassifyWatch classifies a storage.Interface.Watch call.
func ClassifyWatch(opts storage.ListOptions, err error) State {
	selector, _ := classifySelectionPredicate(opts.Predicate)
	return State{
		Verb:                 VerbWatch,
		ResourceVersion:      classifyResourceVersionMode(opts.ResourceVersion),
		ResourceVersionMatch: classifyResourceVersionMatchMode(opts.ResourceVersionMatch),
		Recursive:            opts.Recursive,
		Selector:             selector,
		AllowWatchBookmarks:  opts.Predicate.AllowWatchBookmarks,
		SendInitialEvents:    classifySendInitialEvents(opts.SendInitialEvents),
		Outcome:              classifyTerminalErrorOutcome(err),
	}
}
