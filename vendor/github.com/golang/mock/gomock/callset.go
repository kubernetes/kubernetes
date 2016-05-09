// Copyright 2011 Google Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package gomock

// callSet represents a set of expected calls, indexed by receiver and method
// name.
type callSet map[interface{}]map[string][]*Call

// Add adds a new expected call.
func (cs callSet) Add(call *Call) {
	methodMap, ok := cs[call.receiver]
	if !ok {
		methodMap = make(map[string][]*Call)
		cs[call.receiver] = methodMap
	}
	methodMap[call.method] = append(methodMap[call.method], call)
}

// Remove removes an expected call.
func (cs callSet) Remove(call *Call) {
	methodMap, ok := cs[call.receiver]
	if !ok {
		return
	}
	sl := methodMap[call.method]
	for i, c := range sl {
		if c == call {
			// quick removal; we don't need to maintain call order
			if len(sl) > 1 {
				sl[i] = sl[len(sl)-1]
			}
			methodMap[call.method] = sl[:len(sl)-1]
			break
		}
	}
}

// FindMatch searches for a matching call. Returns nil if no call matched.
func (cs callSet) FindMatch(receiver interface{}, method string, args []interface{}) *Call {
	methodMap, ok := cs[receiver]
	if !ok {
		return nil
	}
	calls, ok := methodMap[method]
	if !ok {
		return nil
	}

	// Search through the unordered set of calls expected on a method on a
	// receiver.
	for _, call := range calls {
		// A call should not normally still be here if exhausted,
		// but it can happen if, for instance, .Times(0) was used.
		// Pretend the call doesn't match.
		if call.exhausted() {
			continue
		}
		if call.matches(args) {
			return call
		}
	}

	return nil
}
