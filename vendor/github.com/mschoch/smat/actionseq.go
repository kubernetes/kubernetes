//  Copyright (c) 2016 Marty Schoch

//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the
//  License. You may obtain a copy of the License at
//    http://www.apache.org/licenses/LICENSE-2.0
//  Unless required by applicable law or agreed to in writing,
//  software distributed under the License is distributed on an "AS
//  IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
//  express or implied. See the License for the specific language
//  governing permissions and limitations under the License.

package smat

// ActionSeq represents a sequence of actions, used for populating a corpus
// of byte sequences for the corresponding fuzz tests
type ActionSeq []ActionID

// ByteEncoding runs the FSM to produce a byte sequence to trigger the
// desired action
func (a ActionSeq) ByteEncoding(ctx Context, setup, teardown ActionID, actionMap ActionMap) ([]byte, error) {
	setupFunc, teardownFunc, err := actionMap.findSetupTeardown(setup, teardown)
	if err != nil {
		return nil, err
	}
	state, err := setupFunc(ctx)
	if err != nil {
		return nil, err
	}
	defer func() {
		_, _ = teardownFunc(ctx)
	}()

	var rv []byte
	for _, actionID := range a {
		b, err := probeStateForAction(state, actionID)
		if err != nil {
			return nil, err
		}
		rv = append(rv, b)
		action, ok := actionMap[actionID]
		if !ok {
			continue
		}
		state, err = action(ctx)
		if err != nil {
			return nil, err
		}
	}
	return rv, nil
}

func probeStateForAction(state State, actionID ActionID) (byte, error) {
	for i := 0; i < 256; i++ {
		nextActionID := state(byte(i))
		if nextActionID == actionID {
			return byte(i), nil
		}
	}
	return 0, ErrActionNotPossible
}
