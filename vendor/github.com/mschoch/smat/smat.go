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

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"log"
	"math/rand"
)

// Logger is a configurable logger used by this package
// by default output is discarded
var Logger = log.New(ioutil.Discard, "smat ", log.LstdFlags)

// Context is a container for any user state
type Context interface{}

// State is a function which describes which action to perform in the event
// that a particular byte is seen
type State func(next byte) ActionID

// PercentAction describes the frequency with which an action should occur
// for example: Action{Percent:10, Action:DonateMoney} means that 10% of
// the time you should donate money.
type PercentAction struct {
	Percent int
	Action  ActionID
}

// Action is any function which returns the next state to transition to
// it can optionally mutate the provided context object
// if any error occurs, it may return an error which will abort execution
type Action func(Context) (State, error)

// ActionID is a unique identifier for an action
type ActionID int

// NopAction does nothing and simply continues to the next input
var NopAction ActionID = -1

// ActionMap is a mapping form ActionID to Action
type ActionMap map[ActionID]Action

func (a ActionMap) findSetupTeardown(setup, teardown ActionID) (Action, Action, error) {
	setupFunc, ok := a[setup]
	if !ok {
		return nil, nil, ErrSetupMissing
	}
	teardownFunc, ok := a[teardown]
	if !ok {
		return nil, nil, ErrTeardownMissing
	}
	return setupFunc, teardownFunc, nil
}

// Fuzz runs the fuzzing state machine with the provided context
// first, the setup action is executed unconditionally
// the start state is determined by this action
// actionMap is a lookup table for all actions
// the data byte slice determines all future state transitions
// finally, the teardown action is executed unconditionally for cleanup
func Fuzz(ctx Context, setup, teardown ActionID, actionMap ActionMap, data []byte) int {
	reader := bytes.NewReader(data)
	err := runReader(ctx, setup, teardown, actionMap, reader, nil)
	if err != nil {
		panic(err)
	}
	return 1
}

// Longevity runs the state machine with the provided context
// first, the setup action is executed unconditionally
// the start state is determined by this action
// actionMap is a lookup table for all actions
// random bytes are generated to determine all future state transitions
// finally, the teardown action is executed unconditionally for cleanup
func Longevity(ctx Context, setup, teardown ActionID, actionMap ActionMap, seed int64, closeChan chan struct{}) error {
	source := rand.NewSource(seed)
	return runReader(ctx, setup, teardown, actionMap, rand.New(source), closeChan)
}

var (
	// ErrSetupMissing is returned when the setup action cannot be found
	ErrSetupMissing = fmt.Errorf("setup action missing")
	// ErrTeardownMissing is returned when the teardown action cannot be found
	ErrTeardownMissing = fmt.Errorf("teardown action missing")
	// ErrClosed is returned when the closeChan was closed to cancel the op
	ErrClosed = fmt.Errorf("closed")
	// ErrActionNotPossible is returned when an action is encountered in a
	// FuzzCase that is not possible in the current state
	ErrActionNotPossible = fmt.Errorf("action not possible in state")
)

func runReader(ctx Context, setup, teardown ActionID, actionMap ActionMap, r io.Reader, closeChan chan struct{}) error {
	setupFunc, teardownFunc, err := actionMap.findSetupTeardown(setup, teardown)
	if err != nil {
		return err
	}
	Logger.Printf("invoking setup action")
	state, err := setupFunc(ctx)
	if err != nil {
		return err
	}
	defer func() {
		Logger.Printf("invoking teardown action")
		_, _ = teardownFunc(ctx)
	}()

	reader := bufio.NewReader(r)
	for next, err := reader.ReadByte(); err == nil; next, err = reader.ReadByte() {
		select {
		case <-closeChan:
			return ErrClosed
		default:
			actionID := state(next)
			action, ok := actionMap[actionID]
			if !ok {
				Logger.Printf("no such action defined, continuing")
				continue
			}
			Logger.Printf("invoking action - %d", actionID)
			state, err = action(ctx)
			if err != nil {
				Logger.Printf("it was action %d that returned err %v", actionID, err)
				return err
			}
		}
	}
	return err
}

// PercentExecute interprets the next byte as a random value and normalizes it
// to values 0-99, it then looks to see which action should be execued based
// on the action distributions
func PercentExecute(next byte, pas ...PercentAction) ActionID {
	percent := int(99 * int(next) / 255)

	sofar := 0
	for _, pa := range pas {
		sofar = sofar + pa.Percent
		if percent < sofar {
			return pa.Action
		}

	}
	return NopAction
}
