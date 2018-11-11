package ansiterm

import (
	"errors"
	"log"
	"os"
)

type AnsiParser struct {
	currState          state
	eventHandler       AnsiEventHandler
	context            *ansiContext
	csiEntry           state
	csiParam           state
	dcsEntry           state
	escape             state
	escapeIntermediate state
	error              state
	ground             state
	oscString          state
	stateMap           []state

	logf func(string, ...interface{})
}

type Option func(*AnsiParser)

func WithLogf(f func(string, ...interface{})) Option {
	return func(ap *AnsiParser) {
		ap.logf = f
	}
}

func CreateParser(initialState string, evtHandler AnsiEventHandler, opts ...Option) *AnsiParser {
	ap := &AnsiParser{
		eventHandler: evtHandler,
		context:      &ansiContext{},
	}
	for _, o := range opts {
		o(ap)
	}

	if isDebugEnv := os.Getenv(LogEnv); isDebugEnv == "1" {
		logFile, _ := os.Create("ansiParser.log")
		logger := log.New(logFile, "", log.LstdFlags)
		if ap.logf != nil {
			l := ap.logf
			ap.logf = func(s string, v ...interface{}) {
				l(s, v...)
				logger.Printf(s, v...)
			}
		} else {
			ap.logf = logger.Printf
		}
	}

	if ap.logf == nil {
		ap.logf = func(string, ...interface{}) {}
	}

	ap.csiEntry = csiEntryState{baseState{name: "CsiEntry", parser: ap}}
	ap.csiParam = csiParamState{baseState{name: "CsiParam", parser: ap}}
	ap.dcsEntry = dcsEntryState{baseState{name: "DcsEntry", parser: ap}}
	ap.escape = escapeState{baseState{name: "Escape", parser: ap}}
	ap.escapeIntermediate = escapeIntermediateState{baseState{name: "EscapeIntermediate", parser: ap}}
	ap.error = errorState{baseState{name: "Error", parser: ap}}
	ap.ground = groundState{baseState{name: "Ground", parser: ap}}
	ap.oscString = oscStringState{baseState{name: "OscString", parser: ap}}

	ap.stateMap = []state{
		ap.csiEntry,
		ap.csiParam,
		ap.dcsEntry,
		ap.escape,
		ap.escapeIntermediate,
		ap.error,
		ap.ground,
		ap.oscString,
	}

	ap.currState = getState(initialState, ap.stateMap)

	ap.logf("CreateParser: parser %p", ap)
	return ap
}

func getState(name string, states []state) state {
	for _, el := range states {
		if el.Name() == name {
			return el
		}
	}

	return nil
}

func (ap *AnsiParser) Parse(bytes []byte) (int, error) {
	for i, b := range bytes {
		if err := ap.handle(b); err != nil {
			return i, err
		}
	}

	return len(bytes), ap.eventHandler.Flush()
}

func (ap *AnsiParser) handle(b byte) error {
	ap.context.currentChar = b
	newState, err := ap.currState.Handle(b)
	if err != nil {
		return err
	}

	if newState == nil {
		ap.logf("WARNING: newState is nil")
		return errors.New("New state of 'nil' is invalid.")
	}

	if newState != ap.currState {
		if err := ap.changeState(newState); err != nil {
			return err
		}
	}

	return nil
}

func (ap *AnsiParser) changeState(newState state) error {
	ap.logf("ChangeState %s --> %s", ap.currState.Name(), newState.Name())

	// Exit old state
	if err := ap.currState.Exit(); err != nil {
		ap.logf("Exit state '%s' failed with : '%v'", ap.currState.Name(), err)
		return err
	}

	// Perform transition action
	if err := ap.currState.Transition(newState); err != nil {
		ap.logf("Transition from '%s' to '%s' failed with: '%v'", ap.currState.Name(), newState.Name, err)
		return err
	}

	// Enter new state
	if err := newState.Enter(); err != nil {
		ap.logf("Enter state '%s' failed with: '%v'", newState.Name(), err)
		return err
	}

	ap.currState = newState
	return nil
}
