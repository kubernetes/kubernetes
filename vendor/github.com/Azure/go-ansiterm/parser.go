package ansiterm

import (
	"errors"
	"io/ioutil"
	"os"

	"github.com/Sirupsen/logrus"
)

var logger *logrus.Logger

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
}

func CreateParser(initialState string, evtHandler AnsiEventHandler) *AnsiParser {
	logFile := ioutil.Discard

	if isDebugEnv := os.Getenv(LogEnv); isDebugEnv == "1" {
		logFile, _ = os.Create("ansiParser.log")
	}

	logger = &logrus.Logger{
		Out:       logFile,
		Formatter: new(logrus.TextFormatter),
		Level:     logrus.InfoLevel,
	}

	parser := &AnsiParser{
		eventHandler: evtHandler,
		context:      &ansiContext{},
	}

	parser.csiEntry = csiEntryState{baseState{name: "CsiEntry", parser: parser}}
	parser.csiParam = csiParamState{baseState{name: "CsiParam", parser: parser}}
	parser.dcsEntry = dcsEntryState{baseState{name: "DcsEntry", parser: parser}}
	parser.escape = escapeState{baseState{name: "Escape", parser: parser}}
	parser.escapeIntermediate = escapeIntermediateState{baseState{name: "EscapeIntermediate", parser: parser}}
	parser.error = errorState{baseState{name: "Error", parser: parser}}
	parser.ground = groundState{baseState{name: "Ground", parser: parser}}
	parser.oscString = oscStringState{baseState{name: "OscString", parser: parser}}

	parser.stateMap = []state{
		parser.csiEntry,
		parser.csiParam,
		parser.dcsEntry,
		parser.escape,
		parser.escapeIntermediate,
		parser.error,
		parser.ground,
		parser.oscString,
	}

	parser.currState = getState(initialState, parser.stateMap)

	logger.Infof("CreateParser: parser %p", parser)
	return parser
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
		logger.Warning("newState is nil")
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
	logger.Infof("ChangeState %s --> %s", ap.currState.Name(), newState.Name())

	// Exit old state
	if err := ap.currState.Exit(); err != nil {
		logger.Infof("Exit state '%s' failed with : '%v'", ap.currState.Name(), err)
		return err
	}

	// Perform transition action
	if err := ap.currState.Transition(newState); err != nil {
		logger.Infof("Transition from '%s' to '%s' failed with: '%v'", ap.currState.Name(), newState.Name, err)
		return err
	}

	// Enter new state
	if err := newState.Enter(); err != nil {
		logger.Infof("Enter state '%s' failed with: '%v'", newState.Name(), err)
		return err
	}

	ap.currState = newState
	return nil
}
