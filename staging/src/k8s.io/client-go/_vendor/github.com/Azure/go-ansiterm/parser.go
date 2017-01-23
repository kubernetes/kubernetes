package ansiterm

import (
	"errors"
	"fmt"
	"io/ioutil"
	"os"

	"github.com/Sirupsen/logrus"
)

var logger *logrus.Logger

type AnsiParser struct {
	currState          State
	eventHandler       AnsiEventHandler
	context            *AnsiContext
	CsiEntry           State
	CsiParam           State
	DcsEntry           State
	Escape             State
	EscapeIntermediate State
	Error              State
	Ground             State
	OscString          State
	stateMap           []State
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
		context:      &AnsiContext{},
	}

	parser.CsiEntry = CsiEntryState{BaseState{name: "CsiEntry", parser: parser}}
	parser.CsiParam = CsiParamState{BaseState{name: "CsiParam", parser: parser}}
	parser.DcsEntry = DcsEntryState{BaseState{name: "DcsEntry", parser: parser}}
	parser.Escape = EscapeState{BaseState{name: "Escape", parser: parser}}
	parser.EscapeIntermediate = EscapeIntermediateState{BaseState{name: "EscapeIntermediate", parser: parser}}
	parser.Error = ErrorState{BaseState{name: "Error", parser: parser}}
	parser.Ground = GroundState{BaseState{name: "Ground", parser: parser}}
	parser.OscString = OscStringState{BaseState{name: "OscString", parser: parser}}

	parser.stateMap = []State{
		parser.CsiEntry,
		parser.CsiParam,
		parser.DcsEntry,
		parser.Escape,
		parser.EscapeIntermediate,
		parser.Error,
		parser.Ground,
		parser.OscString,
	}

	parser.currState = getState(initialState, parser.stateMap)

	logger.Infof("CreateParser: parser %p", parser)
	return parser
}

func getState(name string, states []State) State {
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
		return errors.New(fmt.Sprintf("New state of 'nil' is invalid."))
	}

	if newState != ap.currState {
		if err := ap.changeState(newState); err != nil {
			return err
		}
	}

	return nil
}

func (ap *AnsiParser) changeState(newState State) error {
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
