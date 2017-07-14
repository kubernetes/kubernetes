package ansiterm

type CsiParamState struct {
	BaseState
}

func (csiState CsiParamState) Handle(b byte) (s State, e error) {
	logger.Infof("CsiParam::Handle %#x", b)

	nextState, err := csiState.BaseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(Alphabetics, b):
		return csiState.parser.Ground, nil
	case sliceContains(CsiCollectables, b):
		csiState.parser.collectParam()
		return csiState, nil
	case sliceContains(Executors, b):
		return csiState, csiState.parser.execute()
	}

	return csiState, nil
}

func (csiState CsiParamState) Transition(s State) error {
	logger.Infof("CsiParam::Transition %s --> %s", csiState.Name(), s.Name())
	csiState.BaseState.Transition(s)

	switch s {
	case csiState.parser.Ground:
		return csiState.parser.csiDispatch()
	}

	return nil
}
