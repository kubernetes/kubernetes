package ansiterm

type CsiEntryState struct {
	BaseState
}

func (csiState CsiEntryState) Handle(b byte) (s State, e error) {
	logger.Infof("CsiEntry::Handle %#x", b)

	nextState, err := csiState.BaseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(Alphabetics, b):
		return csiState.parser.Ground, nil
	case sliceContains(CsiCollectables, b):
		return csiState.parser.CsiParam, nil
	case sliceContains(Executors, b):
		return csiState, csiState.parser.execute()
	}

	return csiState, nil
}

func (csiState CsiEntryState) Transition(s State) error {
	logger.Infof("CsiEntry::Transition %s --> %s", csiState.Name(), s.Name())
	csiState.BaseState.Transition(s)

	switch s {
	case csiState.parser.Ground:
		return csiState.parser.csiDispatch()
	case csiState.parser.CsiParam:
		switch {
		case sliceContains(CsiParams, csiState.parser.context.currentChar):
			csiState.parser.collectParam()
		case sliceContains(Intermeds, csiState.parser.context.currentChar):
			csiState.parser.collectInter()
		}
	}

	return nil
}

func (csiState CsiEntryState) Enter() error {
	csiState.parser.clear()
	return nil
}
