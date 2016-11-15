package ansiterm

type GroundState struct {
	BaseState
}

func (gs GroundState) Handle(b byte) (s State, e error) {
	gs.parser.context.currentChar = b

	nextState, err := gs.BaseState.Handle(b)
	if nextState != nil || err != nil {
		return nextState, err
	}

	switch {
	case sliceContains(Printables, b):
		return gs, gs.parser.print()

	case sliceContains(Executors, b):
		return gs, gs.parser.execute()
	}

	return gs, nil
}
