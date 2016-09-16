package ansiterm

type StateId int

type State interface {
	Enter() error
	Exit() error
	Handle(byte) (State, error)
	Name() string
	Transition(State) error
}

type BaseState struct {
	name   string
	parser *AnsiParser
}

func (base BaseState) Enter() error {
	return nil
}

func (base BaseState) Exit() error {
	return nil
}

func (base BaseState) Handle(b byte) (s State, e error) {

	switch {
	case b == CSI_ENTRY:
		return base.parser.CsiEntry, nil
	case b == DCS_ENTRY:
		return base.parser.DcsEntry, nil
	case b == ANSI_ESCAPE_PRIMARY:
		return base.parser.Escape, nil
	case b == OSC_STRING:
		return base.parser.OscString, nil
	case sliceContains(ToGroundBytes, b):
		return base.parser.Ground, nil
	}

	return nil, nil
}

func (base BaseState) Name() string {
	return base.name
}

func (base BaseState) Transition(s State) error {
	if s == base.parser.Ground {
		execBytes := []byte{0x18}
		execBytes = append(execBytes, 0x1A)
		execBytes = append(execBytes, getByteRange(0x80, 0x8F)...)
		execBytes = append(execBytes, getByteRange(0x91, 0x97)...)
		execBytes = append(execBytes, 0x99)
		execBytes = append(execBytes, 0x9A)

		if sliceContains(execBytes, base.parser.context.currentChar) {
			return base.parser.execute()
		}
	}

	return nil
}

type DcsEntryState struct {
	BaseState
}

type ErrorState struct {
	BaseState
}
