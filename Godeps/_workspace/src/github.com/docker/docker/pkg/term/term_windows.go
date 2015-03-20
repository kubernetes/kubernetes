// +build windows

package term

type State struct {
	mode uint32
}

type Winsize struct {
	Height uint16
	Width  uint16
	x      uint16
	y      uint16
}

func GetWinsize(fd uintptr) (*Winsize, error) {
	ws := &Winsize{}
	var info *CONSOLE_SCREEN_BUFFER_INFO
	info, err := GetConsoleScreenBufferInfo(fd)
	if err != nil {
		return nil, err
	}
	ws.Height = uint16(info.srWindow.Right - info.srWindow.Left + 1)
	ws.Width = uint16(info.srWindow.Bottom - info.srWindow.Top + 1)

	ws.x = 0 // todo azlinux -- this is the pixel size of the Window, and not currently used by any caller
	ws.y = 0

	return ws, nil
}

func SetWinsize(fd uintptr, ws *Winsize) error {
	return nil
}

// IsTerminal returns true if the given file descriptor is a terminal.
func IsTerminal(fd uintptr) bool {
	_, e := GetConsoleMode(fd)
	return e == nil
}

// Restore restores the terminal connected to the given file descriptor to a
// previous state.
func RestoreTerminal(fd uintptr, state *State) error {
	return SetConsoleMode(fd, state.mode)
}

func SaveState(fd uintptr) (*State, error) {
	mode, e := GetConsoleMode(fd)
	if e != nil {
		return nil, e
	}
	return &State{mode}, nil
}

// see http://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx for these flag settings
func DisableEcho(fd uintptr, state *State) error {
	state.mode &^= (ENABLE_ECHO_INPUT)
	state.mode |= (ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT)
	return SetConsoleMode(fd, state.mode)
}

func SetRawTerminal(fd uintptr) (*State, error) {
	oldState, err := MakeRaw(fd)
	if err != nil {
		return nil, err
	}
	// TODO (azlinux): implement handling interrupt and restore state of terminal
	return oldState, err
}

// MakeRaw puts the terminal connected to the given file descriptor into raw
// mode and returns the previous state of the terminal so that it can be
// restored.
func MakeRaw(fd uintptr) (*State, error) {
	var state *State
	state, err := SaveState(fd)
	if err != nil {
		return nil, err
	}

	// see http://msdn.microsoft.com/en-us/library/windows/desktop/ms683462(v=vs.85).aspx for these flag settings
	state.mode &^= (ENABLE_ECHO_INPUT | ENABLE_PROCESSED_INPUT | ENABLE_LINE_INPUT)
	err = SetConsoleMode(fd, state.mode)
	if err != nil {
		return nil, err
	}
	return state, nil
}
