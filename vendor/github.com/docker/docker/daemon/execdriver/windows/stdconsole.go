// +build windows

package windows

// StdConsole is for when using a container non-interactively
type StdConsole struct {
}

func NewStdConsole() *StdConsole {
	return &StdConsole{}
}

func (s *StdConsole) Resize(h, w int) error {
	// we do not need to resize a non tty
	return nil
}

func (s *StdConsole) Close() error {
	// nothing to close here
	return nil
}
