// +build windows

package windows

import (
	"github.com/microsoft/hcsshim"
)

// TtyConsole is for when using a container interactively
type TtyConsole struct {
	id        string
	processid uint32
}

func NewTtyConsole(id string, processid uint32) *TtyConsole {
	tty := &TtyConsole{
		id:        id,
		processid: processid,
	}
	return tty
}

func (t *TtyConsole) Resize(h, w int) error {
	return hcsshim.ResizeConsoleInComputeSystem(t.id, t.processid, h, w)
}

func (t *TtyConsole) Close() error {
	return nil
}
