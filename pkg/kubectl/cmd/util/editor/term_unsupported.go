// +build windows

package editor

import (
	"os"
)

// childSignals are the allowed signals that can be sent to children in Windows to terminate
var childSignals = []os.Signal{os.Interrupt}
