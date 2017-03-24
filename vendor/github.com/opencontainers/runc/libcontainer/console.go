package libcontainer

import (
	"encoding/json"
	"fmt"
	"io"
	"os"
)

// Console represents a pseudo TTY.
type Console interface {
	io.ReadWriteCloser

	// Path returns the filesystem path to the slave side of the pty.
	Path() string

	// Fd returns the fd for the master of the pty.
	File() *os.File
}

const (
	TerminalInfoVersion uint32 = 201610041
	TerminalInfoType    uint8  = 'T'
)

// TerminalInfo is the structure which is passed as the non-ancillary data
// in the sendmsg(2) call when runc is run with --console-socket. It
// contains some information about the container which the console master fd
// relates to (to allow for consumers to use a single unix socket to handle
// multiple containers). This structure will probably move to runtime-spec
// at some point. But for now it lies in libcontainer.
type TerminalInfo struct {
	// Version of the API.
	Version uint32 `json:"version"`

	// Type of message (future proofing).
	Type uint8 `json:"type"`

	// Container contains the ID of the container.
	ContainerID string `json:"container_id"`
}

func (ti *TerminalInfo) String() string {
	encoded, err := json.Marshal(*ti)
	if err != nil {
		panic(err)
	}
	return string(encoded)
}

func NewTerminalInfo(containerId string) *TerminalInfo {
	return &TerminalInfo{
		Version:     TerminalInfoVersion,
		Type:        TerminalInfoType,
		ContainerID: containerId,
	}
}

func GetTerminalInfo(encoded string) (*TerminalInfo, error) {
	ti := new(TerminalInfo)
	if err := json.Unmarshal([]byte(encoded), ti); err != nil {
		return nil, err
	}

	if ti.Type != TerminalInfoType {
		return nil, fmt.Errorf("terminal info: incorrect type in payload (%q): %q", TerminalInfoType, ti.Type)
	}
	if ti.Version != TerminalInfoVersion {
		return nil, fmt.Errorf("terminal info: incorrect version in payload (%q): %q", TerminalInfoVersion, ti.Version)
	}

	return ti, nil
}
