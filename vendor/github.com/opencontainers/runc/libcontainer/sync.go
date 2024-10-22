package libcontainer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/opencontainers/runc/libcontainer/utils"

	"github.com/sirupsen/logrus"
)

type syncType string

// Constants that are used for synchronisation between the parent and child
// during container setup. They come in pairs (with procError being a generic
// response which is followed by an &initError).
//
//	     [  child  ] <-> [   parent   ]
//
//	procMountPlease      --> [open(2) or open_tree(2) and configure mount]
//	  Arg: configs.Mount
//	                     <-- procMountFd
//	                           file: mountfd
//
//	procSeccomp         --> [forward fd to listenerPath]
//	  file: seccomp fd
//	                    --- no return synchronisation
//
//	procHooks --> [run hooks]
//	          <-- procHooksDone
//
//	procReady --> [final setup]
//	          <-- procRun
//
//	procSeccomp --> [grab seccomp fd with pidfd_getfd()]
//	            <-- procSeccompDone
const (
	procError       syncType = "procError"
	procReady       syncType = "procReady"
	procRun         syncType = "procRun"
	procHooks       syncType = "procHooks"
	procHooksDone   syncType = "procHooksDone"
	procMountPlease syncType = "procMountPlease"
	procMountFd     syncType = "procMountFd"
	procSeccomp     syncType = "procSeccomp"
	procSeccompDone syncType = "procSeccompDone"
)

type syncFlags int

const (
	syncFlagHasFd syncFlags = (1 << iota)
)

type syncT struct {
	Type  syncType         `json:"type"`
	Flags syncFlags        `json:"flags"`
	Arg   *json.RawMessage `json:"arg,omitempty"`
	File  *os.File         `json:"-"` // passed oob through SCM_RIGHTS
}

func (s syncT) String() string {
	str := "type:" + string(s.Type)
	if s.Flags != 0 {
		str += " flags:0b" + strconv.FormatInt(int64(s.Flags), 2)
	}
	if s.Arg != nil {
		str += " arg:" + string(*s.Arg)
	}
	if s.File != nil {
		str += " file:" + s.File.Name() + " (fd:" + strconv.Itoa(int(s.File.Fd())) + ")"
	}
	return str
}

// initError is used to wrap errors for passing them via JSON,
// as encoding/json can't unmarshal into error type.
type initError struct {
	Message string `json:"message,omitempty"`
}

func (i initError) Error() string {
	return i.Message
}

func doWriteSync(pipe *syncSocket, sync syncT) error {
	sync.Flags &= ^syncFlagHasFd
	if sync.File != nil {
		sync.Flags |= syncFlagHasFd
	}
	logrus.Debugf("writing sync %s", sync)
	data, err := json.Marshal(sync)
	if err != nil {
		return fmt.Errorf("marshal sync %v: %w", sync.Type, err)
	}
	if _, err := pipe.WritePacket(data); err != nil {
		return fmt.Errorf("writing sync %v: %w", sync.Type, err)
	}
	if sync.Flags&syncFlagHasFd != 0 {
		logrus.Debugf("writing sync file %s", sync)
		if err := utils.SendFile(pipe.File(), sync.File); err != nil {
			return fmt.Errorf("sending file after sync %q: %w", sync.Type, err)
		}
	}
	return nil
}

func writeSync(pipe *syncSocket, sync syncType) error {
	return doWriteSync(pipe, syncT{Type: sync})
}

func writeSyncArg(pipe *syncSocket, sync syncType, arg interface{}) error {
	argJSON, err := json.Marshal(arg)
	if err != nil {
		return fmt.Errorf("writing sync %v: marshal argument failed: %w", sync, err)
	}
	argJSONMsg := json.RawMessage(argJSON)
	return doWriteSync(pipe, syncT{Type: sync, Arg: &argJSONMsg})
}

func doReadSync(pipe *syncSocket) (syncT, error) {
	var sync syncT
	logrus.Debugf("reading sync")
	packet, err := pipe.ReadPacket()
	if err != nil {
		if errors.Is(err, io.EOF) {
			logrus.Debugf("sync pipe closed")
			return sync, err
		}
		return sync, fmt.Errorf("reading from parent failed: %w", err)
	}
	if err := json.Unmarshal(packet, &sync); err != nil {
		return sync, fmt.Errorf("unmarshal sync from parent failed: %w", err)
	}
	logrus.Debugf("read sync %s", sync)
	if sync.Type == procError {
		var ierr initError
		if sync.Arg == nil {
			return sync, errors.New("procError missing error payload")
		}
		if err := json.Unmarshal(*sync.Arg, &ierr); err != nil {
			return sync, fmt.Errorf("unmarshal procError failed: %w", err)
		}
		return sync, &ierr
	}
	if sync.Flags&syncFlagHasFd != 0 {
		logrus.Debugf("reading sync file %s", sync)
		file, err := utils.RecvFile(pipe.File())
		if err != nil {
			return sync, fmt.Errorf("receiving fd from sync %v failed: %w", sync.Type, err)
		}
		sync.File = file
	}
	return sync, nil
}

func readSyncFull(pipe *syncSocket, expected syncType) (syncT, error) {
	sync, err := doReadSync(pipe)
	if err != nil {
		return sync, err
	}
	if sync.Type != expected {
		return sync, fmt.Errorf("unexpected synchronisation flag: got %q, expected %q", sync.Type, expected)
	}
	return sync, nil
}

func readSync(pipe *syncSocket, expected syncType) error {
	sync, err := readSyncFull(pipe, expected)
	if err != nil {
		return err
	}
	if sync.Arg != nil {
		return fmt.Errorf("sync %v had unexpected argument passed: %q", expected, string(*sync.Arg))
	}
	if sync.File != nil {
		_ = sync.File.Close()
		return fmt.Errorf("sync %v had unexpected file passed", sync.Type)
	}
	return nil
}

// parseSync runs the given callback function on each syncT received from the
// child. It will return once io.EOF is returned from the given pipe.
func parseSync(pipe *syncSocket, fn func(*syncT) error) error {
	for {
		sync, err := doReadSync(pipe)
		if err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}
		if err := fn(&sync); err != nil {
			return err
		}
	}
	return nil
}
