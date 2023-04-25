package libcontainer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io"

	"github.com/opencontainers/runc/libcontainer/utils"
)

type syncType string

// Constants that are used for synchronisation between the parent and child
// during container setup. They come in pairs (with procError being a generic
// response which is followed by an &initError).
//
//	[  child  ] <-> [   parent   ]
//
//	procHooks   --> [run hooks]
//	            <-- procResume
//
//	procReady   --> [final setup]
//	            <-- procRun
//
//	procSeccomp --> [pick up seccomp fd with pidfd_getfd()]
//	            <-- procSeccompDone
const (
	procError       syncType = "procError"
	procReady       syncType = "procReady"
	procRun         syncType = "procRun"
	procHooks       syncType = "procHooks"
	procResume      syncType = "procResume"
	procSeccomp     syncType = "procSeccomp"
	procSeccompDone syncType = "procSeccompDone"
)

type syncT struct {
	Type syncType `json:"type"`
	Fd   int      `json:"fd"`
}

// initError is used to wrap errors for passing them via JSON,
// as encoding/json can't unmarshal into error type.
type initError struct {
	Message string `json:"message,omitempty"`
}

func (i initError) Error() string {
	return i.Message
}

// writeSync is used to write to a synchronisation pipe. An error is returned
// if there was a problem writing the payload.
func writeSync(pipe io.Writer, sync syncType) error {
	return writeSyncWithFd(pipe, sync, -1)
}

// writeSyncWithFd is used to write to a synchronisation pipe. An error is
// returned if there was a problem writing the payload.
func writeSyncWithFd(pipe io.Writer, sync syncType, fd int) error {
	if err := utils.WriteJSON(pipe, syncT{sync, fd}); err != nil {
		return fmt.Errorf("writing syncT %q: %w", string(sync), err)
	}
	return nil
}

// readSync is used to read from a synchronisation pipe. An error is returned
// if we got an initError, the pipe was closed, or we got an unexpected flag.
func readSync(pipe io.Reader, expected syncType) error {
	var procSync syncT
	if err := json.NewDecoder(pipe).Decode(&procSync); err != nil {
		if errors.Is(err, io.EOF) {
			return errors.New("parent closed synchronisation channel")
		}
		return fmt.Errorf("failed reading error from parent: %w", err)
	}

	if procSync.Type == procError {
		var ierr initError

		if err := json.NewDecoder(pipe).Decode(&ierr); err != nil {
			return fmt.Errorf("failed reading error from parent: %w", err)
		}

		return &ierr
	}

	if procSync.Type != expected {
		return errors.New("invalid synchronisation flag from parent")
	}
	return nil
}

// parseSync runs the given callback function on each syncT received from the
// child. It will return once io.EOF is returned from the given pipe.
func parseSync(pipe io.Reader, fn func(*syncT) error) error {
	dec := json.NewDecoder(pipe)
	for {
		var sync syncT
		if err := dec.Decode(&sync); err != nil {
			if errors.Is(err, io.EOF) {
				break
			}
			return err
		}

		// We handle this case outside fn for cleanliness reasons.
		var ierr *initError
		if sync.Type == procError {
			if err := dec.Decode(&ierr); err != nil && !errors.Is(err, io.EOF) {
				return fmt.Errorf("error decoding proc error from init: %w", err)
			}
			if ierr != nil {
				return ierr
			}
			// Programmer error.
			panic("No error following JSON procError payload.")
		}

		if err := fn(&sync); err != nil {
			return err
		}
	}
	return nil
}
