//go:build linux

package userns

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
	"os"
	"unsafe"

	"github.com/opencontainers/runc/libcontainer/configs"
	"github.com/sirupsen/logrus"
)

/*
#include <stdlib.h>
extern int spawn_userns_cat(char *userns_path, char *path, int outfd, int errfd);
*/
import "C"

func parseIdmapData(data []byte) (ms []configs.IDMap, err error) {
	scanner := bufio.NewScanner(bytes.NewReader(data))
	for scanner.Scan() {
		var m configs.IDMap
		line := scanner.Text()
		if _, err := fmt.Sscanf(line, "%d %d %d", &m.ContainerID, &m.HostID, &m.Size); err != nil {
			return nil, fmt.Errorf("parsing id map failed: invalid format in line %q: %w", line, err)
		}
		ms = append(ms, m)
	}
	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("parsing id map failed: %w", err)
	}
	return ms, nil
}

// Do something equivalent to nsenter --user=<nsPath> cat <path>, but more
// efficiently. Returns the contents of the requested file from within the user
// namespace.
func spawnUserNamespaceCat(nsPath string, path string) ([]byte, error) {
	rdr, wtr, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("create pipe for userns spawn failed: %w", err)
	}
	defer rdr.Close()
	defer wtr.Close()

	errRdr, errWtr, err := os.Pipe()
	if err != nil {
		return nil, fmt.Errorf("create error pipe for userns spawn failed: %w", err)
	}
	defer errRdr.Close()
	defer errWtr.Close()

	cNsPath := C.CString(nsPath)
	defer C.free(unsafe.Pointer(cNsPath))
	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	childPid := C.spawn_userns_cat(cNsPath, cPath, C.int(wtr.Fd()), C.int(errWtr.Fd()))

	if childPid < 0 {
		return nil, fmt.Errorf("failed to spawn fork for userns")
	} else if childPid == 0 {
		// this should never happen
		panic("runc executing inside fork child -- unsafe state!")
	}

	// We are in the parent -- close the write end of the pipe before reading.
	wtr.Close()
	output, err := io.ReadAll(rdr)
	rdr.Close()
	if err != nil {
		return nil, fmt.Errorf("reading from userns spawn failed: %w", err)
	}

	// Ditto for the error pipe.
	errWtr.Close()
	errOutput, err := io.ReadAll(errRdr)
	errRdr.Close()
	if err != nil {
		return nil, fmt.Errorf("reading from userns spawn error pipe failed: %w", err)
	}
	errOutput = bytes.TrimSpace(errOutput)

	// Clean up the child.
	child, err := os.FindProcess(int(childPid))
	if err != nil {
		return nil, fmt.Errorf("could not find userns spawn process: %w", err)
	}
	state, err := child.Wait()
	if err != nil {
		return nil, fmt.Errorf("failed to wait for userns spawn process: %w", err)
	}
	if !state.Success() {
		errStr := string(errOutput)
		if errStr == "" {
			errStr = fmt.Sprintf("unknown error (status code %d)", state.ExitCode())
		}
		return nil, fmt.Errorf("userns spawn: %s", errStr)
	} else if len(errOutput) > 0 {
		// We can just ignore weird output in the error pipe if the process
		// didn't bail(), but for completeness output for debugging.
		logrus.Debugf("userns spawn succeeded but unexpected error message found: %s", string(errOutput))
	}
	// The subprocess succeeded, return whatever it wrote to the pipe.
	return output, nil
}

func GetUserNamespaceMappings(nsPath string) (uidMap, gidMap []configs.IDMap, err error) {
	var (
		pid         int
		extra       rune
		tryFastPath bool
	)

	// nsPath is usually of the form /proc/<pid>/ns/user, which means that we
	// already have a pid that is part of the user namespace and thus we can
	// just use the pid to read from /proc/<pid>/*id_map.
	//
	// Note that Sscanf doesn't consume the whole input, so we check for any
	// trailing data with %c. That way, we can be sure the pattern matched
	// /proc/$pid/ns/user _exactly_ iff n === 1.
	if n, _ := fmt.Sscanf(nsPath, "/proc/%d/ns/user%c", &pid, &extra); n == 1 {
		tryFastPath = pid > 0
	}

	for _, mapType := range []struct {
		name  string
		idMap *[]configs.IDMap
	}{
		{"uid_map", &uidMap},
		{"gid_map", &gidMap},
	} {
		var mapData []byte

		if tryFastPath {
			path := fmt.Sprintf("/proc/%d/%s", pid, mapType.name)
			data, err := os.ReadFile(path)
			if err != nil {
				// Do not error out here -- we need to try the slow path if the
				// fast path failed.
				logrus.Debugf("failed to use fast path to read %s from userns %s (error: %s), falling back to slow userns-join path", mapType.name, nsPath, err)
			} else {
				mapData = data
			}
		} else {
			logrus.Debugf("cannot use fast path to read %s from userns %s, falling back to slow userns-join path", mapType.name, nsPath)
		}

		if mapData == nil {
			// We have to actually join the namespace if we cannot take the
			// fast path. The path is resolved with respect to the child
			// process, so just use /proc/self.
			data, err := spawnUserNamespaceCat(nsPath, "/proc/self/"+mapType.name)
			if err != nil {
				return nil, nil, err
			}
			mapData = data
		}
		idMap, err := parseIdmapData(mapData)
		if err != nil {
			return nil, nil, fmt.Errorf("failed to parse %s of userns %s: %w", mapType.name, nsPath, err)
		}
		*mapType.idMap = idMap
	}

	return uidMap, gidMap, nil
}

// IsSameMapping returns whether or not the two id mappings are the same. Note
// that if the order of the mappings is different, or a mapping has been split,
// the mappings will be considered different.
func IsSameMapping(a, b []configs.IDMap) bool {
	if len(a) != len(b) {
		return false
	}
	for idx := range a {
		if a[idx] != b[idx] {
			return false
		}
	}
	return true
}
