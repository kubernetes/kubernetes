// Copyright 2015 The rkt Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package tar

import (
	"archive/tar"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"path/filepath"
	"strconv"
	"syscall"

	"github.com/coreos/rkt/pkg/multicall"
	"github.com/coreos/rkt/pkg/sys"
	"github.com/coreos/rkt/pkg/user"
	"github.com/hashicorp/errwrap"
)

const (
	multicallName = "extracttar"
	fileMapFdNum  = 3
)

var mcEntrypoint multicall.Entrypoint

func init() {
	mcEntrypoint = multicall.Add(multicallName, extractTarCommand)
}

// Because this function is executed by multicall in a different process, it is not possible to use errwrap to return errors
func extractTarCommand() error {
	if len(os.Args) != 5 {
		return fmt.Errorf("incorrect number of arguments. Usage: %s DIR {true|false} uidShift uidCount", multicallName)
	}
	if !sys.HasChrootCapability() {
		return fmt.Errorf("chroot capability not available.")
	}
	dir := os.Args[1]
	if !filepath.IsAbs(dir) {
		return fmt.Errorf("dir %s must be an absolute path", dir)
	}
	overwrite, err := strconv.ParseBool(os.Args[2])
	if err != nil {
		return fmt.Errorf("error parsing overwrite argument: %v", err)
	}

	us, err := strconv.ParseUint(os.Args[3], 10, 32)
	if err != nil {
		return fmt.Errorf("error parsing uidShift argument: %v", err)
	}
	uc, err := strconv.ParseUint(os.Args[4], 10, 32)
	if err != nil {
		return fmt.Errorf("error parsing uidCount argument: %v", err)
	}

	uidRange := &user.UidRange{Shift: uint32(us), Count: uint32(uc)}

	if err := syscall.Chroot(dir); err != nil {
		return fmt.Errorf("failed to chroot in %s: %v", dir, err)
	}
	if err := syscall.Chdir("/"); err != nil {
		return fmt.Errorf("failed to chdir: %v", err)
	}
	fileMapFile := os.NewFile(uintptr(fileMapFdNum), "fileMap")

	fileMap := map[string]struct{}{}
	if err := json.NewDecoder(fileMapFile).Decode(&fileMap); err != nil {
		return fmt.Errorf("error decoding fileMap: %v", err)
	}
	editor, err := NewUidShiftingFilePermEditor(uidRange)
	if err != nil {
		return fmt.Errorf("error determining current user: %v", err)
	}
	if err := ExtractTarInsecure(tar.NewReader(os.Stdin), "/", overwrite, fileMap, editor); err != nil {
		return fmt.Errorf("error extracting tar: %v", err)
	}

	// flush remaining bytes
	io.Copy(ioutil.Discard, os.Stdin)

	return nil
}

// ExtractTar extracts a tarball (from a io.Reader) into the given directory
// if pwl is not nil, only the paths in the map are extracted.
// If overwrite is true, existing files will be overwritten.
// The extraction is executed by fork/exec()ing a new process. The new process
// needs the CAP_SYS_CHROOT capability.
func ExtractTar(rs io.Reader, dir string, overwrite bool, uidRange *user.UidRange, pwl PathWhitelistMap) error {
	r, w, err := os.Pipe()
	if err != nil {
		return err
	}
	defer w.Close()
	enc := json.NewEncoder(w)
	cmd := mcEntrypoint.Cmd(dir, strconv.FormatBool(overwrite),
		strconv.FormatUint(uint64(uidRange.Shift), 10),
		strconv.FormatUint(uint64(uidRange.Count), 10))
	cmd.ExtraFiles = []*os.File{r}

	cmd.Stdin = rs
	encodeCh := make(chan error)
	go func() {
		encodeCh <- enc.Encode(pwl)
	}()

	out, err := cmd.CombinedOutput()

	// read from blocking encodeCh to release the goroutine
	encodeErr := <-encodeCh
	if err != nil {
		return fmt.Errorf("extracttar error: %v, output: %s", err, out)
	}
	if encodeErr != nil {
		return errwrap.Wrap(errors.New("extracttar failed to json encode filemap"), encodeErr)
	}
	return nil
}
