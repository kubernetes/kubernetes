// +build windows

package lcow

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"

	"github.com/Microsoft/hcsshim"
	"github.com/Microsoft/opengcs/service/gcsutils/remotefs"
	"github.com/containerd/continuity/driver"
)

type lcowfile struct {
	process   hcsshim.Process
	stdin     io.WriteCloser
	stdout    io.ReadCloser
	stderr    io.ReadCloser
	fs        *lcowfs
	guestPath string
}

func (l *lcowfs) Open(path string) (driver.File, error) {
	return l.OpenFile(path, os.O_RDONLY, 0)
}

func (l *lcowfs) OpenFile(path string, flag int, perm os.FileMode) (_ driver.File, err error) {
	flagStr := strconv.FormatInt(int64(flag), 10)
	permStr := strconv.FormatUint(uint64(perm), 8)

	commandLine := fmt.Sprintf("%s %s %s %s", remotefs.RemotefsCmd, remotefs.OpenFileCmd, flagStr, permStr)
	env := make(map[string]string)
	env["PATH"] = "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:"
	processConfig := &hcsshim.ProcessConfig{
		EmulateConsole:    false,
		CreateStdInPipe:   true,
		CreateStdOutPipe:  true,
		CreateStdErrPipe:  true,
		CreateInUtilityVm: true,
		WorkingDirectory:  "/bin",
		Environment:       env,
		CommandLine:       commandLine,
	}

	process, err := l.currentSVM.config.Uvm.CreateProcess(processConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to open file %s: %s", path, err)
	}

	stdin, stdout, stderr, err := process.Stdio()
	if err != nil {
		process.Kill()
		process.Close()
		return nil, fmt.Errorf("failed to open file pipes %s: %s", path, err)
	}

	lf := &lcowfile{
		process:   process,
		stdin:     stdin,
		stdout:    stdout,
		stderr:    stderr,
		fs:        l,
		guestPath: path,
	}

	if _, err := lf.getResponse(); err != nil {
		return nil, fmt.Errorf("failed to open file %s: %s", path, err)
	}
	return lf, nil
}

func (l *lcowfile) Read(b []byte) (int, error) {
	hdr := &remotefs.FileHeader{
		Cmd:  remotefs.Read,
		Size: uint64(len(b)),
	}

	if err := remotefs.WriteFileHeader(l.stdin, hdr, nil); err != nil {
		return 0, err
	}

	buf, err := l.getResponse()
	if err != nil {
		return 0, nil
	}

	n := copy(b, buf)
	return n, nil
}

func (l *lcowfile) Write(b []byte) (int, error) {
	hdr := &remotefs.FileHeader{
		Cmd:  remotefs.Write,
		Size: uint64(len(b)),
	}

	if err := remotefs.WriteFileHeader(l.stdin, hdr, b); err != nil {
		return 0, err
	}

	_, err := l.getResponse()
	if err != nil {
		return 0, nil
	}

	return len(b), nil
}

func (l *lcowfile) Seek(offset int64, whence int) (int64, error) {
	seekHdr := &remotefs.SeekHeader{
		Offset: offset,
		Whence: int32(whence),
	}

	buf := &bytes.Buffer{}
	if err := binary.Write(buf, binary.BigEndian, seekHdr); err != nil {
		return 0, err
	}

	hdr := &remotefs.FileHeader{
		Cmd:  remotefs.Write,
		Size: uint64(buf.Len()),
	}
	if err := remotefs.WriteFileHeader(l.stdin, hdr, buf.Bytes()); err != nil {
		return 0, err
	}

	resBuf, err := l.getResponse()
	if err != nil {
		return 0, err
	}

	var res int64
	if err := binary.Read(bytes.NewBuffer(resBuf), binary.BigEndian, &res); err != nil {
		return 0, err
	}
	return res, nil
}

func (l *lcowfile) Close() error {
	hdr := &remotefs.FileHeader{
		Cmd:  remotefs.Close,
		Size: 0,
	}

	if err := remotefs.WriteFileHeader(l.stdin, hdr, nil); err != nil {
		return err
	}

	_, err := l.getResponse()
	return err
}

func (l *lcowfile) Readdir(n int) ([]os.FileInfo, error) {
	nStr := strconv.FormatInt(int64(n), 10)

	// Unlike the other File functions, this one can just be run without maintaining state,
	// so just do the normal runRemoteFSProcess way.
	buf := &bytes.Buffer{}
	if err := l.fs.runRemoteFSProcess(nil, buf, remotefs.ReadDirCmd, l.guestPath, nStr); err != nil {
		return nil, err
	}

	var info []remotefs.FileInfo
	if err := json.Unmarshal(buf.Bytes(), &info); err != nil {
		return nil, nil
	}

	osInfo := make([]os.FileInfo, len(info))
	for i := range info {
		osInfo[i] = &info[i]
	}
	return osInfo, nil
}

func (l *lcowfile) getResponse() ([]byte, error) {
	hdr, err := remotefs.ReadFileHeader(l.stdout)
	if err != nil {
		return nil, err
	}

	if hdr.Cmd != remotefs.CmdOK {
		// Something went wrong during the openfile in the server.
		// Parse stderr and return that as an error
		eerr, err := remotefs.ReadError(l.stderr)
		if eerr != nil {
			return nil, remotefs.ExportedToError(eerr)
		}

		// Maybe the parsing went wrong?
		if err != nil {
			return nil, err
		}

		// At this point, we know something went wrong in the remotefs program, but
		// we we don't know why.
		return nil, fmt.Errorf("unknown error")
	}

	// Successful command, we might have some data to read (for Read + Seek)
	buf := make([]byte, hdr.Size, hdr.Size)
	if _, err := io.ReadFull(l.stdout, buf); err != nil {
		return nil, err
	}
	return buf, nil
}
