// +build windows

package lcow

import (
	"bytes"
	"encoding/json"
	"os"
	"strconv"

	"github.com/Microsoft/opengcs/service/gcsutils/remotefs"

	"github.com/containerd/continuity/driver"
	"github.com/sirupsen/logrus"
)

var _ driver.Driver = &lcowfs{}

func (l *lcowfs) Readlink(p string) (string, error) {
	logrus.Debugf("removefs.readlink args: %s", p)

	result := &bytes.Buffer{}
	if err := l.runRemoteFSProcess(nil, result, remotefs.ReadlinkCmd, p); err != nil {
		return "", err
	}
	return result.String(), nil
}

func (l *lcowfs) Mkdir(path string, mode os.FileMode) error {
	return l.mkdir(path, mode, remotefs.MkdirCmd)
}

func (l *lcowfs) MkdirAll(path string, mode os.FileMode) error {
	return l.mkdir(path, mode, remotefs.MkdirAllCmd)
}

func (l *lcowfs) mkdir(path string, mode os.FileMode, cmd string) error {
	modeStr := strconv.FormatUint(uint64(mode), 8)
	logrus.Debugf("remotefs.%s args: %s %s", cmd, path, modeStr)
	return l.runRemoteFSProcess(nil, nil, cmd, path, modeStr)
}

func (l *lcowfs) Remove(path string) error {
	return l.remove(path, remotefs.RemoveCmd)
}

func (l *lcowfs) RemoveAll(path string) error {
	return l.remove(path, remotefs.RemoveAllCmd)
}

func (l *lcowfs) remove(path string, cmd string) error {
	logrus.Debugf("remotefs.%s args: %s", cmd, path)
	return l.runRemoteFSProcess(nil, nil, cmd, path)
}

func (l *lcowfs) Link(oldname, newname string) error {
	return l.link(oldname, newname, remotefs.LinkCmd)
}

func (l *lcowfs) Symlink(oldname, newname string) error {
	return l.link(oldname, newname, remotefs.SymlinkCmd)
}

func (l *lcowfs) link(oldname, newname, cmd string) error {
	logrus.Debugf("remotefs.%s args: %s %s", cmd, oldname, newname)
	return l.runRemoteFSProcess(nil, nil, cmd, oldname, newname)
}

func (l *lcowfs) Lchown(name string, uid, gid int64) error {
	uidStr := strconv.FormatInt(uid, 10)
	gidStr := strconv.FormatInt(gid, 10)

	logrus.Debugf("remotefs.lchown args: %s %s %s", name, uidStr, gidStr)
	return l.runRemoteFSProcess(nil, nil, remotefs.LchownCmd, name, uidStr, gidStr)
}

// Lchmod changes the mode of an file not following symlinks.
func (l *lcowfs) Lchmod(path string, mode os.FileMode) error {
	modeStr := strconv.FormatUint(uint64(mode), 8)
	logrus.Debugf("remotefs.lchmod args: %s %s", path, modeStr)
	return l.runRemoteFSProcess(nil, nil, remotefs.LchmodCmd, path, modeStr)
}

func (l *lcowfs) Mknod(path string, mode os.FileMode, major, minor int) error {
	modeStr := strconv.FormatUint(uint64(mode), 8)
	majorStr := strconv.FormatUint(uint64(major), 10)
	minorStr := strconv.FormatUint(uint64(minor), 10)

	logrus.Debugf("remotefs.mknod args: %s %s %s %s", path, modeStr, majorStr, minorStr)
	return l.runRemoteFSProcess(nil, nil, remotefs.MknodCmd, path, modeStr, majorStr, minorStr)
}

func (l *lcowfs) Mkfifo(path string, mode os.FileMode) error {
	modeStr := strconv.FormatUint(uint64(mode), 8)
	logrus.Debugf("remotefs.mkfifo args: %s %s", path, modeStr)
	return l.runRemoteFSProcess(nil, nil, remotefs.MkfifoCmd, path, modeStr)
}

func (l *lcowfs) Stat(p string) (os.FileInfo, error) {
	return l.stat(p, remotefs.StatCmd)
}

func (l *lcowfs) Lstat(p string) (os.FileInfo, error) {
	return l.stat(p, remotefs.LstatCmd)
}

func (l *lcowfs) stat(path string, cmd string) (os.FileInfo, error) {
	logrus.Debugf("remotefs.stat inputs: %s %s", cmd, path)

	output := &bytes.Buffer{}
	err := l.runRemoteFSProcess(nil, output, cmd, path)
	if err != nil {
		return nil, err
	}

	var fi remotefs.FileInfo
	if err := json.Unmarshal(output.Bytes(), &fi); err != nil {
		return nil, err
	}

	logrus.Debugf("remotefs.stat success. got: %v\n", fi)
	return &fi, nil
}
