// Copyright (c) 2013, Suryandaru Triandana <syndtr@gmail.com>
// All rights reserved.
//
// Use of this source code is governed by a BSD-style license that can be
// found in the LICENSE file.

package capability

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"strings"
	"syscall"
)

var errUnknownVers = errors.New("unknown capability version")

const (
	linuxCapVer1 = 0x19980330
	linuxCapVer2 = 0x20071026
	linuxCapVer3 = 0x20080522
)

var (
	capVers    uint32
	capLastCap Cap
)

func init() {
	var hdr capHeader
	capget(&hdr, nil)
	capVers = hdr.version

	if initLastCap() == nil {
		CAP_LAST_CAP = capLastCap
		if capLastCap > 31 {
			capUpperMask = (uint32(1) << (uint(capLastCap) - 31)) - 1
		} else {
			capUpperMask = 0
		}
	}
}

func initLastCap() error {
	if capLastCap != 0 {
		return nil
	}

	f, err := os.Open("/proc/sys/kernel/cap_last_cap")
	if err != nil {
		return err
	}
	defer f.Close()

	var b []byte = make([]byte, 11)
	_, err = f.Read(b)
	if err != nil {
		return err
	}

	fmt.Sscanf(string(b), "%d", &capLastCap)

	return nil
}

func mkStringCap(c Capabilities, which CapType) (ret string) {
	for i, first := Cap(0), true; i <= CAP_LAST_CAP; i++ {
		if !c.Get(which, i) {
			continue
		}
		if first {
			first = false
		} else {
			ret += ", "
		}
		ret += i.String()
	}
	return
}

func mkString(c Capabilities, max CapType) (ret string) {
	ret = "{"
	for i := CapType(1); i <= max; i <<= 1 {
		ret += " " + i.String() + "=\""
		if c.Empty(i) {
			ret += "empty"
		} else if c.Full(i) {
			ret += "full"
		} else {
			ret += c.StringCap(i)
		}
		ret += "\""
	}
	ret += " }"
	return
}

func newPid(pid int) (c Capabilities, err error) {
	switch capVers {
	case linuxCapVer1:
		p := new(capsV1)
		p.hdr.version = capVers
		p.hdr.pid = pid
		c = p
	case linuxCapVer2, linuxCapVer3:
		p := new(capsV3)
		p.hdr.version = capVers
		p.hdr.pid = pid
		c = p
	default:
		err = errUnknownVers
		return
	}
	err = c.Load()
	if err != nil {
		c = nil
	}
	return
}

type capsV1 struct {
	hdr  capHeader
	data capData
}

func (c *capsV1) Get(which CapType, what Cap) bool {
	if what > 32 {
		return false
	}

	switch which {
	case EFFECTIVE:
		return (1<<uint(what))&c.data.effective != 0
	case PERMITTED:
		return (1<<uint(what))&c.data.permitted != 0
	case INHERITABLE:
		return (1<<uint(what))&c.data.inheritable != 0
	}

	return false
}

func (c *capsV1) getData(which CapType) (ret uint32) {
	switch which {
	case EFFECTIVE:
		ret = c.data.effective
	case PERMITTED:
		ret = c.data.permitted
	case INHERITABLE:
		ret = c.data.inheritable
	}
	return
}

func (c *capsV1) Empty(which CapType) bool {
	return c.getData(which) == 0
}

func (c *capsV1) Full(which CapType) bool {
	return (c.getData(which) & 0x7fffffff) == 0x7fffffff
}

func (c *capsV1) Set(which CapType, caps ...Cap) {
	for _, what := range caps {
		if what > 32 {
			continue
		}

		if which&EFFECTIVE != 0 {
			c.data.effective |= 1 << uint(what)
		}
		if which&PERMITTED != 0 {
			c.data.permitted |= 1 << uint(what)
		}
		if which&INHERITABLE != 0 {
			c.data.inheritable |= 1 << uint(what)
		}
	}
}

func (c *capsV1) Unset(which CapType, caps ...Cap) {
	for _, what := range caps {
		if what > 32 {
			continue
		}

		if which&EFFECTIVE != 0 {
			c.data.effective &= ^(1 << uint(what))
		}
		if which&PERMITTED != 0 {
			c.data.permitted &= ^(1 << uint(what))
		}
		if which&INHERITABLE != 0 {
			c.data.inheritable &= ^(1 << uint(what))
		}
	}
}

func (c *capsV1) Fill(kind CapType) {
	if kind&CAPS == CAPS {
		c.data.effective = 0x7fffffff
		c.data.permitted = 0x7fffffff
		c.data.inheritable = 0
	}
}

func (c *capsV1) Clear(kind CapType) {
	if kind&CAPS == CAPS {
		c.data.effective = 0
		c.data.permitted = 0
		c.data.inheritable = 0
	}
}

func (c *capsV1) StringCap(which CapType) (ret string) {
	return mkStringCap(c, which)
}

func (c *capsV1) String() (ret string) {
	return mkString(c, BOUNDING)
}

func (c *capsV1) Load() (err error) {
	return capget(&c.hdr, &c.data)
}

func (c *capsV1) Apply(kind CapType) error {
	if kind&CAPS == CAPS {
		return capset(&c.hdr, &c.data)
	}
	return nil
}

type capsV3 struct {
	hdr    capHeader
	data   [2]capData
	bounds [2]uint32
}

func (c *capsV3) Get(which CapType, what Cap) bool {
	var i uint
	if what > 31 {
		i = uint(what) >> 5
		what %= 32
	}

	switch which {
	case EFFECTIVE:
		return (1<<uint(what))&c.data[i].effective != 0
	case PERMITTED:
		return (1<<uint(what))&c.data[i].permitted != 0
	case INHERITABLE:
		return (1<<uint(what))&c.data[i].inheritable != 0
	case BOUNDING:
		return (1<<uint(what))&c.bounds[i] != 0
	}

	return false
}

func (c *capsV3) getData(which CapType, dest []uint32) {
	switch which {
	case EFFECTIVE:
		dest[0] = c.data[0].effective
		dest[1] = c.data[1].effective
	case PERMITTED:
		dest[0] = c.data[0].permitted
		dest[1] = c.data[1].permitted
	case INHERITABLE:
		dest[0] = c.data[0].inheritable
		dest[1] = c.data[1].inheritable
	case BOUNDING:
		dest[0] = c.bounds[0]
		dest[1] = c.bounds[1]
	}
}

func (c *capsV3) Empty(which CapType) bool {
	var data [2]uint32
	c.getData(which, data[:])
	return data[0] == 0 && data[1] == 0
}

func (c *capsV3) Full(which CapType) bool {
	var data [2]uint32
	c.getData(which, data[:])
	if (data[0] & 0xffffffff) != 0xffffffff {
		return false
	}
	return (data[1] & capUpperMask) == capUpperMask
}

func (c *capsV3) Set(which CapType, caps ...Cap) {
	for _, what := range caps {
		var i uint
		if what > 31 {
			i = uint(what) >> 5
			what %= 32
		}

		if which&EFFECTIVE != 0 {
			c.data[i].effective |= 1 << uint(what)
		}
		if which&PERMITTED != 0 {
			c.data[i].permitted |= 1 << uint(what)
		}
		if which&INHERITABLE != 0 {
			c.data[i].inheritable |= 1 << uint(what)
		}
		if which&BOUNDING != 0 {
			c.bounds[i] |= 1 << uint(what)
		}
	}
}

func (c *capsV3) Unset(which CapType, caps ...Cap) {
	for _, what := range caps {
		var i uint
		if what > 31 {
			i = uint(what) >> 5
			what %= 32
		}

		if which&EFFECTIVE != 0 {
			c.data[i].effective &= ^(1 << uint(what))
		}
		if which&PERMITTED != 0 {
			c.data[i].permitted &= ^(1 << uint(what))
		}
		if which&INHERITABLE != 0 {
			c.data[i].inheritable &= ^(1 << uint(what))
		}
		if which&BOUNDING != 0 {
			c.bounds[i] &= ^(1 << uint(what))
		}
	}
}

func (c *capsV3) Fill(kind CapType) {
	if kind&CAPS == CAPS {
		c.data[0].effective = 0xffffffff
		c.data[0].permitted = 0xffffffff
		c.data[0].inheritable = 0
		c.data[1].effective = 0xffffffff
		c.data[1].permitted = 0xffffffff
		c.data[1].inheritable = 0
	}

	if kind&BOUNDS == BOUNDS {
		c.bounds[0] = 0xffffffff
		c.bounds[1] = 0xffffffff
	}
}

func (c *capsV3) Clear(kind CapType) {
	if kind&CAPS == CAPS {
		c.data[0].effective = 0
		c.data[0].permitted = 0
		c.data[0].inheritable = 0
		c.data[1].effective = 0
		c.data[1].permitted = 0
		c.data[1].inheritable = 0
	}

	if kind&BOUNDS == BOUNDS {
		c.bounds[0] = 0
		c.bounds[1] = 0
	}
}

func (c *capsV3) StringCap(which CapType) (ret string) {
	return mkStringCap(c, which)
}

func (c *capsV3) String() (ret string) {
	return mkString(c, BOUNDING)
}

func (c *capsV3) Load() (err error) {
	err = capget(&c.hdr, &c.data[0])
	if err != nil {
		return
	}

	var status_path string

	if c.hdr.pid == 0 {
		status_path = fmt.Sprintf("/proc/self/status")
	} else {
		status_path = fmt.Sprintf("/proc/%d/status", c.hdr.pid)
	}

	f, err := os.Open(status_path)
	if err != nil {
		return
	}
	b := bufio.NewReader(f)
	for {
		line, e := b.ReadString('\n')
		if e != nil {
			if e != io.EOF {
				err = e
			}
			break
		}
		if strings.HasPrefix(line, "CapB") {
			fmt.Sscanf(line[4:], "nd:  %08x%08x", &c.bounds[1], &c.bounds[0])
			break
		}
	}
	f.Close()

	return
}

func (c *capsV3) Apply(kind CapType) (err error) {
	if kind&BOUNDS == BOUNDS {
		var data [2]capData
		err = capget(&c.hdr, &data[0])
		if err != nil {
			return
		}
		if (1<<uint(CAP_SETPCAP))&data[0].effective != 0 {
			for i := Cap(0); i <= CAP_LAST_CAP; i++ {
				if c.Get(BOUNDING, i) {
					continue
				}
				err = prctl(syscall.PR_CAPBSET_DROP, uintptr(i), 0, 0, 0)
				if err != nil {
					// Ignore EINVAL since the capability may not be supported in this system.
					if errno, ok := err.(syscall.Errno); ok && errno == syscall.EINVAL {
						err = nil
						continue
					}
					return
				}
			}
		}
	}

	if kind&CAPS == CAPS {
		return capset(&c.hdr, &c.data[0])
	}

	return
}

func newFile(path string) (c Capabilities, err error) {
	c = &capsFile{path: path}
	err = c.Load()
	if err != nil {
		c = nil
	}
	return
}

type capsFile struct {
	path string
	data vfscapData
}

func (c *capsFile) Get(which CapType, what Cap) bool {
	var i uint
	if what > 31 {
		if c.data.version == 1 {
			return false
		}
		i = uint(what) >> 5
		what %= 32
	}

	switch which {
	case EFFECTIVE:
		return (1<<uint(what))&c.data.effective[i] != 0
	case PERMITTED:
		return (1<<uint(what))&c.data.data[i].permitted != 0
	case INHERITABLE:
		return (1<<uint(what))&c.data.data[i].inheritable != 0
	}

	return false
}

func (c *capsFile) getData(which CapType, dest []uint32) {
	switch which {
	case EFFECTIVE:
		dest[0] = c.data.effective[0]
		dest[1] = c.data.effective[1]
	case PERMITTED:
		dest[0] = c.data.data[0].permitted
		dest[1] = c.data.data[1].permitted
	case INHERITABLE:
		dest[0] = c.data.data[0].inheritable
		dest[1] = c.data.data[1].inheritable
	}
}

func (c *capsFile) Empty(which CapType) bool {
	var data [2]uint32
	c.getData(which, data[:])
	return data[0] == 0 && data[1] == 0
}

func (c *capsFile) Full(which CapType) bool {
	var data [2]uint32
	c.getData(which, data[:])
	if c.data.version == 0 {
		return (data[0] & 0x7fffffff) == 0x7fffffff
	}
	if (data[0] & 0xffffffff) != 0xffffffff {
		return false
	}
	return (data[1] & capUpperMask) == capUpperMask
}

func (c *capsFile) Set(which CapType, caps ...Cap) {
	for _, what := range caps {
		var i uint
		if what > 31 {
			if c.data.version == 1 {
				continue
			}
			i = uint(what) >> 5
			what %= 32
		}

		if which&EFFECTIVE != 0 {
			c.data.effective[i] |= 1 << uint(what)
		}
		if which&PERMITTED != 0 {
			c.data.data[i].permitted |= 1 << uint(what)
		}
		if which&INHERITABLE != 0 {
			c.data.data[i].inheritable |= 1 << uint(what)
		}
	}
}

func (c *capsFile) Unset(which CapType, caps ...Cap) {
	for _, what := range caps {
		var i uint
		if what > 31 {
			if c.data.version == 1 {
				continue
			}
			i = uint(what) >> 5
			what %= 32
		}

		if which&EFFECTIVE != 0 {
			c.data.effective[i] &= ^(1 << uint(what))
		}
		if which&PERMITTED != 0 {
			c.data.data[i].permitted &= ^(1 << uint(what))
		}
		if which&INHERITABLE != 0 {
			c.data.data[i].inheritable &= ^(1 << uint(what))
		}
	}
}

func (c *capsFile) Fill(kind CapType) {
	if kind&CAPS == CAPS {
		c.data.effective[0] = 0xffffffff
		c.data.data[0].permitted = 0xffffffff
		c.data.data[0].inheritable = 0
		if c.data.version == 2 {
			c.data.effective[1] = 0xffffffff
			c.data.data[1].permitted = 0xffffffff
			c.data.data[1].inheritable = 0
		}
	}
}

func (c *capsFile) Clear(kind CapType) {
	if kind&CAPS == CAPS {
		c.data.effective[0] = 0
		c.data.data[0].permitted = 0
		c.data.data[0].inheritable = 0
		if c.data.version == 2 {
			c.data.effective[1] = 0
			c.data.data[1].permitted = 0
			c.data.data[1].inheritable = 0
		}
	}
}

func (c *capsFile) StringCap(which CapType) (ret string) {
	return mkStringCap(c, which)
}

func (c *capsFile) String() (ret string) {
	return mkString(c, INHERITABLE)
}

func (c *capsFile) Load() (err error) {
	return getVfsCap(c.path, &c.data)
}

func (c *capsFile) Apply(kind CapType) (err error) {
	if kind&CAPS == CAPS {
		return setVfsCap(c.path, &c.data)
	}
	return
}
