package mount // import "github.com/docker/docker/pkg/mount"

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

func parseInfoFile(r io.Reader, filter FilterFunc) ([]*Info, error) {
	s := bufio.NewScanner(r)
	out := []*Info{}
	for s.Scan() {
		if err := s.Err(); err != nil {
			return nil, err
		}
		/*
		   36 35 98:0 /mnt1 /mnt2 rw,noatime master:1 - ext3 /dev/root rw,errors=continue
		   (1)(2)(3)   (4)   (5)      (6)      (7)   (8) (9)   (10)         (11)

		   (1) mount ID:  unique identifier of the mount (may be reused after umount)
		   (2) parent ID:  ID of parent (or of self for the top of the mount tree)
		   (3) major:minor:  value of st_dev for files on filesystem
		   (4) root:  root of the mount within the filesystem
		   (5) mount point:  mount point relative to the process's root
		   (6) mount options:  per mount options
		   (7) optional fields:  zero or more fields of the form "tag[:value]"
		   (8) separator:  marks the end of the optional fields
		   (9) filesystem type:  name of filesystem of the form "type[.subtype]"
		   (10) mount source:  filesystem specific information or "none"
		   (11) super options:  per super block options
		*/

		text := s.Text()
		fields := strings.Split(text, " ")
		numFields := len(fields)
		if numFields < 10 {
			// should be at least 10 fields
			return nil, fmt.Errorf("Parsing '%s' failed: not enough fields (%d)", text, numFields)
		}

		p := &Info{}
		// ignore any numbers parsing errors, as there should not be any
		p.ID, _ = strconv.Atoi(fields[0])
		p.Parent, _ = strconv.Atoi(fields[1])
		mm := strings.Split(fields[2], ":")
		if len(mm) != 2 {
			return nil, fmt.Errorf("Parsing '%s' failed: unexpected minor:major pair %s", text, mm)
		}
		p.Major, _ = strconv.Atoi(mm[0])
		p.Minor, _ = strconv.Atoi(mm[1])

		p.Root = fields[3]
		p.Mountpoint = fields[4]
		p.Opts = fields[5]

		var skip, stop bool
		if filter != nil {
			// filter out entries we're not interested in
			skip, stop = filter(p)
			if skip {
				continue
			}
		}

		// one or more optional fields, when a separator (-)
		i := 6
		for ; i < numFields && fields[i] != "-"; i++ {
			switch i {
			case 6:
				p.Optional = fields[6]
			default:
				/* NOTE there might be more optional fields before the such as
				   fields[7]...fields[N] (where N < sepIndex), although
				   as of Linux kernel 4.15 the only known ones are
				   mount propagation flags in fields[6]. The correct
				   behavior is to ignore any unknown optional fields.
				*/
				break
			}
		}
		if i == numFields {
			return nil, fmt.Errorf("Parsing '%s' failed: missing separator ('-')", text)
		}

		// There should be 3 fields after the separator...
		if i+4 > numFields {
			return nil, fmt.Errorf("Parsing '%s' failed: not enough fields after a separator", text)
		}
		// ... but in Linux <= 3.9 mounting a cifs with spaces in a share name
		// (like "//serv/My Documents") _may_ end up having a space in the last field
		// of mountinfo (like "unc=//serv/My Documents"). Since kernel 3.10-rc1, cifs
		// option unc= is ignored,  so a space should not appear. In here we ignore
		// those "extra" fields caused by extra spaces.
		p.Fstype = fields[i+1]
		p.Source = fields[i+2]
		p.VfsOpts = fields[i+3]

		out = append(out, p)
		if stop {
			break
		}
	}
	return out, nil
}

// Parse /proc/self/mountinfo because comparing Dev and ino does not work from
// bind mounts
func parseMountTable(filter FilterFunc) ([]*Info, error) {
	f, err := os.Open("/proc/self/mountinfo")
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseInfoFile(f, filter)
}

// PidMountInfo collects the mounts for a specific process ID. If the process
// ID is unknown, it is better to use `GetMounts` which will inspect
// "/proc/self/mountinfo" instead.
func PidMountInfo(pid int) ([]*Info, error) {
	f, err := os.Open(fmt.Sprintf("/proc/%d/mountinfo", pid))
	if err != nil {
		return nil, err
	}
	defer f.Close()

	return parseInfoFile(f, nil)
}
