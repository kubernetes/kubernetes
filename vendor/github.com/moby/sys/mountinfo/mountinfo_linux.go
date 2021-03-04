// +build go1.13

package mountinfo

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
	var err error
	for s.Scan() {
		if err = s.Err(); err != nil {
			return nil, err
		}
		/*
		   See http://man7.org/linux/man-pages/man5/proc.5.html

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

		   In other words, we have:
		    * 6 mandatory fields	(1)..(6)
		    * 0 or more optional fields	(7)
		    * a separator field		(8)
		    * 3 mandatory fields	(9)..(11)
		*/

		text := s.Text()
		fields := strings.Split(text, " ")
		numFields := len(fields)
		if numFields < 10 {
			// should be at least 10 fields
			return nil, fmt.Errorf("Parsing '%s' failed: not enough fields (%d)", text, numFields)
		}

		// separator field
		sepIdx := numFields - 4
		// In Linux <= 3.9 mounting a cifs with spaces in a share
		// name (like "//srv/My Docs") _may_ end up having a space
		// in the last field of mountinfo (like "unc=//serv/My Docs").
		// Since kernel 3.10-rc1, cifs option "unc=" is ignored,
		// so spaces should not appear.
		//
		// Check for a separator, and work around the spaces bug
		for fields[sepIdx] != "-" {
			sepIdx--
			if sepIdx == 5 {
				return nil, fmt.Errorf("Parsing '%s' failed: missing - separator", text)
			}
		}

		p := &Info{}

		// Fill in the fields that a filter might check
		p.Mountpoint, err = strconv.Unquote(`"` + fields[4] + `"`)
		if err != nil {
			return nil, fmt.Errorf("Parsing '%s' failed: unable to unquote mount point field: %w", fields[4], err)
		}
		p.Fstype = fields[sepIdx+1]
		p.Source = fields[sepIdx+2]
		p.VfsOpts = fields[sepIdx+3]

		// Run a filter soon so we can skip parsing/adding entries
		// the caller is not interested in
		var skip, stop bool
		if filter != nil {
			skip, stop = filter(p)
			if skip {
				continue
			}
		}

		// Fill in the rest of the fields

		// ignore any numbers parsing errors, as there should not be any
		p.ID, _ = strconv.Atoi(fields[0])
		p.Parent, _ = strconv.Atoi(fields[1])
		mm := strings.Split(fields[2], ":")
		if len(mm) != 2 {
			return nil, fmt.Errorf("Parsing '%s' failed: unexpected minor:major pair %s", text, mm)
		}
		p.Major, _ = strconv.Atoi(mm[0])
		p.Minor, _ = strconv.Atoi(mm[1])

		p.Root, err = strconv.Unquote(`"` + fields[3] + `"`)
		if err != nil {
			return nil, fmt.Errorf("Parsing '%s' failed: unable to unquote root field: %w", fields[3], err)
		}

		p.Opts = fields[5]

		// zero or more optional fields
		switch {
		case sepIdx == 6:
			// zero, do nothing
		case sepIdx == 7:
			p.Optional = fields[6]
		default:
			p.Optional = strings.Join(fields[6:sepIdx-1], " ")
		}

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
