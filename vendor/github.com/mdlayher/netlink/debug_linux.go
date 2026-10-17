//go:build linux

package netlink

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"os"
	"strconv"
	"strings"
	"syscall"
	"unsafe"

	"golang.org/x/sys/unix"
)

// newDebugger creates a debugger by parsing key=value arguments.
func newDebugger(args []string) *debugger {
	d := &debugger{
		Log:    log.New(os.Stderr, "nl: ", 0),
		Level:  1,
		Format: "mnl",
	}
	for _, a := range args {
		kv := strings.Split(a, "=")
		if len(kv) != 2 {
			continue
		}
		switch kv[0] {
		case "level":
			level, err := strconv.Atoi(kv[1])
			if err != nil {
				panicf("netlink: invalid NLDEBUG level: %q", a)
			}
			d.Level = level
		case "format":
			d.Format = kv[1]
		}
	}
	return d
}

// debugf prints debugging information at the specified level, if d.Level is high enough to print the message.
func (d *debugger) debugf(level int, format string, v ...any) {
	if d.Level < level {
		return
	}
	switch d.Format {
	case "mnl":
		colorize := true
		_, err := unix.IoctlGetWinsize(int(os.Stdout.Fd()), unix.TIOCGWINSZ)
		if err != nil {
			colorize = false
		}
		for _, iface := range v {
			if msg, ok := iface.(Message); ok {
				nlmsgFprintf(d.Log.Writer(), msg, colorize)
			} else {
				d.Log.Printf(format, v...)
			}
		}
	default:
		d.Log.Printf(format, v...)
	}
}

// nlmsgFprintfHeader prints the netlink message header to fd.
func nlmsgFprintfHeader(fd io.Writer, nlh Header) {
	fmt.Fprintf(fd, "----------------\t------------------\n")
	fmt.Fprintf(fd, "|  %010d  |\t| message length |\n", nlh.Length)
	fmt.Fprintf(fd, "| %05d | %s%s%s%s |\t|  type | flags  |\n",
		nlh.Type,
		ternary(nlh.Flags&Request != 0, "R", "-"),
		ternary(nlh.Flags&Multi != 0, "M", "-"),
		ternary(nlh.Flags&Acknowledge != 0, "A", "-"),
		ternary(nlh.Flags&Echo != 0, "E", "-"),
	)
	fmt.Fprintf(fd, "|  %010d  |\t| sequence number|\n", nlh.Sequence)
	fmt.Fprintf(fd, "|  %010d  |\t|     port ID    |\n", nlh.PID)
	fmt.Fprintf(fd, "----------------\t------------------\n")
}

// nlmsgFprintf prints a single Message for netlink errors and attributes.
func nlmsgFprintf(fd io.Writer, m Message, colorize bool) {
	var hasHeader bool
	nlmsgFprintfHeader(fd, m.Header)
	switch {
	case m.Header.Type == Error:
		hasHeader = true
	case m.Header.Type == Done && m.Header.Flags&Multi != 0:
		if len(m.Data) == 0 {
			return
		}
	default:
		// Neither, nothing to do.
	}
	// Errno occupies 4 bytes.
	const endErrno = 4
	if len(m.Data) < endErrno {
		return
	}

	c := int32(binary.NativeEndian.Uint32(m.Data[:endErrno]))
	if c != 0 {
		b := m.Data[0:4]
		fmt.Fprintf(fd, "| %.2x %.2x %.2x %.2x  |\t",
			0xff&b[0], 0xff&b[1],
			0xff&b[2], 0xff&b[3])
		fmt.Fprintf(fd, "|  extra header  |\n")
	}

	// Flags indicate an extended acknowledgement. The type/flags combination
	// checked above determines the offset where the TLVs occur.
	var off int
	if hasHeader {
		// There is an nlmsghdr preceding the TLVs.
		if len(m.Data) < endErrno+nlmsgHeaderLen {
			return
		}
		// The TLVs should be at the offset indicated by the nlmsghdr.length,
		// plus the offset where the header began. But make sure the calculated
		// offset is still in-bounds.
		h := *(*Header)(unsafe.Pointer(&m.Data[endErrno : endErrno+nlmsgHeaderLen][0]))
		off = endErrno + int(h.Length)
		if len(m.Data) < off {
			return
		}
	} else {
		// There is no nlmsghdr preceding the TLVs, parse them directly.
		off = endErrno
	}

	data := m.Data[off:]
	for i := 0; i < len(data); {
		// Make sure there's at least a header's worth of data to read on each iteration.
		if len(data[i:]) < nlaHeaderLen {
			break
		}
		// Extract the length of the attribute.
		l := int(binary.NativeEndian.Uint16(data[i:]))
		// extract the type
		t := binary.NativeEndian.Uint16(data[i+2:])
		// print attribute header
		if colorize {
			fmt.Fprintf(fd, "|\033[1;31m%05d|\033[1;32m%s%s|\033[1;34m%05d\033[0m|\t",
				l,
				ternary(t&syscall.NLA_F_NESTED != 0, "N", "-"),
				ternary(t&syscall.NLA_F_NET_BYTEORDER != 0, "B", "-"),
				t&attrTypeMask)
			fmt.Fprintf(fd, "|len |flags| type|\n")
		} else {
			fmt.Fprintf(fd, "|%05d|%s%s|%05d|\t",
				l,
				ternary(t&syscall.NLA_F_NESTED != 0, "N", "-"),
				ternary(t&syscall.NLA_F_NET_BYTEORDER != 0, "B", "-"),
				t&attrTypeMask)
			fmt.Fprintf(fd, "|len |flags| type|\n")
		}

		nextAttr := i + nlaAlign(l)
		i += nlaHeaderLen

		// Ignore zero-length attributes.
		if l == 0 {
			continue
		}
		// If nested check the next attribute
		if t&syscall.NLA_F_NESTED != 0 {
			continue
		}
		// Print the remaining attributes bytes
		for ; i < nextAttr; i += 4 {
			fmt.Fprintf(fd, "| %.2x %.2x %.2x %.2x  |\t",
				0xff&data[i], 0xff&data[i+1],
				0xff&data[i+2], 0xff&data[i+3])
			fmt.Fprintf(fd, "|      data      |")
			fmt.Fprintf(fd, "\t %s %s %s %s\n",
				ternary(strconv.IsPrint(rune(data[i])), string(data[i]), " "),
				ternary(strconv.IsPrint(rune(data[i+1])), string(data[i+1]), " "),
				ternary(strconv.IsPrint(rune(data[i+2])), string(data[i+2]), " "),
				ternary(strconv.IsPrint(rune(data[i+3])), string(data[i+3]), " "),
			)
		}
	}
	fmt.Fprintf(fd, "----------------\t------------------\n")
}

// ternary returns iftrue if cond is true, else iffalse.
func ternary(cond bool, iftrue string, iffalse string) string {
	if cond {
		return iftrue
	}
	return iffalse
}
