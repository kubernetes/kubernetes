//go:build (freebsd && cgo) || (openbsd && cgo) || (darwin && cgo)
// +build freebsd,cgo openbsd,cgo darwin,cgo

package mountinfo

/*
#include <sys/param.h>
#include <sys/ucred.h>
#include <sys/mount.h>
*/
import "C"

import (
	"fmt"
	"reflect"
	"unsafe"
)

// parseMountTable returns information about mounted filesystems
func parseMountTable(filter FilterFunc) ([]*Info, error) {
	var rawEntries *C.struct_statfs

	count := int(C.getmntinfo(&rawEntries, C.MNT_WAIT))
	if count == 0 {
		return nil, fmt.Errorf("failed to call getmntinfo")
	}

	var entries []C.struct_statfs
	header := (*reflect.SliceHeader)(unsafe.Pointer(&entries))
	header.Cap = count
	header.Len = count
	header.Data = uintptr(unsafe.Pointer(rawEntries))

	var out []*Info
	for _, entry := range entries {
		var mountinfo Info
		var skip, stop bool
		mountinfo.Mountpoint = C.GoString(&entry.f_mntonname[0])
		mountinfo.FSType = C.GoString(&entry.f_fstypename[0])
		mountinfo.Source = C.GoString(&entry.f_mntfromname[0])

		if filter != nil {
			// filter out entries we're not interested in
			skip, stop = filter(&mountinfo)
			if skip {
				continue
			}
		}

		out = append(out, &mountinfo)
		if stop {
			break
		}
	}
	return out, nil
}

func mounted(path string) (bool, error) {
	path, err := normalizePath(path)
	if err != nil {
		return false, err
	}
	// Fast path: compare st.st_dev fields.
	// This should always work for FreeBSD and OpenBSD.
	mounted, err := mountedByStat(path)
	if err == nil {
		return mounted, nil
	}

	// Fallback to parsing mountinfo
	return mountedByMountinfo(path)
}
