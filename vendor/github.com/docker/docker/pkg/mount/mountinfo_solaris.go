// +build solaris,cgo

package mount

/*
#include <stdio.h>
#include <sys/mnttab.h>
*/
import "C"

import (
	"fmt"
)

func parseMountTable() ([]*Info, error) {
	mnttab := C.fopen(C.CString(C.MNTTAB), C.CString("r"))
	if mnttab == nil {
		return nil, fmt.Errorf("Failed to open %s", C.MNTTAB)
	}

	var out []*Info
	var mp C.struct_mnttab

	ret := C.getmntent(mnttab, &mp)
	for ret == 0 {
		var mountinfo Info
		mountinfo.Mountpoint = C.GoString(mp.mnt_mountp)
		mountinfo.Source = C.GoString(mp.mnt_special)
		mountinfo.Fstype = C.GoString(mp.mnt_fstype)
		mountinfo.Opts = C.GoString(mp.mnt_mntopts)
		out = append(out, &mountinfo)
		ret = C.getmntent(mnttab, &mp)
	}

	C.fclose(mnttab)
	return out, nil
}
