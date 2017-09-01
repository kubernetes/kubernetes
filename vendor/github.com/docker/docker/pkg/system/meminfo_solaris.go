// +build solaris,cgo

package system

import (
	"fmt"
	"unsafe"
)

// #cgo LDFLAGS: -lkstat
// #include <unistd.h>
// #include <stdlib.h>
// #include <stdio.h>
// #include <kstat.h>
// #include <sys/swap.h>
// #include <sys/param.h>
// struct swaptable *allocSwaptable(int num) {
//	struct swaptable *st;
//	struct swapent *swapent;
// 	st = (struct swaptable *)malloc(num * sizeof(swapent_t) + sizeof (int));
//	swapent = st->swt_ent;
//	for (int i = 0; i < num; i++,swapent++) {
//		swapent->ste_path = (char *)malloc(MAXPATHLEN * sizeof (char));
//	}
//	st->swt_n = num;
//	return st;
//}
// void freeSwaptable (struct swaptable *st) {
//	struct swapent *swapent = st->swt_ent;
//	for (int i = 0; i < st->swt_n; i++,swapent++) {
//		free(swapent->ste_path);
//	}
//	free(st);
// }
// swapent_t getSwapEnt(swapent_t *ent, int i) {
//	return ent[i];
// }
// int64_t getPpKernel() {
//	int64_t pp_kernel = 0;
//	kstat_ctl_t *ksc;
//	kstat_t *ks;
//	kstat_named_t *knp;
//	kid_t kid;
//
//	if ((ksc = kstat_open()) == NULL) {
//		return -1;
//	}
//	if ((ks = kstat_lookup(ksc, "unix", 0, "system_pages")) == NULL) {
//		return -1;
//	}
//	if (((kid = kstat_read(ksc, ks, NULL)) == -1) ||
//	    ((knp = kstat_data_lookup(ks, "pp_kernel")) == NULL)) {
//		return -1;
//	}
//	switch (knp->data_type) {
//	case KSTAT_DATA_UINT64:
//		pp_kernel = knp->value.ui64;
//		break;
//	case KSTAT_DATA_UINT32:
//		pp_kernel = knp->value.ui32;
//		break;
//	}
//	pp_kernel *= sysconf(_SC_PAGESIZE);
//	return (pp_kernel > 0 ? pp_kernel : -1);
// }
import "C"

// Get the system memory info using sysconf same as prtconf
func getTotalMem() int64 {
	pagesize := C.sysconf(C._SC_PAGESIZE)
	npages := C.sysconf(C._SC_PHYS_PAGES)
	return int64(pagesize * npages)
}

func getFreeMem() int64 {
	pagesize := C.sysconf(C._SC_PAGESIZE)
	npages := C.sysconf(C._SC_AVPHYS_PAGES)
	return int64(pagesize * npages)
}

// ReadMemInfo retrieves memory statistics of the host system and returns a
//  MemInfo type.
func ReadMemInfo() (*MemInfo, error) {

	ppKernel := C.getPpKernel()
	MemTotal := getTotalMem()
	MemFree := getFreeMem()
	SwapTotal, SwapFree, err := getSysSwap()

	if ppKernel < 0 || MemTotal < 0 || MemFree < 0 || SwapTotal < 0 ||
		SwapFree < 0 {
		return nil, fmt.Errorf("error getting system memory info %v\n", err)
	}

	meminfo := &MemInfo{}
	// Total memory is total physical memory less than memory locked by kernel
	meminfo.MemTotal = MemTotal - int64(ppKernel)
	meminfo.MemFree = MemFree
	meminfo.SwapTotal = SwapTotal
	meminfo.SwapFree = SwapFree

	return meminfo, nil
}

func getSysSwap() (int64, int64, error) {
	var tSwap int64
	var fSwap int64
	var diskblksPerPage int64
	num, err := C.swapctl(C.SC_GETNSWP, nil)
	if err != nil {
		return -1, -1, err
	}
	st := C.allocSwaptable(num)
	_, err = C.swapctl(C.SC_LIST, unsafe.Pointer(st))
	if err != nil {
		C.freeSwaptable(st)
		return -1, -1, err
	}

	diskblksPerPage = int64(C.sysconf(C._SC_PAGESIZE) >> C.DEV_BSHIFT)
	for i := 0; i < int(num); i++ {
		swapent := C.getSwapEnt(&st.swt_ent[0], C.int(i))
		tSwap += int64(swapent.ste_pages) * diskblksPerPage
		fSwap += int64(swapent.ste_free) * diskblksPerPage
	}
	C.freeSwaptable(st)
	return tSwap, fSwap, nil
}
