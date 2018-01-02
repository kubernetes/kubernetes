// +build linux,cgo

package devicemapper

/*
#cgo LDFLAGS: -L. -ldevmapper
#define _GNU_SOURCE
#include <libdevmapper.h>
#include <linux/fs.h>   // FIXME: present only for BLKGETSIZE64, maybe we can remove it?

// FIXME: Can't we find a way to do the logging in pure Go?
extern void DevmapperLogCallback(int level, char *file, int line, int dm_errno_or_class, char *str);

static void	log_cb(int level, const char *file, int line, int dm_errno_or_class, const char *f, ...)
{
	char *buffer = NULL;
	va_list ap;
	int ret;

	va_start(ap, f);
	ret = vasprintf(&buffer, f, ap);
	va_end(ap);
	if (ret < 0) {
		// memory allocation failed -- should never happen?
		return;
	}

	DevmapperLogCallback(level, (char *)file, line, dm_errno_or_class, buffer);
	free(buffer);
}

static void	log_with_errno_init()
{
	dm_log_with_errno_init(log_cb);
}
*/
import "C"

import (
	"reflect"
	"unsafe"
)

type (
	cdmTask C.struct_dm_task
)

// IOCTL consts
const (
	BlkGetSize64 = C.BLKGETSIZE64
	BlkDiscard   = C.BLKDISCARD
)

// Devicemapper cookie flags.
const (
	DmUdevDisableSubsystemRulesFlag = C.DM_UDEV_DISABLE_SUBSYSTEM_RULES_FLAG
	DmUdevDisableDiskRulesFlag      = C.DM_UDEV_DISABLE_DISK_RULES_FLAG
	DmUdevDisableOtherRulesFlag     = C.DM_UDEV_DISABLE_OTHER_RULES_FLAG
	DmUdevDisableLibraryFallback    = C.DM_UDEV_DISABLE_LIBRARY_FALLBACK
)

// DeviceMapper mapped functions.
var (
	DmGetLibraryVersion       = dmGetLibraryVersionFct
	DmGetNextTarget           = dmGetNextTargetFct
	DmSetDevDir               = dmSetDevDirFct
	DmTaskAddTarget           = dmTaskAddTargetFct
	DmTaskCreate              = dmTaskCreateFct
	DmTaskDestroy             = dmTaskDestroyFct
	DmTaskGetDeps             = dmTaskGetDepsFct
	DmTaskGetInfo             = dmTaskGetInfoFct
	DmTaskGetDriverVersion    = dmTaskGetDriverVersionFct
	DmTaskRun                 = dmTaskRunFct
	DmTaskSetAddNode          = dmTaskSetAddNodeFct
	DmTaskSetCookie           = dmTaskSetCookieFct
	DmTaskSetMessage          = dmTaskSetMessageFct
	DmTaskSetName             = dmTaskSetNameFct
	DmTaskSetRo               = dmTaskSetRoFct
	DmTaskSetSector           = dmTaskSetSectorFct
	DmUdevWait                = dmUdevWaitFct
	DmUdevSetSyncSupport      = dmUdevSetSyncSupportFct
	DmUdevGetSyncSupport      = dmUdevGetSyncSupportFct
	DmCookieSupported         = dmCookieSupportedFct
	LogWithErrnoInit          = logWithErrnoInitFct
	DmTaskDeferredRemove      = dmTaskDeferredRemoveFct
	DmTaskGetInfoWithDeferred = dmTaskGetInfoWithDeferredFct
)

func free(p *C.char) {
	C.free(unsafe.Pointer(p))
}

func dmTaskDestroyFct(task *cdmTask) {
	C.dm_task_destroy((*C.struct_dm_task)(task))
}

func dmTaskCreateFct(taskType int) *cdmTask {
	return (*cdmTask)(C.dm_task_create(C.int(taskType)))
}

func dmTaskRunFct(task *cdmTask) int {
	ret, _ := C.dm_task_run((*C.struct_dm_task)(task))
	return int(ret)
}

func dmTaskSetNameFct(task *cdmTask, name string) int {
	Cname := C.CString(name)
	defer free(Cname)

	return int(C.dm_task_set_name((*C.struct_dm_task)(task), Cname))
}

func dmTaskSetMessageFct(task *cdmTask, message string) int {
	Cmessage := C.CString(message)
	defer free(Cmessage)

	return int(C.dm_task_set_message((*C.struct_dm_task)(task), Cmessage))
}

func dmTaskSetSectorFct(task *cdmTask, sector uint64) int {
	return int(C.dm_task_set_sector((*C.struct_dm_task)(task), C.uint64_t(sector)))
}

func dmTaskSetCookieFct(task *cdmTask, cookie *uint, flags uint16) int {
	cCookie := C.uint32_t(*cookie)
	defer func() {
		*cookie = uint(cCookie)
	}()
	return int(C.dm_task_set_cookie((*C.struct_dm_task)(task), &cCookie, C.uint16_t(flags)))
}

func dmTaskSetAddNodeFct(task *cdmTask, addNode AddNodeType) int {
	return int(C.dm_task_set_add_node((*C.struct_dm_task)(task), C.dm_add_node_t(addNode)))
}

func dmTaskSetRoFct(task *cdmTask) int {
	return int(C.dm_task_set_ro((*C.struct_dm_task)(task)))
}

func dmTaskAddTargetFct(task *cdmTask,
	start, size uint64, ttype, params string) int {

	Cttype := C.CString(ttype)
	defer free(Cttype)

	Cparams := C.CString(params)
	defer free(Cparams)

	return int(C.dm_task_add_target((*C.struct_dm_task)(task), C.uint64_t(start), C.uint64_t(size), Cttype, Cparams))
}

func dmTaskGetDepsFct(task *cdmTask) *Deps {
	Cdeps := C.dm_task_get_deps((*C.struct_dm_task)(task))
	if Cdeps == nil {
		return nil
	}

	// golang issue: https://github.com/golang/go/issues/11925
	hdr := reflect.SliceHeader{
		Data: uintptr(unsafe.Pointer(uintptr(unsafe.Pointer(Cdeps)) + unsafe.Sizeof(*Cdeps))),
		Len:  int(Cdeps.count),
		Cap:  int(Cdeps.count),
	}
	devices := *(*[]C.uint64_t)(unsafe.Pointer(&hdr))

	deps := &Deps{
		Count:  uint32(Cdeps.count),
		Filler: uint32(Cdeps.filler),
	}
	for _, device := range devices {
		deps.Device = append(deps.Device, uint64(device))
	}
	return deps
}

func dmTaskGetInfoFct(task *cdmTask, info *Info) int {
	Cinfo := C.struct_dm_info{}
	defer func() {
		info.Exists = int(Cinfo.exists)
		info.Suspended = int(Cinfo.suspended)
		info.LiveTable = int(Cinfo.live_table)
		info.InactiveTable = int(Cinfo.inactive_table)
		info.OpenCount = int32(Cinfo.open_count)
		info.EventNr = uint32(Cinfo.event_nr)
		info.Major = uint32(Cinfo.major)
		info.Minor = uint32(Cinfo.minor)
		info.ReadOnly = int(Cinfo.read_only)
		info.TargetCount = int32(Cinfo.target_count)
	}()
	return int(C.dm_task_get_info((*C.struct_dm_task)(task), &Cinfo))
}

func dmTaskGetDriverVersionFct(task *cdmTask) string {
	buffer := C.malloc(128)
	defer C.free(buffer)
	res := C.dm_task_get_driver_version((*C.struct_dm_task)(task), (*C.char)(buffer), 128)
	if res == 0 {
		return ""
	}
	return C.GoString((*C.char)(buffer))
}

func dmGetNextTargetFct(task *cdmTask, next unsafe.Pointer, start, length *uint64, target, params *string) unsafe.Pointer {
	var (
		Cstart, Clength      C.uint64_t
		CtargetType, Cparams *C.char
	)
	defer func() {
		*start = uint64(Cstart)
		*length = uint64(Clength)
		*target = C.GoString(CtargetType)
		*params = C.GoString(Cparams)
	}()

	nextp := C.dm_get_next_target((*C.struct_dm_task)(task), next, &Cstart, &Clength, &CtargetType, &Cparams)
	return nextp
}

func dmUdevSetSyncSupportFct(syncWithUdev int) {
	(C.dm_udev_set_sync_support(C.int(syncWithUdev)))
}

func dmUdevGetSyncSupportFct() int {
	return int(C.dm_udev_get_sync_support())
}

func dmUdevWaitFct(cookie uint) int {
	return int(C.dm_udev_wait(C.uint32_t(cookie)))
}

func dmCookieSupportedFct() int {
	return int(C.dm_cookie_supported())
}

func logWithErrnoInitFct() {
	C.log_with_errno_init()
}

func dmSetDevDirFct(dir string) int {
	Cdir := C.CString(dir)
	defer free(Cdir)

	return int(C.dm_set_dev_dir(Cdir))
}

func dmGetLibraryVersionFct(version *string) int {
	buffer := C.CString(string(make([]byte, 128)))
	defer free(buffer)
	defer func() {
		*version = C.GoString(buffer)
	}()
	return int(C.dm_get_library_version(buffer, 128))
}
