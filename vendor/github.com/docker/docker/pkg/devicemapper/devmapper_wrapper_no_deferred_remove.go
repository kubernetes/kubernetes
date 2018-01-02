// +build linux,cgo,libdm_no_deferred_remove

package devicemapper

// LibraryDeferredRemovalSupport is not supported when statically linked.
const LibraryDeferredRemovalSupport = false

func dmTaskDeferredRemoveFct(task *cdmTask) int {
	// Error. Nobody should be calling it.
	return -1
}

func dmTaskGetInfoWithDeferredFct(task *cdmTask, info *Info) int {
	return -1
}
