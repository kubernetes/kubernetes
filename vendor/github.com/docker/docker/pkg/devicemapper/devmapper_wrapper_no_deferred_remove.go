// +build linux,cgo,libdm_no_deferred_remove

package devicemapper

// LibraryDeferredRemovalSupport tells if the feature is enabled in the build
const LibraryDeferredRemovalSupport = false

func dmTaskDeferredRemoveFct(task *cdmTask) int {
	// Error. Nobody should be calling it.
	return -1
}

func dmTaskGetInfoWithDeferredFct(task *cdmTask, info *Info) int {
	return -1
}
