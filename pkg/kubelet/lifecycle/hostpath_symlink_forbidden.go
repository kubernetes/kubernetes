package lifecycle

import (
	"fmt"
	"k8s.io/api/core/v1"
	"os"
	"strings"
)

type hostpathSymlinkForbidden struct {
}

func NewHostpathSymlinkForbidden() *hostpathSymlinkForbidden {
	return &hostpathSymlinkForbidden{}
}

func (w *hostpathSymlinkForbidden) Admit(attrs *PodAdmitAttributes) PodAdmitResult {
	pod := attrs.Pod
	errList := w.checkHostpathSymlink(pod)
	return PodAdmitResult{
		Admit:   len(errList) == 0,
		Reason:  "UnexpectedAdmissionError",
		Message: strings.Join(errList, "\n"),
	}
}

func (sf *hostpathSymlinkForbidden) checkHostpathSymlink(pod *v1.Pod) []string {
	errList := make([]string, 0)
	for _, vol := range pod.Spec.Volumes {
		if vol.VolumeSource.HostPath != nil {
			// if hostpath is not found, kubelet will create directory/file according hostpath type, skip
			// NOTICE: if vol.HostPath.Path does not exist, checkSymlink will return [nil, false]
			// if hostpath is a symlink, add err to errlist
			// if checkSymlink got a error, add err to errlist
			err, symlink := checkSymlink(vol.HostPath.Path)
			if err != nil {
				errList = append(errList, err.Error())
			} else if symlink {
				errList = append(errList, fmt.Sprintf("[hostpath] %v is a symlink which is not allowed.", vol.HostPath.Path))
			}
		}
	}
	return errList
}

func checkSymlink(path string) (error, bool) {
	fileInfo, err := os.Lstat(path)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, false
		}
		return err, false
	}
	if (fileInfo.Mode() & os.ModeSymlink) != 0 {
		return nil, true
	}
	return err, false
}
