package noop

import (
	"github.com/golang/glog"
	"k8s.io/kubernetes/pkg/api/v1"
	"k8s.io/kubernetes/pkg/kubelet/api/v1alpha1/lifecycle"
)

type noopIsolator struct{ name string }

func (i *noopIsolator) Init() error {
	glog.Infof("noopIsolator[Init]")
	return nil
}

func New(name string) (*noopIsolator, error) {
	return &noopIsolator{name: name}, nil
}

func (i *noopIsolator) PreStartPod(podName string, containerName string, pod *v1.Pod, cgroupInfo *lifecycle.CgroupInfo) ([]*lifecycle.IsolationControl, error) {
	glog.Infof("noopIsolator[PreStartPod]:\npod: %s\ncgroupInfo: %v", pod, cgroupInfo)
	return []*lifecycle.IsolationControl{}, nil
}

func (i *noopIsolator) PostStopPod(podName string, containerName string, cgroupInfo *lifecycle.CgroupInfo) error {
	glog.Infof("noopIsolator[PostStopPod]:\npodName: %s\ncontainerName: %s\ncgroupInfo: %v", podName, containerName, cgroupInfo)
	return nil
}

func (i *noopIsolator) PreStartContainer(podName, containerName string) ([]*lifecycle.IsolationControl, error) {
	glog.Infof("noopIsolator[PreStartContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return []*lifecycle.IsolationControl{
		&lifecycle.IsolationControl{
			Kind: lifecycle.IsolationControl_CONTAINER_ENV_VAR,
			MapValue: map[string]string{
				"ISOLATOR": i.Name(),
			},
		},
	}, nil
}

func (i *noopIsolator) PostStopContainer(podName, containerName string) error {
	glog.Infof("noopIsolator[PostStopContainer]:\npodName: %s\ncontainerName: %v", podName, containerName)
	return nil
}

func (i *noopIsolator) ShutDown() {
	glog.Infof("noopIsolator[ShutDown]")
}

func (i *noopIsolator) Name() string {
	return i.name
}
