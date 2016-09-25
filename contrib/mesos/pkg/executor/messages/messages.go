/*
Copyright 2015 The Kubernetes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package messages

// messages that ship with TaskStatus objects

const (
	ContainersDisappeared    = "containers-disappeared"
	CreateBindingFailure     = "create-binding-failure"
	CreateBindingSuccess     = "create-binding-success"
	ExecutorUnregistered     = "executor-unregistered"
	ExecutorShutdown         = "executor-shutdown"
	LaunchTaskFailed         = "launch-task-failed"
	KubeletPodLaunchFailed   = "kubelet-pod-launch-failed"
	TaskKilled               = "task-killed"
	TaskLost                 = "task-lost"
	UnmarshalTaskDataFailure = "unmarshal-task-data-failure"
	TaskLostAck              = "task-lost-ack" // executor acknowledgment of forwarded TASK_LOST framework message
	Kamikaze                 = "kamikaze"
	WrongSlaveFailure        = "pod-for-wrong-slave-failure"
	AnnotationUpdateFailure  = "annotation-update-failure"
)
