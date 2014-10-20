/*
Copyright 2014 Google Inc. All rights reserved.

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

package ports

const (
	// KubeletPort is the default port for the kubelet status server on each host machine.
	// May be overridden by a flag at startup.
	KubeletPort = 10250
	// SchedulerPort is the default port for the scheduler status server.
	// May be overridden by a flag at startup.
	SchedulerPort = 10251
	// ControllerManagerPort is the default port for the controller manager status server.
	// May be overridden by a flag at startup.
	ControllerManagerPort = 10252
)
