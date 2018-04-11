/*
Copyright 2018 The Kubernetes Authors.

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

/*
Package factory implements a set of object with the responsibility to create
accessory components used during the execution of kubeadm command, and once
components are created, to store this components for reuse during the command
flow.

A common use case for factories is to share components instances across phases
executed in a command workflow.

Important! factories are implemented as Singleton without support for thread-safety.
*/
package factory
