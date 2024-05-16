/*
Copyright 2014 The Kubernetes Authors.

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

// Package config provides decoupling between various configuration sources (etcd, files,...) and
// the pieces that actually care about them (loadbalancer, proxy). Config takes 1 or more
// configuration sources and allows for incremental (add/remove) and full replace (set)
// changes from each of the sources, then creates a union of the configuration and provides
// a unified view for both service handlers as well as endpoint handlers. There is no attempt
// to resolve conflicts of any sort. Basic idea is that each configuration source gets a channel
// from the Config service and pushes updates to it via that channel. Config then keeps track of
// incremental & replace changes and distributes them to listeners as appropriate.
package config // import "k8s.io/kubernetes/pkg/proxy/config"
