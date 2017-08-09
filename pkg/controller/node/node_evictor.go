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

package node

type nodeEvictor interface {
}

// taintBasedNodeEvictor satisfies the Evictor interface
type taintBasedNodeEvictor struct {
	nc *Controller
}

// defaultNodeEvictor satisfies the Evictor interface
type defaultNodeEvictor struct {
	nc *Controller
}

func newTaintBasedNodeEvictor(nc *Controller) *taintBasedNodeEvictor {
	return &taintBasedNodeEvictor{nc}
}

func newDefaultNodeEvictor(nc *Controller) *defaultNodeEvictor {
	return &defaultNodeEvictor{nc}
}
