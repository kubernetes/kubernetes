/*
Copyright 2017 The Kubernetes Authors.

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

package apps

import (
	"fmt"

	"k8s.io/apimachinery/pkg/runtime/schema"
)

// KindVisitor is used with GroupKindElement to call a particular function depending on the
// Kind of a schema.GroupKind
type KindVisitor interface {
	VisitDaemonSet(kind GroupKindElement)
	VisitDeployment(kind GroupKindElement)
	VisitJob(kind GroupKindElement)
	VisitPod(kind GroupKindElement)
	VisitReplicaSet(kind GroupKindElement)
	VisitReplicationController(kind GroupKindElement)
	VisitStatefulSet(kind GroupKindElement)
}

// GroupKindElement defines a Kubernetes API group elem
type GroupKindElement schema.GroupKind

// Accept calls the Visit method on visitor that corresponds to elem's Kind
func (elem GroupKindElement) Accept(visitor KindVisitor) error {
	if elem.GroupMatch("apps", "extensions") && elem.Kind == "DaemonSet" {
		visitor.VisitDaemonSet(elem)
		return nil
	}
	if elem.GroupMatch("apps", "extensions") && elem.Kind == "Deployment" {
		visitor.VisitDeployment(elem)
		return nil
	}
	if elem.GroupMatch("batch") && elem.Kind == "Job" {
		visitor.VisitJob(elem)
		return nil
	}
	if elem.GroupMatch("", "core") && elem.Kind == "Pod" {
		visitor.VisitPod(elem)
		return nil
	}
	if elem.GroupMatch("extensions") && elem.Kind == "ReplicaSet" {
		visitor.VisitReplicaSet(elem)
		return nil
	}
	if elem.GroupMatch("", "core") && elem.Kind == "ReplicationController" {
		visitor.VisitReplicationController(elem)
		return nil
	}
	if elem.GroupMatch("apps") && elem.Kind == "StatefulSet" {
		visitor.VisitStatefulSet(elem)
		return nil
	}
	return fmt.Errorf("no visitor method exists for %v", elem)
}

// GroupMatch returns true if and only if elem's group matches one
// of the group arguments
func (elem GroupKindElement) GroupMatch(groups ...string) bool {
	for _, g := range groups {
		if elem.Group == g {
			return true
		}
	}
	return false
}

// NoOpKindVisitor implements KindVisitor with no-op functions.
type NoOpKindVisitor struct{}

var _ KindVisitor = &NoOpKindVisitor{}

func (*NoOpKindVisitor) VisitDaemonSet(kind GroupKindElement)             {}
func (*NoOpKindVisitor) VisitDeployment(kind GroupKindElement)            {}
func (*NoOpKindVisitor) VisitJob(kind GroupKindElement)                   {}
func (*NoOpKindVisitor) VisitPod(kind GroupKindElement)                   {}
func (*NoOpKindVisitor) VisitReplicaSet(kind GroupKindElement)            {}
func (*NoOpKindVisitor) VisitReplicationController(kind GroupKindElement) {}
func (*NoOpKindVisitor) VisitStatefulSet(kind GroupKindElement)           {}
