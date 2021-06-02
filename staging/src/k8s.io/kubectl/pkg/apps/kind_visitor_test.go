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
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

var _ = Describe("When KindVisitor accepts a GroupKind", func() {

	var visitor *TestKindVisitor

	BeforeEach(func() {
		visitor = &TestKindVisitor{map[string]int{}}
	})

	It("should Visit DaemonSet iff the Kind is a DaemonSet", func() {
		kind := GroupKindElement{
			Kind:  "DaemonSet",
			Group: "apps",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"DaemonSet": 1,
		}))

		kind = GroupKindElement{
			Kind:  "DaemonSet",
			Group: "extensions",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"DaemonSet": 2,
		}))
	})

	It("should Visit Deployment iff the Kind is a Deployment", func() {
		kind := GroupKindElement{
			Kind:  "Deployment",
			Group: "apps",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"Deployment": 1,
		}))

		kind = GroupKindElement{
			Kind:  "Deployment",
			Group: "extensions",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"Deployment": 2,
		}))
	})

	It("should Visit Job iff the Kind is a Job", func() {
		kind := GroupKindElement{
			Kind:  "Job",
			Group: "batch",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"Job": 1,
		}))

	})

	It("should Visit Pod iff the Kind is a Pod", func() {
		kind := GroupKindElement{
			Kind:  "Pod",
			Group: "",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"Pod": 1,
		}))

		kind = GroupKindElement{
			Kind:  "Pod",
			Group: "core",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"Pod": 2,
		}))
	})

	It("should Visit ReplicationController iff the Kind is a ReplicationController", func() {
		kind := GroupKindElement{
			Kind:  "ReplicationController",
			Group: "",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"ReplicationController": 1,
		}))

		kind = GroupKindElement{
			Kind:  "ReplicationController",
			Group: "core",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"ReplicationController": 2,
		}))
	})

	It("should Visit ReplicaSet iff the Kind is a ReplicaSet", func() {
		kind := GroupKindElement{
			Kind:  "ReplicaSet",
			Group: "extensions",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"ReplicaSet": 1,
		}))
	})

	It("should Visit StatefulSet iff the Kind is a StatefulSet", func() {
		kind := GroupKindElement{
			Kind:  "StatefulSet",
			Group: "apps",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"StatefulSet": 1,
		}))
	})

	It("should Visit CronJob iff the Kind is a CronJob", func() {
		kind := GroupKindElement{
			Kind:  "CronJob",
			Group: "batch",
		}
		Expect(kind.Accept(visitor)).ShouldNot(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{
			"CronJob": 1,
		}))
	})

	It("should give an error if the Kind is unknown", func() {
		kind := GroupKindElement{
			Kind:  "Unknown",
			Group: "apps",
		}
		Expect(kind.Accept(visitor)).Should(HaveOccurred())
		Expect(visitor.visits).To(Equal(map[string]int{}))
	})
})

// TestKindVisitor increments a value each time a Visit method was called
type TestKindVisitor struct {
	visits map[string]int
}

var _ KindVisitor = &TestKindVisitor{}

func (t *TestKindVisitor) Visit(kind GroupKindElement) { t.visits[kind.Kind]++ }

func (t *TestKindVisitor) VisitDaemonSet(kind GroupKindElement)             { t.Visit(kind) }
func (t *TestKindVisitor) VisitDeployment(kind GroupKindElement)            { t.Visit(kind) }
func (t *TestKindVisitor) VisitJob(kind GroupKindElement)                   { t.Visit(kind) }
func (t *TestKindVisitor) VisitPod(kind GroupKindElement)                   { t.Visit(kind) }
func (t *TestKindVisitor) VisitReplicaSet(kind GroupKindElement)            { t.Visit(kind) }
func (t *TestKindVisitor) VisitReplicationController(kind GroupKindElement) { t.Visit(kind) }
func (t *TestKindVisitor) VisitStatefulSet(kind GroupKindElement)           { t.Visit(kind) }
func (t *TestKindVisitor) VisitCronJob(kind GroupKindElement)               { t.Visit(kind) }
