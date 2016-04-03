/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package mesosutil

import (
	"github.com/gogo/protobuf/proto"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/stretchr/testify/assert"
	"testing"
)

func TestFilterResources(t *testing.T) {
	resources := []*mesos.Resource{
		NewScalarResource("mem", 200),
		NewScalarResource("cpu", 4),
		NewScalarResource("mem", 500),
	}

	memRes := FilterResources(resources, func(res *mesos.Resource) bool {
		if res.GetType() == mesos.Value_SCALAR && res.GetName() == "mem" {
			return true
		}
		return false
	})

	assert.Equal(t, 2, len(memRes))
}

func TestNewValueRange(t *testing.T) {
	val := NewValueRange(20, 40)
	if val == nil {
		t.Fatal("Not creating protobuf object Value_Range.")
	}

	if (val.GetEnd() - val.GetBegin()) != 20 {
		t.Fatal("Protobuf object Value_Range not returning expected values.")
	}
}

func TestNewScalarResource(t *testing.T) {
	val := NewScalarResource("mem", 200)
	if val == nil {
		t.Fatal("Not creating protobuf object Resource properly.")
	}
	if val.GetType() != mesos.Value_SCALAR {
		t.Fatal("Expected type SCALAR for protobuf, got", val.GetType())
	}
	if val.GetName() != "mem" && val.GetScalar().GetValue() != 200 {
		t.Fatal("Protobuf object Resource has wrong name and Scalar values.")
	}
}

func TestNewRangesResource(t *testing.T) {
	val := NewRangesResource("quotas", []*mesos.Value_Range{NewValueRange(20, 40)})
	if val == nil {
		t.Fatal("Not creating protobuf object Resource properly.")
	}
	if val.GetType() != mesos.Value_RANGES {
		t.Fatal("Expected type SCALAR for protobuf, got", val.GetType())
	}
	if len(val.GetRanges().GetRange()) != 1 {
		t.Fatal("Expected Resource of type RANGES with 1 range, but got", len(val.GetRanges().GetRange()))
	}

}

func TestNewSetResource(t *testing.T) {
	val := NewSetResource("greeting", []string{"hello", "world"})
	if val == nil {
		t.Fatal("Not creating protobuf object Resource properly.")
	}
	if val.GetType() != mesos.Value_SET {
		t.Fatal("Expected type SET for protobuf, got", val.GetType())
	}
	if len(val.GetSet().GetItem()) != 2 {
		t.Fatal("Expected Resource of type SET with 2 items, but got", len(val.GetRanges().GetRange()))
	}
	if val.GetSet().GetItem()[0] != "hello" {
		t.Fatal("Protobuf Resource of type SET got wrong value.")
	}
}

func TestNewFrameworkID(t *testing.T) {
	id := NewFrameworkID("test-id")
	if id == nil {
		t.Fatal("Not creating protobuf oject FrameworkID.")
	}
	if id.GetValue() != "test-id" {
		t.Fatal("Protobuf object not returning expected value.")
	}
}

func TestNewFrameworkInfo(t *testing.T) {
	info := NewFrameworkInfo("test-user", "test-name", NewFrameworkID("test-id"))
	info.Hostname = proto.String("localhost")
	if info == nil {
		t.Fatal("Not creating protobuf object FrameworkInfo")
	}
	if info.GetUser() != "test-user" {
		t.Fatal("Protobuf object FrameworkInfo.User missing value.")
	}
	if info.GetName() != "test-name" {
		t.Fatal("Protobuf object FrameworkInfo.Name missing value.")
	}
	if info.GetId() == nil {
		t.Fatal("Protobuf object FrameowrkInfo.Id missing value.")
	}
	if info.GetHostname() != "localhost" {
		t.Fatal("Protobuf object FrameworkInfo.Hostname missing value.")
	}
}

func TestNewMasterInfo(t *testing.T) {
	master := NewMasterInfo("master-1", 1234, 5678)
	if master == nil {
		t.Fatal("Not creating protobuf object MasterInfo")
	}
	if master.GetId() != "master-1" {
		t.Fatal("Protobuf object MasterInfo.Id missing.")
	}
	if master.GetIp() != 1234 {
		t.Fatal("Protobuf object MasterInfo.Ip missing.")
	}
	if master.GetPort() != 5678 {
		t.Fatal("Protobuf object MasterInfo.Port missing.")
	}
}

func TestNewOfferID(t *testing.T) {
	id := NewOfferID("offer-1")
	if id == nil {
		t.Fatal("Not creating protobuf object OfferID")
	}
	if id.GetValue() != "offer-1" {
		t.Fatal("Protobuf object OfferID.Value missing.")
	}
}

func TestNewOffer(t *testing.T) {
	offer := NewOffer(NewOfferID("offer-1"), NewFrameworkID("framework-1"), NewSlaveID("slave-1"), "localhost")
	if offer == nil {
		t.Fatal("Not creating protobuf object Offer")
	}
	if offer.GetId().GetValue() != "offer-1" {
		t.Fatal("Protobuf object Offer.Id missing")
	}
	if offer.GetFrameworkId().GetValue() != "framework-1" {
		t.Fatal("Protobuf object Offer.FrameworkId missing.")
	}
	if offer.GetSlaveId().GetValue() != "slave-1" {
		t.Fatal("Protobuf object Offer.SlaveId missing.")
	}
	if offer.GetHostname() != "localhost" {
		t.Fatal("Protobuf object offer.Hostname missing.")
	}
}

func TestNewSlaveID(t *testing.T) {
	id := NewSlaveID("slave-1")
	if id == nil {
		t.Fatal("Not creating protobuf object SlaveID")
	}
	if id.GetValue() != "slave-1" {
		t.Fatal("Protobuf object SlaveID.Value missing.")
	}
}

func TestNewTaskID(t *testing.T) {
	id := NewSlaveID("task-1")
	if id == nil {
		t.Fatal("Not creating protobuf object TaskID")
	}
	if id.GetValue() != "task-1" {
		t.Fatal("Protobuf object TaskID.Value missing.")
	}
}

func TestNewTaskInfo(t *testing.T) {
	info := NewTaskInfo(
		"simple-task",
		NewTaskID("simpe-task-1"),
		NewSlaveID("slave-1"),
		[]*mesos.Resource{NewScalarResource("mem", 400)},
	)
	if info == nil {
		t.Fatal("Not creating protobuf object TaskInfo")
	}
	if info.GetName() != "simple-task" {
		t.Fatal("Protobuf object TaskInfo.Name missing.")
	}
	if info.GetTaskId() == nil {
		t.Fatal("Protobuf object TaskInfo.TaskId missing.")
	}
	if info.GetSlaveId() == nil {
		t.Fatal("Protobuf object TaskInfo.SlaveId missing.")
	}
	if len(info.GetResources()) != 1 {
		t.Fatal("Protobuf object TaskInfo.Resources missing.")
	}
}

func TestNewTaskStatus(t *testing.T) {
	status := NewTaskStatus(NewTaskID("task-1"), mesos.TaskState_TASK_RUNNING)
	if status == nil {
		t.Fatal("Not creating protobuf object TaskStatus")
	}
	if status.GetTaskId().GetValue() != "task-1" {
		t.Fatal("Protobuf object TaskStatus.TaskId missing.")
	}
	if status.GetState() != mesos.TaskState(mesos.TaskState_TASK_RUNNING) {
		t.Fatal("Protobuf object TaskStatus.State missing.")
	}
}

func TestNewCommandInfo(t *testing.T) {
	cmd := NewCommandInfo("echo Hello!")
	if cmd == nil {
		t.Fatal("Not creating protobuf object CommandInfo")
	}
	if cmd.GetValue() != "echo Hello!" {
		t.Fatal("Protobuf object CommandInfo.Value missing")
	}
}

func TestNewExecutorInfo(t *testing.T) {
	info := NewExecutorInfo(NewExecutorID("exec-1"), NewCommandInfo("ls -l"))
	if info == nil {
		t.Fatal("Not creating protobuf object ExecutorInfo")
	}
	if info.GetExecutorId().GetValue() != "exec-1" {
		t.Fatal("Protobuf object ExecutorInfo.ExecutorId missing")
	}
	if info.GetCommand().GetValue() != "ls -l" {
		t.Fatal("Protobuf object ExecutorInfo.Command missing")
	}
}
