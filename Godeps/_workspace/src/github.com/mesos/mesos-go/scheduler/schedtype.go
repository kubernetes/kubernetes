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

package scheduler

import (
	mesos "github.com/mesos/mesos-go/mesosproto"
)

// Interface for connecting a scheduler to Mesos. This
// interface is used both to manage the scheduler's lifecycle (start
// it, stop it, or wait for it to finish) and to interact with Mesos
// (e.g., launch tasks, kill tasks, etc.).
// See the MesosSchedulerDriver type for a concrete
// impl of a SchedulerDriver.
type SchedulerDriver interface {
	// Starts the scheduler driver. This needs to be called before any
	// other driver calls are made.
	Start() (mesos.Status, error)

	// Stops the scheduler driver. If the 'failover' flag is set to
	// false then it is expected that this framework will never
	// reconnect to Mesos and all of its executors and tasks can be
	// terminated. Otherwise, all executors and tasks will remain
	// running (for some framework specific failover timeout) allowing the
	// scheduler to reconnect (possibly in the same process, or from a
	// different process, for example, on a different machine).
	Stop(failover bool) (mesos.Status, error)

	// Aborts the driver so that no more callbacks can be made to the
	// scheduler. The semantics of abort and stop have deliberately been
	// separated so that code can detect an aborted driver (i.e., via
	// the return status of SchedulerDriver::join, see below), and
	// instantiate and start another driver if desired (from within the
	// same process). Note that 'Stop()' is not automatically called
	// inside 'Abort()'.
	Abort() (mesos.Status, error)

	// Waits for the driver to be stopped or aborted, possibly
	// _blocking_ the current thread indefinitely. The return status of
	// this function can be used to determine if the driver was aborted
	// (see mesos.proto for a description of Status).
	Join() (mesos.Status, error)

	// Starts and immediately joins (i.e., blocks on) the driver.
	Run() (mesos.Status, error)

	// Requests resources from Mesos (see mesos.proto for a description
	// of Request and how, for example, to request resources
	// from specific slaves). Any resources available are offered to the
	// framework via Scheduler.ResourceOffers callback, asynchronously.
	RequestResources(requests []*mesos.Request) (mesos.Status, error)

	// AcceptOffers utilizes the new HTTP API to send a Scheduler Call Message
	// to the Mesos Master. Valid operation types are LAUNCH, RESERVE, UNRESERVE,
	// CREATE, DESTROY, and more.
	AcceptOffers(offerIDs []*mesos.OfferID, operations []*mesos.Offer_Operation, filters *mesos.Filters) (mesos.Status, error)

	// Launches the given set of tasks. Any resources remaining (i.e.,
	// not used by the tasks or their executors) will be considered
	// declined. The specified filters are applied on all unused
	// resources (see mesos.proto for a description of Filters).
	// Available resources are aggregated when mutiple offers are
	// provided. Note that all offers must belong to the same slave.
	// Invoking this function with an empty collection of tasks declines
	// offers in their entirety (see Scheduler::declineOffer).
	LaunchTasks(offerIDs []*mesos.OfferID, tasks []*mesos.TaskInfo, filters *mesos.Filters) (mesos.Status, error)

	// Kills the specified task. Note that attempting to kill a task is
	// currently not reliable. If, for example, a scheduler fails over
	// while it was attempting to kill a task it will need to retry in
	// the future. Likewise, if unregistered / disconnected, the request
	// will be dropped (these semantics may be changed in the future).
	KillTask(taskID *mesos.TaskID) (mesos.Status, error)

	// Declines an offer in its entirety and applies the specified
	// filters on the resources (see mesos.proto for a description of
	// Filters). Note that this can be done at any time, it is not
	// necessary to do this within the Scheduler::resourceOffers
	// callback.
	DeclineOffer(offerID *mesos.OfferID, filters *mesos.Filters) (mesos.Status, error)

	// Removes all filters previously set by the framework (via
	// LaunchTasks()). This enables the framework to receive offers from
	// those filtered slaves.
	ReviveOffers() (mesos.Status, error)

	// Sends a message from the framework to one of its executors. These
	// messages are best effort; do not expect a framework message to be
	// retransmitted in any reliable fashion.
	SendFrameworkMessage(executorID *mesos.ExecutorID, slaveID *mesos.SlaveID, data string) (mesos.Status, error)

	// Allows the framework to query the status for non-terminal tasks.
	// This causes the master to send back the latest task status for
	// each task in 'statuses', if possible. Tasks that are no longer
	// known will result in a TASK_LOST update. If statuses is empty,
	// then the master will send the latest status for each task
	// currently known.
	ReconcileTasks(statuses []*mesos.TaskStatus) (mesos.Status, error)
}

// Scheduler a type with callback attributes to be provided by frameworks
// schedulers.
//
// Each callback includes a reference to the scheduler driver that was
// used to run this scheduler. The pointer will not change for the
// duration of a scheduler (i.e., from the point you do
// SchedulerDriver.Start() to the point that SchedulerDriver.Stop()
// returns). This is intended for convenience so that a scheduler
// doesn't need to store a reference to the driver itself.
type Scheduler interface {

	// Invoked when the scheduler successfully registers with a Mesos
	// master. A unique ID (generated by the master) used for
	// distinguishing this framework from others and MasterInfo
	// with the ip and port of the current master are provided as arguments.
	Registered(SchedulerDriver, *mesos.FrameworkID, *mesos.MasterInfo)

	// Invoked when the scheduler re-registers with a newly elected Mesos master.
	// This is only called when the scheduler has previously been registered.
	// MasterInfo containing the updated information about the elected master
	// is provided as an argument.
	Reregistered(SchedulerDriver, *mesos.MasterInfo)

	// Invoked when the scheduler becomes "disconnected" from the master
	// (e.g., the master fails and another is taking over).
	Disconnected(SchedulerDriver)

	// Invoked when resources have been offered to this framework. A
	// single offer will only contain resources from a single slave.
	// Resources associated with an offer will not be re-offered to
	// _this_ framework until either (a) this framework has rejected
	// those resources (see SchedulerDriver::launchTasks) or (b) those
	// resources have been rescinded (see Scheduler::offerRescinded).
	// Note that resources may be concurrently offered to more than one
	// framework at a time (depending on the allocator being used). In
	// that case, the first framework to launch tasks using those
	// resources will be able to use them while the other frameworks
	// will have those resources rescinded (or if a framework has
	// already launched tasks with those resources then those tasks will
	// fail with a TASK_LOST status and a message saying as much).
	ResourceOffers(SchedulerDriver, []*mesos.Offer)

	// Invoked when an offer is no longer valid (e.g., the slave was
	// lost or another framework used resources in the offer). If for
	// whatever reason an offer is never rescinded (e.g., dropped
	// message, failing over framework, etc.), a framwork that attempts
	// to launch tasks using an invalid offer will receive TASK_LOST
	// status updates for those tasks (see Scheduler::resourceOffers).
	OfferRescinded(SchedulerDriver, *mesos.OfferID)

	// Invoked when the status of a task has changed (e.g., a slave is
	// lost and so the task is lost, a task finishes and an executor
	// sends a status update saying so, etc). Note that returning from
	// this callback _acknowledges_ receipt of this status update! If
	// for whatever reason the scheduler aborts during this callback (or
	// the process exits) another status update will be delivered (note,
	// however, that this is currently not true if the slave sending the
	// status update is lost/fails during that time).
	StatusUpdate(SchedulerDriver, *mesos.TaskStatus)

	// Invoked when an executor sends a message. These messages are best
	// effort; do not expect a framework message to be retransmitted in
	// any reliable fashion.
	FrameworkMessage(SchedulerDriver, *mesos.ExecutorID, *mesos.SlaveID, string)

	// Invoked when a slave has been determined unreachable (e.g.,
	// machine failure, network partition). Most frameworks will need to
	// reschedule any tasks launched on this slave on a new slave.
	SlaveLost(SchedulerDriver, *mesos.SlaveID)

	// Invoked when an executor has exited/terminated. Note that any
	// tasks running will have TASK_LOST status updates automagically
	// generated.
	ExecutorLost(SchedulerDriver, *mesos.ExecutorID, *mesos.SlaveID, int)

	// Invoked when there is an unrecoverable error in the scheduler or
	// scheduler driver. The driver will be aborted BEFORE invoking this
	// callback.
	Error(SchedulerDriver, string)
}
