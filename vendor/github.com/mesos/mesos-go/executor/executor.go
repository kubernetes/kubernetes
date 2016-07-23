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

package executor

import (
	"fmt"
	"net"
	"os"
	"sync"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/mesosutil/process"
	"github.com/mesos/mesos-go/messenger"
	"github.com/mesos/mesos-go/messenger/sessionid"
	"github.com/mesos/mesos-go/upid"
	"github.com/pborman/uuid"
	"golang.org/x/net/context"
)

const (
	defaultRecoveryTimeout = 15 * time.Minute
)

type DriverConfig struct {
	Executor         Executor
	HostnameOverride string                              // optional
	BindingAddress   net.IP                              // optional
	BindingPort      uint16                              // optional
	PublishedAddress net.IP                              // optional
	NewMessenger     func() (messenger.Messenger, error) // optional
}

// MesosExecutorDriver is a implementation of the ExecutorDriver.
type MesosExecutorDriver struct {
	lock            sync.RWMutex
	cond            *sync.Cond
	self            *upid.UPID
	stopCh          chan struct{}
	status          mesosproto.Status
	messenger       messenger.Messenger
	slaveUPID       *upid.UPID
	slaveID         *mesosproto.SlaveID
	frameworkID     *mesosproto.FrameworkID
	executorID      *mesosproto.ExecutorID
	workDir         string
	connected       bool
	connection      uuid.UUID
	local           bool   // TODO(yifan): Not used yet.
	directory       string // TODO(yifan): Not used yet.
	checkpoint      bool
	recoveryTimeout time.Duration
	recoveryTimer   *time.Timer
	updates         map[string]*mesosproto.StatusUpdate // Key is a UUID string. TODO(yifan): Not used yet.
	tasks           map[string]*mesosproto.TaskInfo     // Key is a UUID string. TODO(yifan): Not used yet.
	withExecutor    func(f func(e Executor))
	started         chan struct{} // signal chan that closes once start has been invoked
}

// NewMesosExecutorDriver creates a new mesos executor driver.
func NewMesosExecutorDriver(config DriverConfig) (*MesosExecutorDriver, error) {
	if config.Executor == nil {
		msg := "Executor callback interface cannot be nil."
		log.Errorln(msg)
		return nil, fmt.Errorf(msg)
	}

	hostname := mesosutil.GetHostname(config.HostnameOverride)
	newMessenger := config.NewMessenger
	if newMessenger == nil {
		newMessenger = func() (messenger.Messenger, error) {
			process := process.New("executor")
			return messenger.ForHostname(process, hostname, config.BindingAddress, config.BindingPort, config.PublishedAddress)
		}
	}

	driver := &MesosExecutorDriver{
		status:          mesosproto.Status_DRIVER_NOT_STARTED,
		stopCh:          make(chan struct{}),
		updates:         make(map[string]*mesosproto.StatusUpdate),
		tasks:           make(map[string]*mesosproto.TaskInfo),
		workDir:         ".",
		started:         make(chan struct{}),
		recoveryTimeout: defaultRecoveryTimeout,
	}
	driver.cond = sync.NewCond(&driver.lock)
	// decouple serialized executor callback execution from goroutines of this driver
	var execLock sync.Mutex
	driver.withExecutor = func(f func(e Executor)) {
		go func() {
			execLock.Lock()
			defer execLock.Unlock()
			f(config.Executor)
		}()
	}
	var err error
	if driver.messenger, err = newMessenger(); err != nil {
		return nil, err
	}
	if err = driver.init(); err != nil {
		log.Errorf("failed to initialize the driver: %v", err)
		return nil, err
	}
	return driver, nil
}

// context returns the driver context, expects driver.lock to be locked
func (driver *MesosExecutorDriver) context() context.Context {
	return sessionid.NewContext(context.TODO(), driver.connection.String())
}

// init initializes the driver.
func (driver *MesosExecutorDriver) init() error {
	log.Infof("Init mesos executor driver\n")
	log.Infof("Protocol Version: %v\n", mesosutil.MesosVersion)

	// Parse environments.
	if err := driver.parseEnviroments(); err != nil {
		log.Errorf("Failed to parse environments: %v\n", err)
		return err
	}

	type messageHandler func(context.Context, *upid.UPID, proto.Message)

	guard := func(h messageHandler) messenger.MessageHandler {
		return messenger.MessageHandler(func(from *upid.UPID, pbMsg proto.Message) {
			driver.lock.Lock()
			defer driver.lock.Unlock()
			h(driver.context(), from, pbMsg)
		})
	}

	// Install handlers.
	driver.messenger.Install(guard(driver.registered), &mesosproto.ExecutorRegisteredMessage{})
	driver.messenger.Install(guard(driver.reregistered), &mesosproto.ExecutorReregisteredMessage{})
	driver.messenger.Install(guard(driver.reconnect), &mesosproto.ReconnectExecutorMessage{})
	driver.messenger.Install(guard(driver.runTask), &mesosproto.RunTaskMessage{})
	driver.messenger.Install(guard(driver.killTask), &mesosproto.KillTaskMessage{})
	driver.messenger.Install(guard(driver.statusUpdateAcknowledgement), &mesosproto.StatusUpdateAcknowledgementMessage{})
	driver.messenger.Install(guard(driver.frameworkMessage), &mesosproto.FrameworkToExecutorMessage{})
	driver.messenger.Install(guard(driver.shutdown), &mesosproto.ShutdownExecutorMessage{})
	driver.messenger.Install(guard(driver.frameworkError), &mesosproto.FrameworkErrorMessage{})
	driver.messenger.Install(guard(driver.networkError), &mesosproto.InternalNetworkError{})
	return nil
}

func (driver *MesosExecutorDriver) parseEnviroments() error {
	var value string

	value = os.Getenv("MESOS_LOCAL")
	if len(value) > 0 {
		driver.local = true
	}

	value = os.Getenv("MESOS_SLAVE_PID")
	if len(value) == 0 {
		return fmt.Errorf("Cannot find MESOS_SLAVE_PID in the environment")
	}
	upid, err := upid.Parse(value)
	if err != nil {
		log.Errorf("Cannot parse UPID %v\n", err)
		return err
	}
	driver.slaveUPID = upid

	value = os.Getenv("MESOS_SLAVE_ID")
	driver.slaveID = &mesosproto.SlaveID{Value: proto.String(value)}

	value = os.Getenv("MESOS_FRAMEWORK_ID")
	driver.frameworkID = &mesosproto.FrameworkID{Value: proto.String(value)}

	value = os.Getenv("MESOS_EXECUTOR_ID")
	driver.executorID = &mesosproto.ExecutorID{Value: proto.String(value)}

	value = os.Getenv("MESOS_DIRECTORY")
	if len(value) > 0 {
		driver.workDir = value
	}

	value = os.Getenv("MESOS_CHECKPOINT")
	if value == "1" {
		driver.checkpoint = true
	}

	/*
		if driver.checkpoint {
			value = os.Getenv("MESOS_RECOVERY_TIMEOUT")
		}
		// TODO(yifan): Parse the duration. For now just use default.
	*/
	return nil
}

// ------------------------- Accessors ----------------------- //
func (driver *MesosExecutorDriver) Status() mesosproto.Status {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.status
}

func (driver *MesosExecutorDriver) Running() bool {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.status == mesosproto.Status_DRIVER_RUNNING
}

func (driver *MesosExecutorDriver) stopped() bool {
	return driver.status != mesosproto.Status_DRIVER_RUNNING
}

func (driver *MesosExecutorDriver) Connected() bool {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.connected
}

// --------------------- Message Handlers --------------------- //

// networkError is invoked when there's a network-level error communicating with the mesos slave.
// The driver reacts by entering a "disconnected" state and invoking the Executor.Disconnected
// callback. The assumption is that if this driver was previously connected, and then there's a
// network error, then the slave process must be dying/dead. The native driver implementation makes
// this same assumption. I have some concerns that this may be a false-positive in some situations;
// some network errors (timeouts) may be indicative of something other than a dead slave process.
func (driver *MesosExecutorDriver) networkError(ctx context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Info("ignoring network error because aborted")
		return
	}
	if driver.connected {
		driver.connected = false
		msg := pbMsg.(*mesosproto.InternalNetworkError)
		session := msg.GetSession()

		if session != driver.connection.String() {
			log.V(1).Infoln("ignoring netwok error for disconnected/stale session")
			return
		}

		if driver.checkpoint {
			log.Infoln("slave disconnected, will wait for recovery")
			driver.withExecutor(func(e Executor) { e.Disconnected(driver) })

			if driver.recoveryTimer != nil {
				driver.recoveryTimer.Stop()
			}
			t := time.NewTimer(driver.recoveryTimeout)
			driver.recoveryTimer = t
			go func() {
				select {
				case <-t.C:
					// timer expired
					driver.lock.Lock()
					defer driver.lock.Unlock()
					driver.recoveryTimedOut(session)

				case <-driver.stopCh:
					// driver stopped
					return
				}
			}()
			return
		}
	}
	log.Infoln("slave exited ... shutting down")
	driver.withExecutor(func(e Executor) { e.Shutdown(driver) }) // abnormal shutdown
	driver.abort()
}

func (driver *MesosExecutorDriver) recoveryTimedOut(connection string) {
	if driver.connected {
		return
	}
	// ensure that connection ID's match otherwise we've been re-registered
	if connection == driver.connection.String() {
		log.Info("recovery timeout of %v exceeded; shutting down", driver.recoveryTimeout)
		driver.shutdown(driver.context(), nil, nil)
	}
}

func (driver *MesosExecutorDriver) registered(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring registration message from slave because aborted")
		return
	}
	log.Infoln("Executor driver registered")

	msg := pbMsg.(*mesosproto.ExecutorRegisteredMessage)
	slaveID := msg.GetSlaveId()
	executorInfo := msg.GetExecutorInfo()
	frameworkInfo := msg.GetFrameworkInfo()
	slaveInfo := msg.GetSlaveInfo()

	if driver.stopped() {
		log.Infof("Ignoring registered message from slave %v, because the driver is stopped!\n", slaveID)
		return
	}

	log.Infof("Registered on slave %v\n", slaveID)
	driver.connected = true
	driver.connection = uuid.NewUUID()
	driver.cond.Broadcast() // useful for testing
	driver.withExecutor(func(e Executor) { e.Registered(driver, executorInfo, frameworkInfo, slaveInfo) })
}

func (driver *MesosExecutorDriver) reregistered(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring reregistration message from slave because aborted")
		return
	}
	log.Infoln("Executor driver reregistered")

	msg := pbMsg.(*mesosproto.ExecutorReregisteredMessage)
	slaveID := msg.GetSlaveId()
	slaveInfo := msg.GetSlaveInfo()

	if driver.stopped() {
		log.Infof("Ignoring re-registered message from slave %v, because the driver is stopped!\n", slaveID)
		return
	}

	log.Infof("Re-registered on slave %v\n", slaveID)
	driver.connected = true
	driver.connection = uuid.NewUUID()
	driver.cond.Broadcast() // useful for testing
	driver.withExecutor(func(e Executor) { e.Reregistered(driver, slaveInfo) })
}

func (driver *MesosExecutorDriver) send(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	c := make(chan error, 1)
	go func() { c <- driver.messenger.Send(ctx, upid, msg) }()

	select {
	case <-ctx.Done():
		<-c // wait for Send(...)
		return ctx.Err()
	case err := <-c:
		return err
	}
}

func (driver *MesosExecutorDriver) reconnect(ctx context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring reconnect message from slave because aborted")
		return
	}
	log.Infoln("Executor driver reconnect")

	msg := pbMsg.(*mesosproto.ReconnectExecutorMessage)
	slaveID := msg.GetSlaveId()

	if driver.stopped() {
		log.Infof("Ignoring reconnect message from slave %v, because the driver is stopped!\n", slaveID)
		return
	}

	log.Infof("Received reconnect request from slave %v\n", slaveID)
	driver.slaveUPID = from

	message := &mesosproto.ReregisterExecutorMessage{
		ExecutorId:  driver.executorID,
		FrameworkId: driver.frameworkID,
	}
	// Send all unacknowledged updates.
	for _, u := range driver.updates {
		message.Updates = append(message.Updates, u)
	}
	// Send all unacknowledged tasks.
	for _, t := range driver.tasks {
		message.Tasks = append(message.Tasks, t)
	}
	// Send the message.
	if err := driver.send(ctx, driver.slaveUPID, message); err != nil {
		log.Errorf("Failed to send %v: %v\n", message, err)
	}
}

func (driver *MesosExecutorDriver) runTask(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring runTask message from slave because aborted")
		return
	}
	log.Infoln("Executor driver runTask")

	msg := pbMsg.(*mesosproto.RunTaskMessage)
	task := msg.GetTask()
	taskID := task.GetTaskId()

	if driver.stopped() {
		log.Infof("Ignoring run task message for task %v because the driver is stopped!\n", taskID)
		return
	}
	if _, ok := driver.tasks[taskID.String()]; ok {
		log.Fatalf("Unexpected duplicate task %v\n", taskID)
	}

	log.Infof("Executor asked to run task '%v'\n", taskID)
	driver.tasks[taskID.String()] = task
	driver.withExecutor(func(e Executor) { e.LaunchTask(driver, task) })
}

func (driver *MesosExecutorDriver) killTask(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring killTask message from slave because aborted")
		return
	}
	log.Infoln("Executor driver killTask")

	msg := pbMsg.(*mesosproto.KillTaskMessage)
	taskID := msg.GetTaskId()

	if driver.stopped() {
		log.Infof("Ignoring kill task message for task %v, because the driver is stopped!\n", taskID)
		return
	}

	log.Infof("Executor driver is asked to kill task '%v'\n", taskID)
	driver.withExecutor(func(e Executor) { e.KillTask(driver, taskID) })
}

func (driver *MesosExecutorDriver) statusUpdateAcknowledgement(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring status update ack message because aborted")
		return
	}
	log.Infoln("Executor statusUpdateAcknowledgement")

	msg := pbMsg.(*mesosproto.StatusUpdateAcknowledgementMessage)
	log.Infof("Receiving status update acknowledgement %v", msg)

	frameworkID := msg.GetFrameworkId()
	taskID := msg.GetTaskId()
	uuid := uuid.UUID(msg.GetUuid())

	if driver.stopped() {
		log.Infof("Ignoring status update acknowledgement %v for task %v of framework %v because the driver is stopped!\n",
			uuid, taskID, frameworkID)
	}

	// Remove the corresponding update.
	delete(driver.updates, uuid.String())
	// Remove the corresponding task.
	delete(driver.tasks, taskID.String())
}

func (driver *MesosExecutorDriver) frameworkMessage(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring frameworkMessage message from slave because aborted")
		return
	}
	log.Infoln("Executor driver received frameworkMessage")

	msg := pbMsg.(*mesosproto.FrameworkToExecutorMessage)
	data := msg.GetData()

	if driver.stopped() {
		log.Infof("Ignoring framework message because the driver is stopped!\n")
		return
	}

	log.Infof("Executor driver receives framework message\n")
	driver.withExecutor(func(e Executor) { e.FrameworkMessage(driver, string(data)) })
}

func (driver *MesosExecutorDriver) shutdown(_ context.Context, _ *upid.UPID, _ proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring shutdown message because aborted")
		return
	}
	log.Infoln("Executor driver received shutdown")

	if driver.stopped() {
		log.Infof("Ignoring shutdown message because the driver is stopped!\n")
		return
	}

	log.Infof("Executor driver is asked to shutdown\n")

	driver.withExecutor(func(e Executor) { e.Shutdown(driver) })
	// driver.stop() will cause process to eventually stop.
	driver.stop()
}

func (driver *MesosExecutorDriver) frameworkError(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesosproto.Status_DRIVER_ABORTED {
		log.V(1).Infof("ignoring framework error message because aborted")
		return
	}
	log.Infoln("Executor driver received error")

	msg := pbMsg.(*mesosproto.FrameworkErrorMessage)
	driver.withExecutor(func(e Executor) { e.Error(driver, msg.GetMessage()) })
}

// ------------------------ Driver Implementation ----------------- //

// Start starts the executor driver
func (driver *MesosExecutorDriver) Start() (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.start()
}

func (driver *MesosExecutorDriver) start() (mesosproto.Status, error) {
	log.Infoln("Starting the executor driver")

	if driver.status != mesosproto.Status_DRIVER_NOT_STARTED {
		return driver.status, fmt.Errorf("Unable to Start, expecting status %s, but got %s", mesosproto.Status_DRIVER_NOT_STARTED, driver.status)
	}

	// Start the messenger.
	if err := driver.messenger.Start(); err != nil {
		log.Errorf("Failed to start executor: %v\n", err)
		return driver.status, err
	}

	pid := driver.messenger.UPID()
	driver.self = &pid

	// Register with slave.
	log.V(3).Infoln("Sending Executor registration")
	message := &mesosproto.RegisterExecutorMessage{
		FrameworkId: driver.frameworkID,
		ExecutorId:  driver.executorID,
	}

	if err := driver.send(driver.context(), driver.slaveUPID, message); err != nil {
		log.Errorf("Stopping the executor, failed to send %v: %v\n", message, err)
		err0 := driver._stop(driver.status)
		if err0 != nil {
			log.Errorf("Failed to stop executor: %v\n", err)
			return driver.status, err0
		}
		return driver.status, err
	}

	driver.status = mesosproto.Status_DRIVER_RUNNING
	close(driver.started)

	log.Infoln("Mesos executor is started with PID=", driver.self.String())
	return driver.status, nil
}

// Stop stops the driver by sending a 'stopEvent' to the event loop, and
// receives the result from the response channel.
func (driver *MesosExecutorDriver) Stop() (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.stop()
}

func (driver *MesosExecutorDriver) stop() (mesosproto.Status, error) {
	log.Infoln("Stopping the executor driver")
	if driver.status != mesosproto.Status_DRIVER_RUNNING {
		return driver.status, fmt.Errorf("Unable to Stop, expecting status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}
	return mesosproto.Status_DRIVER_STOPPED, driver._stop(mesosproto.Status_DRIVER_STOPPED)
}

// internal function for stopping the driver and set reason for stopping
// Note that messages inflight or queued will not be processed.
func (driver *MesosExecutorDriver) _stop(stopStatus mesosproto.Status) error {
	err := driver.messenger.Stop()
	defer func() {
		select {
		case <-driver.stopCh:
			// already closed
		default:
			close(driver.stopCh)
		}
		driver.cond.Broadcast()
	}()

	driver.status = stopStatus
	if err != nil {
		return err
	}
	return nil
}

// Abort aborts the driver by sending an 'abortEvent' to the event loop, and
// receives the result from the response channel.
func (driver *MesosExecutorDriver) Abort() (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.abort()
}

func (driver *MesosExecutorDriver) abort() (mesosproto.Status, error) {
	if driver.status != mesosproto.Status_DRIVER_RUNNING {
		return driver.status, fmt.Errorf("Unable to Stop, expecting status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}

	log.Infoln("Aborting the executor driver")
	return mesosproto.Status_DRIVER_ABORTED, driver._stop(mesosproto.Status_DRIVER_ABORTED)
}

// Join waits for the driver by sending a 'joinEvent' to the event loop, and wait
// on a channel for the notification of driver termination.
func (driver *MesosExecutorDriver) Join() (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.join()
}

func (driver *MesosExecutorDriver) join() (mesosproto.Status, error) {
	log.Infoln("Waiting for the executor driver to stop")
	if driver.status != mesosproto.Status_DRIVER_RUNNING {
		return driver.status, fmt.Errorf("Unable to Join, expecting status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}
	for {
		select {
		case <-driver.stopCh: // wait for stop signal
			return driver.status, nil
		default:
			driver.cond.Wait()
		}
	}
}

// Run starts the driver and calls Join() to wait for stop request.
func (driver *MesosExecutorDriver) Run() (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.run()
}

func (driver *MesosExecutorDriver) run() (mesosproto.Status, error) {
	stat, err := driver.start()

	if err != nil {
		return driver.stop()
	}

	if stat != mesosproto.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to continue to Run, expecting status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}

	return driver.join()
}

// SendStatusUpdate sends status updates to the slave.
func (driver *MesosExecutorDriver) SendStatusUpdate(taskStatus *mesosproto.TaskStatus) (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.sendStatusUpdate(taskStatus)
}

func (driver *MesosExecutorDriver) sendStatusUpdate(taskStatus *mesosproto.TaskStatus) (mesosproto.Status, error) {
	log.V(3).Infoln("Sending task status update: ", taskStatus.String())

	if driver.status != mesosproto.Status_DRIVER_RUNNING {
		return driver.status, fmt.Errorf("Unable to SendStatusUpdate, expecting driver.status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}

	if taskStatus.GetState() == mesosproto.TaskState_TASK_STAGING {
		err := fmt.Errorf("Executor is not allowed to send TASK_STAGING status update. Aborting!")
		log.Errorln(err)
		if err0 := driver._stop(mesosproto.Status_DRIVER_ABORTED); err0 != nil {
			log.Errorln("Error while stopping the driver", err0)
		}

		return driver.status, err
	}

	// Set up status update.
	update := driver.makeStatusUpdate(taskStatus)
	log.Infof("Executor sending status update %v\n", update.String())

	// Capture the status update.
	driver.updates[uuid.UUID(update.GetUuid()).String()] = update

	// Put the status update in the message.
	message := &mesosproto.StatusUpdateMessage{
		Update: update,
		Pid:    proto.String(driver.self.String()),
	}
	// Send the message.
	if err := driver.send(driver.context(), driver.slaveUPID, message); err != nil {
		log.Errorf("Failed to send %v: %v\n", message, err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosExecutorDriver) makeStatusUpdate(taskStatus *mesosproto.TaskStatus) *mesosproto.StatusUpdate {
	now := float64(time.Now().Unix())
	// Fill in all the fields.
	taskStatus.Timestamp = proto.Float64(now)
	taskStatus.SlaveId = driver.slaveID
	update := &mesosproto.StatusUpdate{
		FrameworkId: driver.frameworkID,
		ExecutorId:  driver.executorID,
		SlaveId:     driver.slaveID,
		Status:      taskStatus,
		Timestamp:   proto.Float64(now),
		Uuid:        uuid.NewUUID(),
	}
	return update
}

// SendFrameworkMessage sends the framework message by sending a 'sendFrameworkMessageEvent'
// to the event loop, and receives the result from the response channel.
func (driver *MesosExecutorDriver) SendFrameworkMessage(data string) (mesosproto.Status, error) {
	driver.lock.Lock()
	defer driver.lock.Unlock()
	return driver.sendFrameworkMessage(data)
}

func (driver *MesosExecutorDriver) sendFrameworkMessage(data string) (mesosproto.Status, error) {
	log.V(3).Infoln("Sending framework message", string(data))

	if driver.status != mesosproto.Status_DRIVER_RUNNING {
		return driver.status, fmt.Errorf("Unable to SendFrameworkMessage, expecting status %s, but got %s", mesosproto.Status_DRIVER_RUNNING, driver.status)
	}

	message := &mesosproto.ExecutorToFrameworkMessage{
		SlaveId:     driver.slaveID,
		FrameworkId: driver.frameworkID,
		ExecutorId:  driver.executorID,
		Data:        []byte(data),
	}

	// Send the message.
	if err := driver.send(driver.context(), driver.slaveUPID, message); err != nil {
		log.Errorln("Failed to send message %v: %v", message, err)
		return driver.status, err
	}
	return driver.status, nil
}
