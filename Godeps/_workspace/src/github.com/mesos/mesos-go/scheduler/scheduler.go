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
	"errors"
	"fmt"
	"math"
	"math/rand"
	"net"
	"os/user"
	"sync"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/auth"
	"github.com/mesos/mesos-go/detector"
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/mesosutil/process"
	"github.com/mesos/mesos-go/messenger"
	"github.com/mesos/mesos-go/upid"
	"github.com/pborman/uuid"
	"golang.org/x/net/context"
)

const (
	authTimeout                  = 5 * time.Second // timeout interval for an authentication attempt
	registrationRetryIntervalMax = float64(1 * time.Minute)
	registrationBackoffFactor    = 2 * time.Second
)

var (
	authenticationCanceledError = errors.New("authentication canceled")
)

// helper to track authentication progress and to prevent multiple close() ops
// against a signalling chan. it's safe to invoke the func's of this struct
// even if the receiver pointer is nil.
type authenticationAttempt struct {
	done     chan struct{}
	doneOnce sync.Once
}

func (a *authenticationAttempt) cancel() {
	if a != nil {
		a.doneOnce.Do(func() { close(a.done) })
	}
}

func (a *authenticationAttempt) inProgress() bool {
	if a != nil {
		select {
		case <-a.done:
			return false
		default:
			return true
		}
	}
	return false
}

type DriverConfig struct {
	Scheduler        Scheduler
	Framework        *mesos.FrameworkInfo
	Master           string
	Credential       *mesos.Credential                     // optional
	WithAuthContext  func(context.Context) context.Context // required when Credential != nil
	HostnameOverride string                                // optional
	BindingAddress   net.IP                                // optional
	BindingPort      uint16                                // optional
	NewMessenger     func() (messenger.Messenger, error)   // optional
}

// Concrete implementation of a SchedulerDriver that connects a
// Scheduler with a Mesos master. The MesosSchedulerDriver is
// thread-safe.
//
// Note that scheduler failover is supported in Mesos. After a
// scheduler is registered with Mesos it may failover (to a new
// process on the same machine or across multiple machines) by
// creating a new driver with the ID given to it in
// Scheduler.Registered().
//
// The driver is responsible for invoking the Scheduler callbacks as
// it communicates with the Mesos master.
//
// Note that blocking on the MesosSchedulerDriver (e.g., via
// MesosSchedulerDriver.Join) doesn't affect the scheduler callbacks
// in anyway because they are handled by a different thread.
//
// TODO(yifan): examples.
// See src/examples/test_framework.cpp for an example of using the
// MesosSchedulerDriver.
type MesosSchedulerDriver struct {
	Scheduler     Scheduler
	MasterPid     *upid.UPID
	FrameworkInfo *mesos.FrameworkInfo

	lock            sync.RWMutex
	self            *upid.UPID
	stopCh          chan struct{}
	stopped         bool
	status          mesos.Status
	messenger       messenger.Messenger
	masterDetector  detector.Master
	connected       bool
	connection      uuid.UUID
	failoverTimeout float64
	failover        bool
	cache           *schedCache
	updates         map[string]*mesos.StatusUpdate // Key is a UUID string.
	tasks           map[string]*mesos.TaskInfo     // Key is a UUID string.
	credential      *mesos.Credential
	authenticated   bool
	authenticating  *authenticationAttempt
	reauthenticate  bool
	withAuthContext func(context.Context) context.Context
}

// Create a new mesos scheduler driver with the given
// scheduler, framework info,
// master address, and credential(optional)
func NewMesosSchedulerDriver(config DriverConfig) (initializedDriver *MesosSchedulerDriver, err error) {
	if config.Scheduler == nil {
		err = fmt.Errorf("Scheduler callbacks required.")
	} else if config.Master == "" {
		err = fmt.Errorf("Missing master location URL.")
	} else if config.Framework == nil {
		err = fmt.Errorf("FrameworkInfo must be provided.")
	} else if config.Credential != nil && config.WithAuthContext == nil {
		err = fmt.Errorf("WithAuthContext must be provided when Credential != nil")
	}
	if err != nil {
		return
	}

	framework := cloneFrameworkInfo(config.Framework)

	// set default userid
	if framework.GetUser() == "" {
		user, err := user.Current()
		if err != nil || user == nil {
			if err != nil {
				log.Warningf("Failed to obtain username: %v\n", err)
			} else {
				log.Warningln("Failed to obtain username.")
			}
			framework.User = proto.String("")
		} else {
			framework.User = proto.String(user.Username)
		}
	}

	// default hostname
	hostname := util.GetHostname(config.HostnameOverride)
	if framework.GetHostname() == "" {
		framework.Hostname = proto.String(hostname)
	}

	driver := &MesosSchedulerDriver{
		Scheduler:       config.Scheduler,
		FrameworkInfo:   framework,
		stopCh:          make(chan struct{}),
		status:          mesos.Status_DRIVER_NOT_STARTED,
		stopped:         true,
		cache:           newSchedCache(),
		credential:      config.Credential,
		failover:        framework.Id != nil && len(framework.Id.GetValue()) > 0,
		withAuthContext: config.WithAuthContext,
	}

	if framework.FailoverTimeout != nil && *framework.FailoverTimeout > 0 {
		driver.failoverTimeout = *framework.FailoverTimeout * float64(time.Second)
		log.V(1).Infof("found failover_timeout = %v", time.Duration(driver.failoverTimeout))
	}

	newMessenger := config.NewMessenger
	if newMessenger == nil {
		newMessenger = func() (messenger.Messenger, error) {
			process := process.New("scheduler")
			return messenger.ForHostname(process, hostname, config.BindingAddress, config.BindingPort)
		}
	}

	// initialize new detector.
	if driver.masterDetector, err = detector.New(config.Master); err != nil {
		return
	} else if driver.messenger, err = newMessenger(); err != nil {
		return
	} else if err = driver.init(); err != nil {
		return
	} else {
		initializedDriver = driver
	}
	return
}

func cloneFrameworkInfo(framework *mesos.FrameworkInfo) *mesos.FrameworkInfo {
	if framework == nil {
		return nil
	}

	clonedInfo := *framework
	if clonedInfo.Id != nil {
		clonedId := *clonedInfo.Id
		clonedInfo.Id = &clonedId
		if framework.FailoverTimeout != nil {
			clonedInfo.FailoverTimeout = proto.Float64(*framework.FailoverTimeout)
		}
		if framework.Checkpoint != nil {
			clonedInfo.Checkpoint = proto.Bool(*framework.Checkpoint)
		}
	}
	return &clonedInfo
}

// init initializes the driver.
func (driver *MesosSchedulerDriver) init() error {
	log.Infof("Initializing mesos scheduler driver\n")

	// Install handlers.
	driver.messenger.Install(driver.frameworkRegistered, &mesos.FrameworkRegisteredMessage{})
	driver.messenger.Install(driver.frameworkReregistered, &mesos.FrameworkReregisteredMessage{})
	driver.messenger.Install(driver.resourcesOffered, &mesos.ResourceOffersMessage{})
	driver.messenger.Install(driver.resourceOfferRescinded, &mesos.RescindResourceOfferMessage{})
	driver.messenger.Install(driver.statusUpdated, &mesos.StatusUpdateMessage{})
	driver.messenger.Install(driver.slaveLost, &mesos.LostSlaveMessage{})
	driver.messenger.Install(driver.frameworkMessageRcvd, &mesos.ExecutorToFrameworkMessage{})
	driver.messenger.Install(driver.frameworkErrorRcvd, &mesos.FrameworkErrorMessage{})
	driver.messenger.Install(driver.handleMasterChanged, &mesos.InternalMasterChangeDetected{})
	driver.messenger.Install(driver.handleAuthenticationResult, &mesos.InternalAuthenticationResult{})
	return nil
}

// lead master detection callback.
func (driver *MesosSchedulerDriver) handleMasterChanged(from *upid.UPID, pbMsg proto.Message) {
	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.Info("Ignoring master change because the driver is aborted.")
		return
	} else if !from.Equal(driver.self) {
		log.Errorf("ignoring master changed message received from upid '%v'", from)
		return
	}

	// Reconnect every time a master is dected.
	if driver.Connected() {
		log.V(3).Info("Disconnecting scheduler.")
		driver.MasterPid = nil
		driver.Scheduler.Disconnected(driver)
	}

	msg := pbMsg.(*mesos.InternalMasterChangeDetected)
	master := msg.Master

	driver.setConnected(false)
	driver.authenticated = false

	if master != nil {
		log.Infof("New master %s detected\n", master.GetPid())

		pid, err := upid.Parse(master.GetPid())
		if err != nil {
			panic("Unable to parse Master's PID value.") // this should not happen.
		}

		driver.MasterPid = pid // save for downstream ops.
		driver.tryAuthentication()
	} else {
		log.Infoln("No master detected.")
	}
}

func (driver *MesosSchedulerDriver) tryAuthentication() {
	if driver.authenticated {
		// programming error
		panic("already authenticated")
	}

	masterPid := driver.MasterPid // save for referencing later in goroutine
	if masterPid == nil {
		log.Info("skipping authentication attempt because we lost the master")
		return
	}

	if driver.authenticating.inProgress() {
		// authentication is in progress, try to cancel it (we may too late already)
		driver.authenticating.cancel()
		driver.reauthenticate = true
		return
	}

	if driver.credential != nil {
		// authentication can block and we don't want to hold up the messenger loop
		authenticating := &authenticationAttempt{done: make(chan struct{})}
		go func() {
			defer authenticating.cancel()
			result := &mesos.InternalAuthenticationResult{
				//TODO(jdef): is this really needed?
				Success:   proto.Bool(false),
				Completed: proto.Bool(false),
				Pid:       proto.String(masterPid.String()),
			}
			// don't reference driver.authenticating here since it may have changed
			if err := driver.authenticate(masterPid, authenticating); err != nil {
				log.Errorf("Scheduler failed to authenticate: %v\n", err)
				if err == auth.AuthenticationFailed {
					result.Completed = proto.Bool(true)
				}
			} else {
				result.Completed = proto.Bool(true)
				result.Success = proto.Bool(true)
			}
			driver.messenger.Route(context.TODO(), driver.messenger.UPID(), result)
		}()
		driver.authenticating = authenticating
	} else {
		log.Infoln("No credentials were provided. " +
			"Attempting to register scheduler without authentication.")
		driver.authenticated = true
		go driver.doReliableRegistration(float64(registrationBackoffFactor))
	}
}

func (driver *MesosSchedulerDriver) handleAuthenticationResult(from *upid.UPID, pbMsg proto.Message) {
	if driver.Status() != mesos.Status_DRIVER_RUNNING {
		log.V(1).Info("ignoring authenticate because driver is not running")
		return
	}
	if !from.Equal(driver.self) {
		log.Errorf("ignoring authentication result message received from upid '%v'", from)
		return
	}
	if driver.authenticated {
		// programming error
		panic("already authenticated")
	}
	if driver.MasterPid == nil {
		log.Infoln("ignoring authentication result because master is lost")
		driver.authenticating.cancel() // cancel any in-progress background attempt

		// disable future retries until we get a new master
		driver.reauthenticate = false
		return
	}
	msg := pbMsg.(*mesos.InternalAuthenticationResult)
	if driver.reauthenticate || !msg.GetCompleted() || driver.MasterPid.String() != msg.GetPid() {
		log.Infof("failed to authenticate with master %v: master changed", driver.MasterPid)
		driver.authenticating.cancel() // cancel any in-progress background authentication
		driver.reauthenticate = false
		driver.tryAuthentication()
		return
	}
	if !msg.GetSuccess() {
		log.Errorf("master %v refused authentication", driver.MasterPid)
		return
	}
	driver.authenticated = true
	go driver.doReliableRegistration(float64(registrationBackoffFactor))
}

// ------------------------- Accessors ----------------------- //
func (driver *MesosSchedulerDriver) Status() mesos.Status {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.status
}
func (driver *MesosSchedulerDriver) setStatus(stat mesos.Status) {
	driver.lock.Lock()
	driver.status = stat
	driver.lock.Unlock()
}

func (driver *MesosSchedulerDriver) Stopped() bool {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.stopped
}

func (driver *MesosSchedulerDriver) setStopped(val bool) {
	driver.lock.Lock()
	driver.stopped = val
	driver.lock.Unlock()
}

func (driver *MesosSchedulerDriver) Connected() bool {
	driver.lock.RLock()
	defer driver.lock.RUnlock()
	return driver.connected
}

func (driver *MesosSchedulerDriver) setConnected(val bool) {
	driver.lock.Lock()
	driver.connected = val
	if val {
		driver.failover = false
	}
	driver.lock.Unlock()
}

// ---------------------- Handlers for Events from Master --------------- //
func (driver *MesosSchedulerDriver) frameworkRegistered(from *upid.UPID, pbMsg proto.Message) {
	log.V(2).Infoln("Handling scheduler driver framework registered event.")

	msg := pbMsg.(*mesos.FrameworkRegisteredMessage)
	masterInfo := msg.GetMasterInfo()
	masterPid := masterInfo.GetPid()
	frameworkId := msg.GetFrameworkId()

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.Infof("ignoring FrameworkRegisteredMessage from master %s, driver is aborted", masterPid)
		return
	}

	if driver.connected {
		log.Infoln("ignoring FrameworkRegisteredMessage from master, driver is already connected", masterPid)
		return
	}

	if driver.stopped {
		log.Infof("ignoring FrameworkRegisteredMessage from master %s, driver is stopped", masterPid)
		return
	}
	if !driver.MasterPid.Equal(from) {
		log.Warningf("ignoring framework registered message because it was sent from '%v' instead of leading master '%v'", from, driver.MasterPid)
		return
	}

	log.Infof("Framework registered with ID=%s\n", frameworkId.GetValue())
	driver.FrameworkInfo.Id = frameworkId // generated by master.

	driver.setConnected(true)
	driver.connection = uuid.NewUUID()
	driver.Scheduler.Registered(driver, frameworkId, masterInfo)
}

func (driver *MesosSchedulerDriver) frameworkReregistered(from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling Scheduler re-registered event.")
	msg := pbMsg.(*mesos.FrameworkReregisteredMessage)

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.Infoln("Ignoring FrameworkReregisteredMessage from master, driver is aborted!")
		return
	}
	if driver.connected {
		log.Infoln("Ignoring FrameworkReregisteredMessage from master,driver is already connected!")
		return
	}
	if !driver.MasterPid.Equal(from) {
		log.Warningf("ignoring framework re-registered message because it was sent from '%v' instead of leading master '%v'", from, driver.MasterPid)
		return
	}

	// TODO(vv) detect if message was from leading-master (sched.cpp)
	log.Infof("Framework re-registered with ID [%s] ", msg.GetFrameworkId().GetValue())
	driver.setConnected(true)
	driver.connection = uuid.NewUUID()

	driver.Scheduler.Reregistered(driver, msg.GetMasterInfo())

}

func (driver *MesosSchedulerDriver) resourcesOffered(from *upid.UPID, pbMsg proto.Message) {
	log.V(2).Infoln("Handling resource offers.")

	msg := pbMsg.(*mesos.ResourceOffersMessage)
	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.Infoln("Ignoring ResourceOffersMessage, the driver is aborted!")
		return
	}

	if !driver.connected {
		log.Infoln("Ignoring ResourceOffersMessage, the driver is not connected!")
		return
	}

	pidStrings := msg.GetPids()
	if len(pidStrings) != len(msg.Offers) {
		log.Errorln("Ignoring offers, Offer count does not match Slave PID count.")
		return
	}

	for i, offer := range msg.Offers {
		if pid, err := upid.Parse(pidStrings[i]); err == nil {
			driver.cache.putOffer(offer, pid)
			log.V(2).Infof("Cached offer %s from SlavePID %s", offer.Id.GetValue(), pid)
		} else {
			log.Warningf("Failed to parse offer PID '%v': %v", pid, err)
		}
	}

	driver.Scheduler.ResourceOffers(driver, msg.Offers)
}

func (driver *MesosSchedulerDriver) resourceOfferRescinded(from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling resource offer rescinded.")

	msg := pbMsg.(*mesos.RescindResourceOfferMessage)

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.Infoln("Ignoring RescindResourceOfferMessage, the driver is aborted!")
		return
	}

	if !driver.connected {
		log.Infoln("Ignoring ResourceOffersMessage, the driver is not connected!")
		return
	}

	// TODO(vv) check for leading master (see sched.cpp)

	log.V(1).Infoln("Rescinding offer ", msg.OfferId.GetValue())
	driver.cache.removeOffer(msg.OfferId)
	driver.Scheduler.OfferRescinded(driver, msg.OfferId)
}

func (driver *MesosSchedulerDriver) send(upid *upid.UPID, msg proto.Message) error {
	//TODO(jdef) should implement timeout here
	ctx, cancel := context.WithCancel(context.TODO())
	defer cancel()

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

func (driver *MesosSchedulerDriver) statusUpdated(from *upid.UPID, pbMsg proto.Message) {
	msg := pbMsg.(*mesos.StatusUpdateMessage)

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Ignoring StatusUpdate message, the driver is aborted!")
		return
	}
	if !driver.connected {
		log.V(1).Infoln("Ignoring StatusUpdate message, the driver is not connected!")
		return
	}
	if !driver.MasterPid.Equal(from) {
		log.Warningf("ignoring status message because it was sent from '%v' instead of leading master '%v'", from, driver.MasterPid)
		return
	}

	log.V(2).Infoln("Received status update from ", from.String(), " status source:", msg.GetPid())

	driver.Scheduler.StatusUpdate(driver, msg.Update.GetStatus())

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Not sending StatusUpdate ACK, the driver is aborted!")
		return
	}

	// Send StatusUpdate Acknowledgement
	// Only send ACK if udpate was not from this driver
	if !from.Equal(driver.self) && msg.GetPid() != from.String() {
		ackMsg := &mesos.StatusUpdateAcknowledgementMessage{
			SlaveId:     msg.Update.SlaveId,
			FrameworkId: driver.FrameworkInfo.Id,
			TaskId:      msg.Update.Status.TaskId,
			Uuid:        msg.Update.Uuid,
		}

		log.V(2).Infoln("Sending status update ACK to ", from.String())
		if err := driver.send(driver.MasterPid, ackMsg); err != nil {
			log.Errorf("Failed to send StatusUpdate ACK message: %v\n", err)
			return
		}
	} else {
		log.V(1).Infoln("Not sending ACK, update is not from slave:", from.String())
	}
}

func (driver *MesosSchedulerDriver) slaveLost(from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling LostSlave event.")

	msg := pbMsg.(*mesos.LostSlaveMessage)

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Ignoring LostSlave message, the driver is aborted!")
		return
	}

	if !driver.connected {
		log.V(1).Infoln("Ignoring LostSlave message, the driver is not connected!")
		return
	}

	// TODO(VV) - detect leading master (see sched.cpp)

	log.V(2).Infoln("Lost slave ", msg.SlaveId.GetValue())
	driver.cache.removeSlavePid(msg.SlaveId)

	driver.Scheduler.SlaveLost(driver, msg.SlaveId)
}

func (driver *MesosSchedulerDriver) frameworkMessageRcvd(from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling framework message event.")

	msg := pbMsg.(*mesos.ExecutorToFrameworkMessage)

	if driver.Status() == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Ignoring framwork message, the driver is aborted!")
		return
	}

	log.V(1).Infoln("Received Framwork Message ", msg.String())

	driver.Scheduler.FrameworkMessage(driver, msg.ExecutorId, msg.SlaveId, string(msg.Data))
}

func (driver *MesosSchedulerDriver) frameworkErrorRcvd(from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling framework error event.")
	msg := pbMsg.(*mesos.FrameworkErrorMessage)
	driver.error(msg.GetMessage(), true)
}

// ---------------------- Interface Methods ---------------------- //

// Starts the scheduler driver.
// Returns immediately if an error occurs within start sequence.
func (driver *MesosSchedulerDriver) Start() (mesos.Status, error) {
	log.Infoln("Starting the scheduler driver...")

	if stat := driver.Status(); stat != mesos.Status_DRIVER_NOT_STARTED {
		return stat, fmt.Errorf("Unable to Start, expecting driver status %s, but is %s:", mesos.Status_DRIVER_NOT_STARTED, stat)
	}

	driver.setStopped(true)
	driver.setStatus(mesos.Status_DRIVER_NOT_STARTED)

	// Start the messenger.
	if err := driver.messenger.Start(); err != nil {
		log.Errorf("Scheduler failed to start the messenger: %v\n", err)
		return driver.Status(), err
	}

	driver.self = driver.messenger.UPID()
	driver.setStatus(mesos.Status_DRIVER_RUNNING)
	driver.setStopped(false)

	log.Infof("Mesos scheduler driver started with PID=%v", driver.self)

	listener := detector.OnMasterChanged(func(m *mesos.MasterInfo) {
		driver.messenger.Route(context.TODO(), driver.self, &mesos.InternalMasterChangeDetected{
			Master: m,
		})
	})

	// register with Detect() AFTER we have a self pid from the messenger, otherwise things get ugly
	// because our internal messaging depends on it. detector callbacks are routed over the messenger
	// bus, maintaining serial (concurrency-safe) callback execution.
	log.V(1).Infof("starting master detector %T: %+v", driver.masterDetector, driver.masterDetector)
	driver.masterDetector.Detect(listener)

	log.V(2).Infoln("master detector started")
	return driver.Status(), nil
}

// authenticate against the spec'd master pid using the configured authenticationProvider.
// the authentication process is canceled upon either cancelation of authenticating, or
// else because it timed out (authTimeout).
//
// TODO(jdef) perhaps at some point in the future this will get pushed down into
// the messenger layer (e.g. to use HTTP-based authentication). We'd probably still
// specify the callback.Handler here, along with the user-selected authentication
// provider. Perhaps in the form of some messenger.AuthenticationConfig.
//
func (driver *MesosSchedulerDriver) authenticate(pid *upid.UPID, authenticating *authenticationAttempt) error {
	log.Infof("authenticating with master %v", pid)
	ctx, cancel := context.WithTimeout(context.Background(), authTimeout)
	handler := &CredentialHandler{
		pid:        pid,
		client:     driver.self,
		credential: driver.credential,
	}
	ctx = driver.withAuthContext(ctx)
	ctx = auth.WithParentUPID(ctx, *driver.self)

	ch := make(chan error, 1)
	go func() { ch <- auth.Login(ctx, handler) }()
	select {
	case <-ctx.Done():
		<-ch
		return ctx.Err()
	case <-authenticating.done:
		cancel()
		<-ch
		return authenticationCanceledError
	case e := <-ch:
		cancel()
		return e
	}
}

func (driver *MesosSchedulerDriver) doReliableRegistration(maxBackoff float64) {
	for {
		if !driver.registerOnce() {
			return
		}
		maxBackoff = math.Min(maxBackoff, registrationRetryIntervalMax)

		// If failover timeout is present, bound the maximum backoff
		// by 1/10th of the failover timeout.
		if driver.failoverTimeout > 0 {
			maxBackoff = math.Min(maxBackoff, driver.failoverTimeout/10.0)
		}

		// Determine the delay for next attempt by picking a random
		// duration between 0 and 'maxBackoff'.
		delay := time.Duration(maxBackoff * rand.Float64())

		log.V(1).Infof("will retry registration in %v if necessary", delay)

		select {
		case <-driver.stopCh:
			return
		case <-time.After(delay):
			maxBackoff *= 2
		}
	}
}

// return true if we should attempt another registration later
func (driver *MesosSchedulerDriver) registerOnce() bool {

	var (
		failover bool
		pid      *upid.UPID
	)
	if func() bool {
		driver.lock.RLock()
		defer driver.lock.RUnlock()

		if driver.stopped || driver.connected || driver.MasterPid == nil || (driver.credential != nil && !driver.authenticated) {
			log.V(1).Infof("skipping registration request: stopped=%v, connected=%v, authenticated=%v",
				driver.stopped, driver.connected, driver.authenticated)
			return false
		}
		failover = driver.failover
		pid = driver.MasterPid
		return true
	}() {
		// register framework
		var message proto.Message
		if driver.FrameworkInfo.Id != nil && len(driver.FrameworkInfo.Id.GetValue()) > 0 {
			// not the first time, or failing over
			log.V(1).Infof("Reregistering with master: %v", pid)
			message = &mesos.ReregisterFrameworkMessage{
				Framework: driver.FrameworkInfo,
				Failover:  proto.Bool(failover),
			}
		} else {
			log.V(1).Infof("Registering with master: %v", pid)
			message = &mesos.RegisterFrameworkMessage{
				Framework: driver.FrameworkInfo,
			}
		}
		if err := driver.send(pid, message); err != nil {
			log.Errorf("failed to send RegisterFramework message: %v", err)
			if _, err = driver.Stop(failover); err != nil {
				log.Errorf("failed to stop scheduler driver: %v", err)
			}
		}
		return true
	}
	return false
}

//Join blocks until the driver is stopped.
//Should follow a call to Start()
func (driver *MesosSchedulerDriver) Join() (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to Join, expecting driver status %s, but is %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	<-driver.stopCh // wait for stop signal
	return driver.Status(), nil
}

//Run starts and joins driver process and waits to be stopped or aborted.
func (driver *MesosSchedulerDriver) Run() (mesos.Status, error) {
	stat, err := driver.Start()

	if err != nil {
		return driver.Stop(false)
	}

	if stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to Run, expecting driver status %s, but is %s:", mesos.Status_DRIVER_RUNNING, driver.status)
	}

	log.Infoln("Scheduler driver running.  Waiting to be stopped.")
	return driver.Join()
}

//Stop stops the driver.
func (driver *MesosSchedulerDriver) Stop(failover bool) (mesos.Status, error) {
	log.Infoln("Stopping the scheduler driver")
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to Stop, expected driver status %s, but is %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if driver.connected && !failover {
		// unregister the framework
		log.Infoln("Unregistering the scheduler driver")
		message := &mesos.UnregisterFrameworkMessage{
			FrameworkId: driver.FrameworkInfo.Id,
		}
		//TODO(jdef) this is actually a little racy: we send an 'unregister' message but then
		// immediately afterward the messenger is stopped in driver.stop(). so the unregister message
		// may not actually end up being sent out.
		if err := driver.send(driver.MasterPid, message); err != nil {
			log.Errorf("Failed to send UnregisterFramework message while stopping driver: %v\n", err)
			return driver.stop(mesos.Status_DRIVER_ABORTED)
		}
	}

	// stop messenger
	return driver.stop(mesos.Status_DRIVER_STOPPED)
}

func (driver *MesosSchedulerDriver) stop(stopStatus mesos.Status) (mesos.Status, error) {
	// stop messenger
	err := driver.messenger.Stop()
	defer func() {
		select {
		case <-driver.stopCh:
			// already closed
		default:
			close(driver.stopCh)
		}
	}()

	driver.setStatus(stopStatus)
	driver.setStopped(true)
	driver.connected = false

	if err != nil {
		return stopStatus, err
	}

	return stopStatus, nil
}

func (driver *MesosSchedulerDriver) Abort() (stat mesos.Status, err error) {
	defer driver.masterDetector.Cancel()
	log.Infof("Aborting framework [%+v]", driver.FrameworkInfo.Id)
	if driver.connected {
		_, err = driver.Stop(true)
	} else {
		driver.messenger.Stop()
	}
	stat = mesos.Status_DRIVER_ABORTED
	driver.setStatus(stat)
	return
}

func (driver *MesosSchedulerDriver) LaunchTasks(offerIds []*mesos.OfferID, tasks []*mesos.TaskInfo, filters *mesos.Filters) (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to LaunchTasks, expected driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	// Launch tasks
	if !driver.connected {
		log.Infoln("Ignoring LaunchTasks message, disconnected from master.")
		// Send statusUpdate with status=TASK_LOST for each task.
		// See sched.cpp L#823
		for _, task := range tasks {
			driver.pushLostTask(task, "Master is disconnected")
		}
		return driver.Status(), fmt.Errorf("Not connected to master.  Tasks marked as lost.")
	}

	okTasks := make([]*mesos.TaskInfo, 0, len(tasks))

	// Set TaskInfo.executor.framework_id, if it's missing.
	for _, task := range tasks {
		if task.Executor != nil && task.Executor.FrameworkId == nil {
			task.Executor.FrameworkId = driver.FrameworkInfo.Id
		}
		okTasks = append(okTasks, task)
	}

	for _, offerId := range offerIds {
		for _, task := range okTasks {
			// Keep only the slave PIDs where we run tasks so we can send
			// framework messages directly.
			if driver.cache.containsOffer(offerId) {
				if driver.cache.getOffer(offerId).offer.SlaveId.Equal(task.SlaveId) {
					// cache the tasked slave, for future communication
					pid := driver.cache.getOffer(offerId).slavePid
					driver.cache.putSlavePid(task.SlaveId, pid)
				} else {
					log.Warningf("Attempting to launch task %s with the wrong slaveId offer %s\n", task.TaskId.GetValue(), task.SlaveId.GetValue())
				}
			} else {
				log.Warningf("Attempting to launch task %s with unknown offer %s\n", task.TaskId.GetValue(), offerId.GetValue())
			}
		}

		driver.cache.removeOffer(offerId) // if offer
	}

	// launch tasks
	message := &mesos.LaunchTasksMessage{
		FrameworkId: driver.FrameworkInfo.Id,
		OfferIds:    offerIds,
		Tasks:       okTasks,
		Filters:     filters,
	}

	if err := driver.send(driver.MasterPid, message); err != nil {
		for _, task := range tasks {
			driver.pushLostTask(task, "Unable to launch tasks: "+err.Error())
		}
		log.Errorf("Failed to send LaunchTask message: %v\n", err)
		return driver.Status(), err
	}

	return driver.Status(), nil
}

func (driver *MesosSchedulerDriver) pushLostTask(taskInfo *mesos.TaskInfo, why string) {
	msg := &mesos.StatusUpdateMessage{
		Update: &mesos.StatusUpdate{
			FrameworkId: driver.FrameworkInfo.Id,
			Status: &mesos.TaskStatus{
				TaskId:  taskInfo.TaskId,
				State:   mesos.TaskState_TASK_LOST.Enum(),
				Message: proto.String(why),
			},
			SlaveId:    taskInfo.SlaveId,
			ExecutorId: taskInfo.Executor.ExecutorId,
			Timestamp:  proto.Float64(float64(time.Now().Unix())),
			Uuid:       []byte(uuid.NewUUID()),
		},
	}

	// put it on internal chanel
	// will cause handler to push to attached Scheduler
	driver.statusUpdated(driver.self, msg)
}

func (driver *MesosSchedulerDriver) KillTask(taskId *mesos.TaskID) (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to KillTask, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if !driver.connected {
		log.Infoln("Ignoring kill task message, disconnected from master.")
		return driver.Status(), fmt.Errorf("Not connected to master")
	}

	message := &mesos.KillTaskMessage{
		FrameworkId: driver.FrameworkInfo.Id,
		TaskId:      taskId,
	}

	if err := driver.send(driver.MasterPid, message); err != nil {
		log.Errorf("Failed to send KillTask message: %v\n", err)
		return driver.Status(), err
	}

	return driver.Status(), nil
}

func (driver *MesosSchedulerDriver) RequestResources(requests []*mesos.Request) (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to RequestResources, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if !driver.connected {
		log.Infoln("Ignoring request resource message, disconnected from master.")
		return driver.status, fmt.Errorf("Not connected to master")
	}

	message := &mesos.ResourceRequestMessage{
		FrameworkId: driver.FrameworkInfo.Id,
		Requests:    requests,
	}

	if err := driver.send(driver.MasterPid, message); err != nil {
		log.Errorf("Failed to send ResourceRequest message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) DeclineOffer(offerId *mesos.OfferID, filters *mesos.Filters) (mesos.Status, error) {
	// launching an empty task list will decline the offer
	return driver.LaunchTasks([]*mesos.OfferID{offerId}, []*mesos.TaskInfo{}, filters)
}

func (driver *MesosSchedulerDriver) ReviveOffers() (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to ReviveOffers, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring revive offers message, disconnected from master.")
		return driver.Status(), fmt.Errorf("Not connected to master.")
	}

	message := &mesos.ReviveOffersMessage{
		FrameworkId: driver.FrameworkInfo.Id,
	}
	if err := driver.send(driver.MasterPid, message); err != nil {
		log.Errorf("Failed to send ReviveOffers message: %v\n", err)
		return driver.Status(), err
	}

	return driver.Status(), nil
}

func (driver *MesosSchedulerDriver) SendFrameworkMessage(executorId *mesos.ExecutorID, slaveId *mesos.SlaveID, data string) (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to SendFrameworkMessage, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring send framework message, disconnected from master.")
		return driver.Status(), fmt.Errorf("Not connected to master")
	}

	message := &mesos.FrameworkToExecutorMessage{
		SlaveId:     slaveId,
		FrameworkId: driver.FrameworkInfo.Id,
		ExecutorId:  executorId,
		Data:        []byte(data),
	}
	// Use list of cached slaveIds from previous offers.
	// Send frameworkMessage directly to cached slave, otherwise to master.
	if driver.cache.containsSlavePid(slaveId) {
		slavePid := driver.cache.getSlavePid(slaveId)
		if slavePid.Equal(driver.self) {
			return driver.Status(), nil
		}
		if err := driver.send(slavePid, message); err != nil {
			log.Errorf("Failed to send framework to executor message: %v\n", err)
			return driver.Status(), err
		}
	} else {
		// slavePid not cached, send to master.
		if err := driver.send(driver.MasterPid, message); err != nil {
			log.Errorf("Failed to send framework to executor message: %v\n", err)
			return driver.Status(), err
		}
	}

	return driver.Status(), nil
}

func (driver *MesosSchedulerDriver) ReconcileTasks(statuses []*mesos.TaskStatus) (mesos.Status, error) {
	if stat := driver.Status(); stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to ReconcileTasks, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring send Reconcile Tasks message, disconnected from master.")
		return driver.Status(), fmt.Errorf("Not connected to master.")
	}

	message := &mesos.ReconcileTasksMessage{
		FrameworkId: driver.FrameworkInfo.Id,
		Statuses:    statuses,
	}
	if err := driver.send(driver.MasterPid, message); err != nil {
		log.Errorf("Failed to send reconcile tasks message: %v\n", err)
		return driver.Status(), err
	}

	return driver.Status(), nil
}

func (driver *MesosSchedulerDriver) error(err string, abortDriver bool) {
	if abortDriver {
		if driver.Status() == mesos.Status_DRIVER_ABORTED {
			log.V(3).Infoln("Ignoring error message, the driver is aborted!")
			return
		}

		log.Infoln("Aborting driver, got error '", err, "'")

		driver.Abort()
	}

	log.V(3).Infof("Sending error '%v'", err)
	driver.Scheduler.Error(driver, err)
}
