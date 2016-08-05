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
	"sync/atomic"
	"time"

	"github.com/gogo/protobuf/proto"
	log "github.com/golang/glog"
	"github.com/mesos/mesos-go/auth"
	"github.com/mesos/mesos-go/detector"
	mesos "github.com/mesos/mesos-go/mesosproto"
	"github.com/mesos/mesos-go/mesosproto/scheduler"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/mesos/mesos-go/mesosutil/process"
	"github.com/mesos/mesos-go/messenger"
	"github.com/mesos/mesos-go/messenger/sessionid"
	"github.com/mesos/mesos-go/upid"
	"github.com/pborman/uuid"
	"golang.org/x/net/context"
)

const (
	defaultAuthenticationTimeout = 30 * time.Second // timeout interval for an authentication attempt
	registrationRetryIntervalMax = float64(1 * time.Minute)
	registrationBackoffFactor    = 2 * time.Second
)

var (
	ErrDisconnected           = errors.New("disconnected from mesos master")
	errAuthenticationCanceled = errors.New("authentication canceled")
)

type ErrDriverAborted struct {
	Reason string
}

func (err *ErrDriverAborted) Error() string {
	if err.Reason != "" {
		return err.Reason
	}
	return "driver-aborted"
}

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
	PublishedAddress net.IP                                // optional
	NewMessenger     func() (messenger.Messenger, error)   // optional
	NewDetector      func() (detector.Master, error)       // optional
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
	masterPid       *upid.UPID
	frameworkInfo   *mesos.FrameworkInfo
	self            *upid.UPID
	stopCh          chan struct{}
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
	dispatch        func(context.Context, *upid.UPID, proto.Message) error // send a message somewhere
	started         chan struct{}                                          // signal chan that closes upon a successful call to Start()
	eventLock       sync.RWMutex                                           // guard for all driver state
	withScheduler   func(f func(s Scheduler))                              // execute some func with respect to the given scheduler; should be the last thing invoked in a handler (lock semantics)
	done            chan struct{}                                          // signal chan that closes when no more events will be processed
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

	framework := proto.Clone(config.Framework).(*mesos.FrameworkInfo)

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
		frameworkInfo:   framework,
		stopCh:          make(chan struct{}),
		status:          mesos.Status_DRIVER_NOT_STARTED,
		cache:           newSchedCache(),
		credential:      config.Credential,
		failover:        framework.Id != nil && len(framework.Id.GetValue()) > 0,
		withAuthContext: config.WithAuthContext,
		started:         make(chan struct{}),
		done:            make(chan struct{}),
	}

	driver.withScheduler = driver.makeWithScheduler(config.Scheduler)

	if framework.FailoverTimeout != nil && *framework.FailoverTimeout > 0 {
		driver.failoverTimeout = *framework.FailoverTimeout * float64(time.Second)
		log.V(1).Infof("found failover_timeout = %v", time.Duration(driver.failoverTimeout))
	}

	newDetector := config.NewDetector
	if newDetector == nil {
		newDetector = func() (detector.Master, error) {
			return detector.New(config.Master)
		}
	}
	newMessenger := config.NewMessenger
	if newMessenger == nil {
		newMessenger = func() (messenger.Messenger, error) {
			process := process.New("scheduler")
			return messenger.ForHostname(process, hostname, config.BindingAddress, config.BindingPort, config.PublishedAddress)
		}
	}

	// initialize new detector.
	if driver.masterDetector, err = newDetector(); err != nil {
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

func (driver *MesosSchedulerDriver) makeWithScheduler(cs Scheduler) func(func(Scheduler)) {
	// mechanism that allows us to asynchronously invoke scheduler callbacks, but in a manner
	// such that the callback invocations are serialized. useful because this will decouple the
	// goroutine executing a messenger callback from the goroutine executing a scheduler callback,
	// while preserving the serialization semantics for each type of callback handling.
	// we use a chan to maintain the order of callback invocations; this is important for maintaining
	// the order in which status updates are processed.
	schedQueue := make(chan func(s Scheduler))
	go func() {
		defer func() {
			close(driver.done)
			log.V(1).Infoln("finished processing scheduler events")
		}()
		for f := range schedQueue {
			f(cs)
		}
	}()

	var schedLock sync.Mutex // synchronize write access to schedQueue
	abort := int32(0)

	// assume that when withScheduler is invoked eventLock is locked
	return func(f func(s Scheduler)) {
		const timeout = 1 * time.Second
		t := time.NewTimer(timeout)
		defer t.Stop()

		trySend := func() (done bool) {
			// don't block while attempting to enqueue a scheduler op; this could
			// take a while depending upon the external scheduler implementation.
			// also, it allows for multiple go-routines to re-compete for the lock
			// every so often - this avoids indefinitely blocking a call to Abort().
			driver.eventLock.Unlock()
			schedLock.Lock()
			defer func() {
				schedLock.Unlock()
				driver.eventLock.Lock()
			}()

			if atomic.LoadInt32(&abort) == 1 {
				// can't send anymore
				return true
			}

			// try to write to event queue...
			select {
			case schedQueue <- f:
				done = true
			case <-driver.stopCh:
				done = true
			case <-t.C:
			}

			// if stopping then close out the queue (keeping this check separate from
			// the above on purpose! otherwise we could miss the close signal)
			select {
			case <-driver.stopCh:
				if atomic.CompareAndSwapInt32(&abort, 0, 1) {
					defer close(schedQueue)
					log.V(1).Infoln("stopping scheduler event queue..")

					// one last attempt, before we run out of time
					select {
					case schedQueue <- f:
					case <-t.C:
					}
				}
			default:
			}
			return
		}
		for !trySend() {
			t.Reset(timeout) // TODO(jdef) add jitter to this
		}
		// have to do this outside trySend because here we're guarded by eventLock; it's ok
		// if this happens more then once.
		if atomic.LoadInt32(&abort) == 1 {
			driver.withScheduler = func(f func(_ Scheduler)) {}
		}
	}
}

// ctx returns the current context.Context for the driver, expects to be invoked
// only when eventLock is locked.
func (driver *MesosSchedulerDriver) context() context.Context {
	// set a "session" attribute so that the messenger can see it
	// and use it for reporting delivery errors.
	return sessionid.NewContext(context.TODO(), driver.connection.String())
}

// init initializes the driver.
func (driver *MesosSchedulerDriver) init() error {
	log.Infof("Initializing mesos scheduler driver\n")
	driver.dispatch = driver.messenger.Send

	// serialize all callbacks from the messenger
	type messageHandler func(context.Context, *upid.UPID, proto.Message)

	guarded := func(h messageHandler) messenger.MessageHandler {
		return messenger.MessageHandler(func(from *upid.UPID, msg proto.Message) {
			driver.eventLock.Lock()
			defer driver.eventLock.Unlock()
			h(driver.context(), from, msg)
		})
	}

	// Install handlers.
	driver.messenger.Install(guarded(driver.frameworkRegistered), &mesos.FrameworkRegisteredMessage{})
	driver.messenger.Install(guarded(driver.frameworkReregistered), &mesos.FrameworkReregisteredMessage{})
	driver.messenger.Install(guarded(driver.resourcesOffered), &mesos.ResourceOffersMessage{})
	driver.messenger.Install(guarded(driver.resourceOfferRescinded), &mesos.RescindResourceOfferMessage{})
	driver.messenger.Install(guarded(driver.statusUpdated), &mesos.StatusUpdateMessage{})
	driver.messenger.Install(guarded(driver.slaveLost), &mesos.LostSlaveMessage{})
	driver.messenger.Install(guarded(driver.frameworkMessageRcvd), &mesos.ExecutorToFrameworkMessage{})
	driver.messenger.Install(guarded(driver.frameworkErrorRcvd), &mesos.FrameworkErrorMessage{})
	driver.messenger.Install(guarded(driver.exitedExecutor), &mesos.ExitedExecutorMessage{})
	driver.messenger.Install(guarded(driver.handleMasterChanged), &mesos.InternalMasterChangeDetected{})
	driver.messenger.Install(guarded(driver.handleAuthenticationResult), &mesos.InternalAuthenticationResult{})
	driver.messenger.Install(guarded(driver.handleNetworkError), &mesos.InternalNetworkError{})
	return nil
}

func (driver *MesosSchedulerDriver) handleNetworkError(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	msg := pbMsg.(*mesos.InternalNetworkError)

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.Info("ignoring network error because the driver is aborted.")
		return
	} else if !from.Equal(driver.self) {
		log.Errorf("ignoring network error because message received from upid '%v'", from)
		return
	} else if !driver.connected {
		log.V(1).Infof("ignoring network error since we're not currently connected")
		return
	}

	if driver.masterPid.String() == msg.GetPid() && driver.connection.String() == msg.GetSession() {
		// fire a disconnection event
		log.V(3).Info("Disconnecting scheduler.")

		// need to set all 3 of these at once, since withScheduler() temporarily releases the lock and we don't
		// want inconsistent connection facts
		driver.masterPid = nil
		driver.connected = false
		driver.authenticated = false

		driver.withScheduler(func(s Scheduler) { s.Disconnected(driver) })
		log.Info("master disconnected")
	}
}

// lead master detection callback.
func (driver *MesosSchedulerDriver) handleMasterChanged(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.Info("Ignoring master change because the driver is aborted.")
		return
	} else if !from.Equal(driver.self) {
		log.Errorf("ignoring master changed message received from upid '%v'", from)
		return
	}

	// Reconnect every time a master is detected.
	wasConnected := driver.connected
	driver.connected = false
	driver.authenticated = false

	alertScheduler := false
	if wasConnected {
		log.V(3).Info("Disconnecting scheduler.")
		driver.masterPid = nil
		alertScheduler = true
	}

	msg := pbMsg.(*mesos.InternalMasterChangeDetected)
	master := msg.Master

	if master != nil {
		log.Infof("New master %s detected\n", master.GetPid())

		pid, err := upid.Parse(master.GetPid())
		if err != nil {
			panic("Unable to parse Master's PID value.") // this should not happen.
		}

		driver.masterPid = pid // save for downstream ops.
		defer driver.tryAuthentication()
	} else {
		log.Infoln("No master detected.")
	}
	if alertScheduler {
		driver.withScheduler(func(s Scheduler) { s.Disconnected(driver) })
	}
}

// tryAuthentication expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) tryAuthentication() {
	if driver.authenticated {
		// programming error
		panic("already authenticated")
	}

	masterPid := driver.masterPid // save for referencing later in goroutine
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
			pid := driver.messenger.UPID()
			driver.messenger.Route(context.TODO(), &pid, result)
		}()
		driver.authenticating = authenticating
	} else {
		log.Infoln("No credentials were provided. " +
			"Attempting to register scheduler without authentication.")
		driver.authenticated = true
		go driver.doReliableRegistration(float64(registrationBackoffFactor))
	}
}

func (driver *MesosSchedulerDriver) handleAuthenticationResult(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	if driver.status != mesos.Status_DRIVER_RUNNING {
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
	if driver.masterPid == nil {
		log.Infoln("ignoring authentication result because master is lost")
		driver.authenticating.cancel() // cancel any in-progress background attempt

		// disable future retries until we get a new master
		driver.reauthenticate = false
		return
	}
	msg := pbMsg.(*mesos.InternalAuthenticationResult)
	if driver.reauthenticate || !msg.GetCompleted() || driver.masterPid.String() != msg.GetPid() {
		log.Infof("failed to authenticate with master %v: master changed", driver.masterPid)
		driver.authenticating.cancel() // cancel any in-progress background authentication
		driver.reauthenticate = false
		driver.tryAuthentication()
		return
	}
	if !msg.GetSuccess() {
		log.Errorf("master %v refused authentication", driver.masterPid)
		return
	}
	driver.authenticated = true
	go driver.doReliableRegistration(float64(registrationBackoffFactor))
}

// ------------------------- Accessors ----------------------- //

// Status returns the current driver status
func (driver *MesosSchedulerDriver) Status() mesos.Status {
	driver.eventLock.RLock()
	defer driver.eventLock.RUnlock()
	return driver.status
}

// Running returns true if the driver is in the DRIVER_RUNNING state
func (driver *MesosSchedulerDriver) Running() bool {
	driver.eventLock.RLock()
	defer driver.eventLock.RUnlock()
	return driver.status == mesos.Status_DRIVER_RUNNING
}

// Connected returns true if the driver has a registered (and authenticated, if enabled)
// connection to the leading mesos master
func (driver *MesosSchedulerDriver) Connected() bool {
	driver.eventLock.RLock()
	defer driver.eventLock.RUnlock()
	return driver.connected
}

// stopped returns true if the driver status != DRIVER_RUNNING; expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) stopped() bool {
	return driver.status != mesos.Status_DRIVER_RUNNING
}

// ---------------------- Handlers for Events from Master --------------- //
func (driver *MesosSchedulerDriver) frameworkRegistered(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(2).Infoln("Handling scheduler driver framework registered event.")

	msg := pbMsg.(*mesos.FrameworkRegisteredMessage)
	masterInfo := msg.GetMasterInfo()
	masterPid := masterInfo.GetPid()
	frameworkId := msg.GetFrameworkId()

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.Infof("ignoring FrameworkRegisteredMessage from master %s, driver is aborted", masterPid)
		return
	}

	if driver.connected {
		log.Infoln("ignoring FrameworkRegisteredMessage from master, driver is already connected", masterPid)
		return
	}

	if driver.stopped() {
		log.Infof("ignoring FrameworkRegisteredMessage from master %s, driver is stopped", masterPid)
		return
	}
	if !driver.masterPid.Equal(from) {
		log.Warningf("ignoring framework registered message because it was sent from '%v' instead of leading master '%v'", from, driver.masterPid)
		return
	}

	log.Infof("Framework registered with ID=%s\n", frameworkId.GetValue())
	driver.frameworkInfo.Id = frameworkId // generated by master.

	driver.connected = true
	driver.failover = false
	driver.connection = uuid.NewUUID()
	driver.withScheduler(func(s Scheduler) { s.Registered(driver, frameworkId, masterInfo) })
}

func (driver *MesosSchedulerDriver) frameworkReregistered(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling Scheduler re-registered event.")
	msg := pbMsg.(*mesos.FrameworkReregisteredMessage)

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.Infoln("Ignoring FrameworkReregisteredMessage from master, driver is aborted!")
		return
	}
	if driver.connected {
		log.Infoln("Ignoring FrameworkReregisteredMessage from master,driver is already connected!")
		return
	}
	if !driver.masterPid.Equal(from) {
		log.Warningf("ignoring framework re-registered message because it was sent from '%v' instead of leading master '%v'", from, driver.masterPid)
		return
	}

	// TODO(vv) detect if message was from leading-master (sched.cpp)
	log.Infof("Framework re-registered with ID [%s] ", msg.GetFrameworkId().GetValue())
	driver.connected = true
	driver.failover = false
	driver.connection = uuid.NewUUID()

	driver.withScheduler(func(s Scheduler) { s.Reregistered(driver, msg.GetMasterInfo()) })
}

func (driver *MesosSchedulerDriver) resourcesOffered(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(2).Infoln("Handling resource offers.")

	msg := pbMsg.(*mesos.ResourceOffersMessage)
	if driver.status == mesos.Status_DRIVER_ABORTED {
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

	driver.withScheduler(func(s Scheduler) { s.ResourceOffers(driver, msg.Offers) })
}

func (driver *MesosSchedulerDriver) resourceOfferRescinded(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling resource offer rescinded.")

	msg := pbMsg.(*mesos.RescindResourceOfferMessage)

	if driver.status == mesos.Status_DRIVER_ABORTED {
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
	driver.withScheduler(func(s Scheduler) { s.OfferRescinded(driver, msg.OfferId) })
}

func (driver *MesosSchedulerDriver) send(ctx context.Context, upid *upid.UPID, msg proto.Message) error {
	c := make(chan error, 1)
	go func() { c <- driver.dispatch(ctx, upid, msg) }()

	select {
	case <-ctx.Done():
		<-c // wait for Send(...)
		return ctx.Err()
	case err := <-c:
		return err
	}
}

// statusUpdated expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) statusUpdated(ctx context.Context, from *upid.UPID, pbMsg proto.Message) {
	msg := pbMsg.(*mesos.StatusUpdateMessage)

	if driver.status != mesos.Status_DRIVER_RUNNING {
		log.V(1).Infoln("Ignoring StatusUpdate message, the driver is not running!")
		return
	}
	if !from.Equal(driver.self) {
		if !driver.connected {
			log.V(1).Infoln("Ignoring StatusUpdate message, the driver is not connected!")
			return
		}
		if !driver.masterPid.Equal(from) {
			log.Warningf("ignoring status message because it was sent from '%v' instead of leading master '%v'", from, driver.masterPid)
			return
		}
	}

	log.V(2).Infof("Received status update from %q status source %q", from.String(), msg.GetPid())

	status := msg.Update.GetStatus()

	// see https://github.com/apache/mesos/blob/master/src/sched/sched.cpp#L887
	// If the update does not have a 'uuid', it does not need
	// acknowledging. However, prior to 0.23.0, the update uuid
	// was required and always set. We also don't want to ACK updates
	// that were internally generated. In 0.24.0, we can rely on the
	// update uuid check here, until then we must still check for
	// this being sent from the driver (from == UPID()) or from
	// the master (pid == UPID()).
	// TODO(vinod): Get rid of this logic in 0.25.0 because master
	// and slave correctly set task status in 0.24.0.
	if clearUUID := len(msg.Update.Uuid) == 0 || from.Equal(driver.self) || msg.GetPid() == driver.self.String(); clearUUID {
		status.Uuid = nil
	} else {
		status.Uuid = msg.Update.Uuid
	}

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Not sending StatusUpdate ACK, the driver is aborted!")
	} else {

		// Send StatusUpdate Acknowledgement; see above for the rules.
		// Only send ACK if udpate was not from this driver and spec'd a UUID; this is compat w/ 0.23+
		ackRequired := len(msg.Update.Uuid) > 0 && !from.Equal(driver.self) && msg.GetPid() != driver.self.String()
		if ackRequired {
			ackMsg := &mesos.StatusUpdateAcknowledgementMessage{
				SlaveId:     msg.Update.SlaveId,
				FrameworkId: driver.frameworkInfo.Id,
				TaskId:      msg.Update.Status.TaskId,
				Uuid:        msg.Update.Uuid,
			}

			log.V(2).Infof("Sending ACK for status update %+v to %q", *msg.Update, from.String())
			if err := driver.send(ctx, driver.masterPid, ackMsg); err != nil {
				log.Errorf("Failed to send StatusUpdate ACK message: %v", err)
			}
		} else {
			log.V(2).Infof("Not sending ACK, update is not from slave %q", from.String())
		}
	}
	driver.withScheduler(func(s Scheduler) { s.StatusUpdate(driver, status) })
}

func (driver *MesosSchedulerDriver) exitedExecutor(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling ExitedExceutor event.")
	msg := pbMsg.(*mesos.ExitedExecutorMessage)

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Ignoring ExitedExecutor message, the driver is aborted!")
		return
	}
	if !driver.connected {
		log.V(1).Infoln("Ignoring ExitedExecutor message, the driver is not connected!")
		return
	}
	status := msg.GetStatus()
	log.V(2).Infoln("Lost executor %q from slave %q for framework %q with status %d",
		msg.GetExecutorId().GetValue(),
		msg.GetSlaveId().GetValue(),
		msg.GetFrameworkId().GetValue(),
		status)
	driver.withScheduler(func(s Scheduler) { s.ExecutorLost(driver, msg.GetExecutorId(), msg.GetSlaveId(), int(status)) })
}

func (driver *MesosSchedulerDriver) slaveLost(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling LostSlave event.")

	msg := pbMsg.(*mesos.LostSlaveMessage)

	if driver.status == mesos.Status_DRIVER_ABORTED {
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

	driver.withScheduler(func(s Scheduler) { s.SlaveLost(driver, msg.SlaveId) })
}

func (driver *MesosSchedulerDriver) frameworkMessageRcvd(_ context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling framework message event.")

	msg := pbMsg.(*mesos.ExecutorToFrameworkMessage)

	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.V(1).Infoln("Ignoring framwork message, the driver is aborted!")
		return
	}

	log.V(1).Infoln("Received Framwork Message ", msg.String())

	driver.withScheduler(func(s Scheduler) { s.FrameworkMessage(driver, msg.ExecutorId, msg.SlaveId, string(msg.Data)) })
}

func (driver *MesosSchedulerDriver) frameworkErrorRcvd(ctx context.Context, from *upid.UPID, pbMsg proto.Message) {
	log.V(1).Infoln("Handling framework error event.")
	msg := pbMsg.(*mesos.FrameworkErrorMessage)
	driver.fatal(ctx, msg.GetMessage())
}

// ---------------------- Interface Methods ---------------------- //

// Starts the scheduler driver.
// Returns immediately if an error occurs within start sequence.
func (driver *MesosSchedulerDriver) Start() (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()
	return driver.start()
}

// start expected to be guarded by eventLock
func (driver *MesosSchedulerDriver) start() (mesos.Status, error) {
	select {
	case <-driver.started:
		return driver.status, errors.New("Unable to Start: driver has already been started once.")
	default: // proceed
	}

	log.Infoln("Starting the scheduler driver...")

	if driver.status != mesos.Status_DRIVER_NOT_STARTED {
		return driver.status, fmt.Errorf("Unable to Start, expecting driver status %s, but is %s:", mesos.Status_DRIVER_NOT_STARTED, driver.status)
	}

	// Start the messenger.
	if err := driver.messenger.Start(); err != nil {
		log.Errorf("Scheduler failed to start the messenger: %v\n", err)
		return driver.status, err
	}

	pid := driver.messenger.UPID()
	driver.self = &pid
	driver.status = mesos.Status_DRIVER_RUNNING
	close(driver.started)

	log.Infof("Mesos scheduler driver started with PID=%v", driver.self)

	listener := detector.OnMasterChanged(func(m *mesos.MasterInfo) {
		driver.messenger.Route(context.TODO(), driver.self, &mesos.InternalMasterChangeDetected{
			Master: m,
		})
	})

	if driver.masterDetector != nil {
		// register with Detect() AFTER we have a self pid from the messenger, otherwise things get ugly
		// because our internal messaging depends on it. detector callbacks are routed over the messenger
		// bus, maintaining serial (concurrency-safe) callback execution.
		log.V(1).Infof("starting master detector %T: %+v", driver.masterDetector, driver.masterDetector)
		driver.masterDetector.Detect(listener)
		log.V(2).Infoln("master detector started")
	}
	return driver.status, nil
}

// authenticate against the spec'd master pid using the configured authenticationProvider.
// the authentication process is canceled upon either cancelation of authenticating, or
// else because it timed out (see defaultAuthenticationTimeout, auth.Timeout).
//
// TODO(jdef) perhaps at some point in the future this will get pushed down into
// the messenger layer (e.g. to use HTTP-based authentication). We'd probably still
// specify the callback.Handler here, along with the user-selected authentication
// provider. Perhaps in the form of some messenger.AuthenticationConfig.
//
func (driver *MesosSchedulerDriver) authenticate(pid *upid.UPID, authenticating *authenticationAttempt) error {
	log.Infof("authenticating with master %v", pid)

	var (
		authTimeout = defaultAuthenticationTimeout
		ctx         = driver.withAuthContext(context.TODO())
		handler     = &CredentialHandler{
			pid:        pid,
			client:     driver.self,
			credential: driver.credential,
		}
	)

	// check for authentication timeout override
	if d, ok := auth.TimeoutFrom(ctx); ok {
		authTimeout = d
	}

	ctx, cancel := context.WithTimeout(ctx, authTimeout)
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
		return errAuthenticationCanceled
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
		// duration between 0 and 'maxBackoff' (jitter).
		delay := time.Duration(maxBackoff * rand.Float64())

		log.V(1).Infof("will retry registration in %v if necessary", delay)

		t := time.NewTimer(delay)
		defer t.Stop()

		select {
		case <-driver.stopCh:
			return
		case <-t.C:
			maxBackoff *= 2
		}
	}
}

// registerOnce returns true if we should attempt another registration later; it is *not*
// guarded by eventLock: all access to mutable members of MesosSchedulerDriver should be
// explicitly synchronized.
func (driver *MesosSchedulerDriver) registerOnce() bool {
	var (
		failover bool
		pid      *upid.UPID
		info     *mesos.FrameworkInfo
		ctx      context.Context
	)
	if func() bool {
		driver.eventLock.RLock()
		defer driver.eventLock.RUnlock()

		if driver.stopped() || driver.connected || driver.masterPid == nil || (driver.credential != nil && !driver.authenticated) {
			log.V(1).Infof("skipping registration request: stopped=%v, connected=%v, authenticated=%v",
				driver.stopped(), driver.connected, driver.authenticated)
			return false
		}
		failover = driver.failover
		pid = driver.masterPid
		info = proto.Clone(driver.frameworkInfo).(*mesos.FrameworkInfo)
		ctx = driver.context()
		return true
	}() {
		// register framework
		var message proto.Message
		if len(info.GetId().GetValue()) > 0 {
			// not the first time, or failing over
			log.V(1).Infof("Reregistering with master: %v", pid)
			message = &mesos.ReregisterFrameworkMessage{
				Framework: info,
				Failover:  proto.Bool(failover),
			}
		} else {
			log.V(1).Infof("Registering with master: %v", pid)
			message = &mesos.RegisterFrameworkMessage{
				Framework: info,
			}
		}
		if err := driver.send(ctx, pid, message); err != nil {
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
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()
	return driver.join()
}

// join expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) join() (stat mesos.Status, err error) {
	if stat = driver.status; stat != mesos.Status_DRIVER_RUNNING {
		err = fmt.Errorf("Unable to Join, expecting driver status %s, but is %s", mesos.Status_DRIVER_RUNNING, stat)
		return
	}

	timeout := 1 * time.Second
	t := time.NewTimer(timeout)
	defer t.Stop()

	driver.eventLock.Unlock()
	defer func() {
		driver.eventLock.Lock()
		stat = driver.status
	}()
waitForDeath:
	for {
		select {
		case <-driver.done:
			break waitForDeath
		case <-t.C:
		}
		t.Reset(timeout)
	}
	return
}

//Run starts and joins driver process and waits to be stopped or aborted.
func (driver *MesosSchedulerDriver) Run() (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()
	return driver.run(driver.context())
}

// run expected to be guarded by eventLock
func (driver *MesosSchedulerDriver) run(ctx context.Context) (mesos.Status, error) {
	stat, err := driver.start()

	if err != nil {
		return driver.stop(ctx, err, false)
	}

	if stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to Run, expecting driver status %s, but is %s:", mesos.Status_DRIVER_RUNNING, driver.status)
	}

	log.Infoln("Scheduler driver running.  Waiting to be stopped.")
	return driver.join()
}

//Stop stops the driver.
func (driver *MesosSchedulerDriver) Stop(failover bool) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()
	return driver.stop(driver.context(), nil, failover)
}

// stop expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) stop(ctx context.Context, cause error, failover bool) (mesos.Status, error) {
	log.Infoln("Stopping the scheduler driver")
	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to Stop, expected driver status %s, but is %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if driver.connected && !failover {
		// unregister the framework
		log.Infoln("Unregistering the scheduler driver")
		message := &mesos.UnregisterFrameworkMessage{
			FrameworkId: driver.frameworkInfo.Id,
		}
		//TODO(jdef) this is actually a little racy: we send an 'unregister' message but then
		// immediately afterward the messenger is stopped in driver._stop(). so the unregister message
		// may not actually end up being sent out.
		if err := driver.send(ctx, driver.masterPid, message); err != nil {
			log.Errorf("Failed to send UnregisterFramework message while stopping driver: %v\n", err)
			if cause == nil {
				cause = &ErrDriverAborted{}
			}
			return driver._stop(cause, mesos.Status_DRIVER_ABORTED)
		}
		time.Sleep(2 * time.Second)
	}

	// stop messenger
	return driver._stop(cause, mesos.Status_DRIVER_STOPPED)
}

// stop expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) _stop(cause error, stopStatus mesos.Status) (mesos.Status, error) {
	// stop messenger
	defer func() {
		select {
		case <-driver.stopCh:
			return
		default:
		}
		close(driver.stopCh)
		// decouple to avoid deadlock (avoid nested withScheduler() invocations)
		go func() {
			driver.eventLock.Lock()
			defer driver.eventLock.Unlock()
			if cause != nil {
				log.V(1).Infof("Sending error via withScheduler: %v", cause)
				driver.withScheduler(func(s Scheduler) { s.Error(driver, cause.Error()) })
			} else {
				// send a noop func, withScheduler needs to see that stopCh is closed
				log.V(1).Infof("Sending kill signal to withScheduler")
				driver.withScheduler(func(_ Scheduler) {})
			}
		}()
	}()
	driver.status = stopStatus
	driver.connected = false
	driver.connection = uuid.UUID{}

	log.Info("stopping messenger")
	err := driver.messenger.Stop()

	log.Infof("Stop() complete with status %v error %v", stopStatus, err)
	return stopStatus, err
}

func (driver *MesosSchedulerDriver) Abort() (stat mesos.Status, err error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()
	return driver.abort(driver.context(), nil)
}

// abort expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) abort(ctx context.Context, cause error) (stat mesos.Status, err error) {
	if driver.masterDetector != nil {
		defer driver.masterDetector.Cancel()
	}

	log.Infof("Aborting framework [%+v]", driver.frameworkInfo.Id)

	if driver.connected {
		_, err = driver.stop(ctx, cause, true)
	} else {
		driver._stop(cause, mesos.Status_DRIVER_ABORTED)
	}

	stat = mesos.Status_DRIVER_ABORTED
	driver.status = stat
	return
}

func (driver *MesosSchedulerDriver) AcceptOffers(offerIds []*mesos.OfferID, operations []*mesos.Offer_Operation, filters *mesos.Filters) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to AcceptOffers, expected driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	ctx := driver.context()
	if !driver.connected {
		err := ErrDisconnected
		for _, operation := range operations {
			if *operation.Type == mesos.Offer_Operation_LAUNCH {
				// decouple lost task processing to avoid deadlock (avoid nested withScheduler() invocations)
				operation := operation
				go func() {
					driver.eventLock.Lock()
					defer driver.eventLock.Unlock()

					for _, task := range operation.Launch.TaskInfos {
						driver.pushLostTask(ctx, task, err.Error())
					}
				}()
			}
		}
		log.Errorf("Failed to send LaunchTask message: %v\n", err)
		return driver.status, err
	}

	okOperations := make([]*mesos.Offer_Operation, 0, len(operations))

	for _, offerId := range offerIds {
		for _, operation := range operations {
			// Keep only the slave PIDs where we run tasks so we can send
			// framework messages directly.
			if !driver.cache.containsOffer(offerId) {
				log.Warningf("Attempting to accept offers with unknown offer %s\n", offerId.GetValue())
				continue
			}

			// Validate
			switch *operation.Type {
			case mesos.Offer_Operation_LAUNCH:
				tasks := []*mesos.TaskInfo{}
				// Set TaskInfo.executor.framework_id, if it's missing.
				for _, task := range operation.Launch.TaskInfos {
					newTask := *task
					if newTask.Executor != nil && newTask.Executor.FrameworkId == nil {
						newTask.Executor.FrameworkId = driver.frameworkInfo.Id
					}
					tasks = append(tasks, &newTask)
				}
				for _, task := range tasks {
					if driver.cache.getOffer(offerId).offer.SlaveId.Equal(task.SlaveId) {
						// cache the tasked slave, for future communication
						pid := driver.cache.getOffer(offerId).slavePid
						driver.cache.putSlavePid(task.SlaveId, pid)
					} else {
						log.Warningf("Attempting to launch task %s with the wrong slaveId offer %s\n", task.TaskId.GetValue(), task.SlaveId.GetValue())
					}
				}
				operation.Launch.TaskInfos = tasks
				okOperations = append(okOperations, operation)
			case mesos.Offer_Operation_RESERVE:
				// Only send reserved resources
				filtered := util.FilterResources(operation.Reserve.Resources, func(res *mesos.Resource) bool { return res.Reservation != nil })
				operation.Reserve.Resources = filtered
				okOperations = append(okOperations, operation)
			case mesos.Offer_Operation_UNRESERVE:
				// Only send reserved resources
				filtered := util.FilterResources(operation.Unreserve.Resources, func(res *mesos.Resource) bool { return res.Reservation != nil })
				operation.Unreserve.Resources = filtered
				okOperations = append(okOperations, operation)
			case mesos.Offer_Operation_CREATE:
				// Only send reserved resources disks with volumes
				filtered := util.FilterResources(operation.Create.Volumes, func(res *mesos.Resource) bool {
					return res.Reservation != nil && res.Disk != nil && res.GetName() == "disk"
				})
				operation.Create.Volumes = filtered
				okOperations = append(okOperations, operation)
			case mesos.Offer_Operation_DESTROY:
				// Only send reserved resources disks with volumes
				filtered := util.FilterResources(operation.Destroy.Volumes, func(res *mesos.Resource) bool {
					return res.Reservation != nil && res.Disk != nil && res.GetName() == "disk"
				})
				operation.Destroy.Volumes = filtered
				okOperations = append(okOperations, operation)
			}
		}

		driver.cache.removeOffer(offerId) // if offer
	}

	// Accept Offers
	message := &scheduler.Call{
		FrameworkId: driver.frameworkInfo.Id,
		Type:        scheduler.Call_ACCEPT.Enum(),
		Accept: &scheduler.Call_Accept{
			OfferIds:   offerIds,
			Operations: okOperations,
			Filters:    filters,
		},
	}

	if err := driver.send(ctx, driver.masterPid, message); err != nil {
		for _, operation := range operations {
			if *operation.Type == mesos.Offer_Operation_LAUNCH {
				// decouple lost task processing to avoid deadlock (avoid nested withScheduler() invocations)
				operation := operation
				go func() {
					driver.eventLock.Lock()
					defer driver.eventLock.Unlock()

					for _, task := range operation.Launch.TaskInfos {
						driver.pushLostTask(ctx, task, "Unable to launch tasks: "+err.Error())
					}
				}()
			}
		}
		log.Errorf("Failed to send LaunchTask message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) LaunchTasks(offerIds []*mesos.OfferID, tasks []*mesos.TaskInfo, filters *mesos.Filters) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to LaunchTasks, expected driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	ctx := driver.context()

	// Launch tasks
	if !driver.connected {
		log.Infoln("Ignoring LaunchTasks message, disconnected from master.")
		// decouple lost task processing to avoid deadlock (avoid nested withScheduler() invocations)
		err := ErrDisconnected
		go func() {
			driver.eventLock.Lock()
			defer driver.eventLock.Unlock()

			// Send statusUpdate with status=TASK_LOST for each task.
			// See sched.cpp L#823
			for _, task := range tasks {
				driver.pushLostTask(ctx, task, err.Error())
			}
		}()
		return driver.status, err
	}

	okTasks := make([]*mesos.TaskInfo, 0, len(tasks))

	// Set TaskInfo.executor.framework_id, if it's missing.
	for _, task := range tasks {
		if task.Executor != nil && task.Executor.FrameworkId == nil {
			task.Executor.FrameworkId = driver.frameworkInfo.Id
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
		FrameworkId: driver.frameworkInfo.Id,
		OfferIds:    offerIds,
		Tasks:       okTasks,
		Filters:     filters,
	}

	if err := driver.send(ctx, driver.masterPid, message); err != nil {
		// decouple lost task processing to avoid deadlock (avoid nested withScheduler() invocations)
		go func() {
			driver.eventLock.Lock()
			defer driver.eventLock.Unlock()

			for _, task := range tasks {
				driver.pushLostTask(ctx, task, "Unable to launch tasks: "+err.Error())
			}
		}()
		log.Errorf("Failed to send LaunchTask message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

// pushLostTask expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) pushLostTask(ctx context.Context, taskInfo *mesos.TaskInfo, why string) {
	msg := &mesos.StatusUpdateMessage{
		Update: &mesos.StatusUpdate{
			FrameworkId: driver.frameworkInfo.Id,
			Status: &mesos.TaskStatus{
				TaskId:  taskInfo.TaskId,
				State:   mesos.TaskState_TASK_LOST.Enum(),
				Source:  mesos.TaskStatus_SOURCE_MASTER.Enum(),
				Message: proto.String(why),
				Reason:  mesos.TaskStatus_REASON_MASTER_DISCONNECTED.Enum(),
			},
			SlaveId:    taskInfo.SlaveId,
			ExecutorId: taskInfo.Executor.ExecutorId,
			Timestamp:  proto.Float64(float64(time.Now().Unix())),
		},
		Pid: proto.String(driver.self.String()),
	}

	// put it on internal chanel
	// will cause handler to push to attached Scheduler
	driver.statusUpdated(ctx, driver.self, msg)
}

func (driver *MesosSchedulerDriver) KillTask(taskId *mesos.TaskID) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to KillTask, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if !driver.connected {
		log.Infoln("Ignoring kill task message, disconnected from master.")
		return driver.status, ErrDisconnected
	}

	message := &mesos.KillTaskMessage{
		FrameworkId: driver.frameworkInfo.Id,
		TaskId:      taskId,
	}

	if err := driver.send(driver.context(), driver.masterPid, message); err != nil {
		log.Errorf("Failed to send KillTask message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) RequestResources(requests []*mesos.Request) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to RequestResources, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}

	if !driver.connected {
		log.Infoln("Ignoring request resource message, disconnected from master.")
		return driver.status, ErrDisconnected
	}

	message := &mesos.ResourceRequestMessage{
		FrameworkId: driver.frameworkInfo.Id,
		Requests:    requests,
	}

	if err := driver.send(driver.context(), driver.masterPid, message); err != nil {
		log.Errorf("Failed to send ResourceRequest message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) DeclineOffer(offerId *mesos.OfferID, filters *mesos.Filters) (mesos.Status, error) {
	// NOTE: don't lock eventLock here because we're delegating to LaunchTasks() and that does it for us
	// launching an empty task list will decline the offer
	return driver.LaunchTasks([]*mesos.OfferID{offerId}, []*mesos.TaskInfo{}, filters)
}

func (driver *MesosSchedulerDriver) ReviveOffers() (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to ReviveOffers, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring revive offers message, disconnected from master.")
		return driver.status, ErrDisconnected
	}

	message := &mesos.ReviveOffersMessage{
		FrameworkId: driver.frameworkInfo.Id,
	}
	if err := driver.send(driver.context(), driver.masterPid, message); err != nil {
		log.Errorf("Failed to send ReviveOffers message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) SendFrameworkMessage(executorId *mesos.ExecutorID, slaveId *mesos.SlaveID, data string) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to SendFrameworkMessage, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring send framework message, disconnected from master.")
		return driver.status, ErrDisconnected
	}

	message := &mesos.FrameworkToExecutorMessage{
		SlaveId:     slaveId,
		FrameworkId: driver.frameworkInfo.Id,
		ExecutorId:  executorId,
		Data:        []byte(data),
	}
	// Use list of cached slaveIds from previous offers.
	// Send frameworkMessage directly to cached slave, otherwise to master.
	if driver.cache.containsSlavePid(slaveId) {
		slavePid := driver.cache.getSlavePid(slaveId)
		if slavePid.Equal(driver.self) {
			return driver.status, nil
		}
		if err := driver.send(driver.context(), slavePid, message); err != nil {
			log.Errorf("Failed to send framework to executor message: %v\n", err)
			return driver.status, err
		}
	} else {
		// slavePid not cached, send to master.
		if err := driver.send(driver.context(), driver.masterPid, message); err != nil {
			log.Errorf("Failed to send framework to executor message: %v\n", err)
			return driver.status, err
		}
	}

	return driver.status, nil
}

func (driver *MesosSchedulerDriver) ReconcileTasks(statuses []*mesos.TaskStatus) (mesos.Status, error) {
	driver.eventLock.Lock()
	defer driver.eventLock.Unlock()

	if stat := driver.status; stat != mesos.Status_DRIVER_RUNNING {
		return stat, fmt.Errorf("Unable to ReconcileTasks, expecting driver status %s, but got %s", mesos.Status_DRIVER_RUNNING, stat)
	}
	if !driver.connected {
		log.Infoln("Ignoring send Reconcile Tasks message, disconnected from master.")
		return driver.status, ErrDisconnected
	}

	message := &mesos.ReconcileTasksMessage{
		FrameworkId: driver.frameworkInfo.Id,
		Statuses:    statuses,
	}
	if err := driver.send(driver.context(), driver.masterPid, message); err != nil {
		log.Errorf("Failed to send reconcile tasks message: %v\n", err)
		return driver.status, err
	}

	return driver.status, nil
}

// error expects to be guarded by eventLock
func (driver *MesosSchedulerDriver) fatal(ctx context.Context, err string) {
	if driver.status == mesos.Status_DRIVER_ABORTED {
		log.V(3).Infoln("Ignoring error message, the driver is aborted!")
		return
	}
	driver.abort(ctx, &ErrDriverAborted{Reason: err})
}
