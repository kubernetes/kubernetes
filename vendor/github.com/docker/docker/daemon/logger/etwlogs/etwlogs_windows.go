// Package etwlogs provides a log driver for forwarding container logs
// as ETW events.(ETW stands for Event Tracing for Windows)
// A client can then create an ETW listener to listen for events that are sent
// by the ETW provider that we register, using the provider's GUID "a3693192-9ed6-46d2-a981-f8226c8363bd".
// Here is an example of how to do this using the logman utility:
// 1. logman start -ets DockerContainerLogs -p {a3693192-9ed6-46d2-a981-f8226c8363bd} 0 0 -o trace.etl
// 2. Run container(s) and generate log messages
// 3. logman stop -ets DockerContainerLogs
// 4. You can then convert the etl log file to XML using: tracerpt -y trace.etl
//
// Each container log message generates an ETW event that also contains:
// the container name and ID, the timestamp, and the stream type.
package etwlogs

import (
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/docker/docker/daemon/logger"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"
)

type etwLogs struct {
	containerName string
	imageName     string
	containerID   string
	imageID       string
}

const (
	name             = "etwlogs"
	win32CallSuccess = 0
)

var (
	modAdvapi32          = windows.NewLazySystemDLL("Advapi32.dll")
	procEventRegister    = modAdvapi32.NewProc("EventRegister")
	procEventWriteString = modAdvapi32.NewProc("EventWriteString")
	procEventUnregister  = modAdvapi32.NewProc("EventUnregister")
)
var providerHandle windows.Handle
var refCount int
var mu sync.Mutex

func init() {
	providerHandle = windows.InvalidHandle
	if err := logger.RegisterLogDriver(name, New); err != nil {
		logrus.Fatal(err)
	}
}

// New creates a new etwLogs logger for the given container and registers the EWT provider.
func New(info logger.Info) (logger.Logger, error) {
	if err := registerETWProvider(); err != nil {
		return nil, err
	}
	logrus.Debugf("logging driver etwLogs configured for container: %s.", info.ContainerID)

	return &etwLogs{
		containerName: info.Name(),
		imageName:     info.ContainerImageName,
		containerID:   info.ContainerID,
		imageID:       info.ContainerImageID,
	}, nil
}

// Log logs the message to the ETW stream.
func (etwLogger *etwLogs) Log(msg *logger.Message) error {
	if providerHandle == windows.InvalidHandle {
		// This should never be hit, if it is, it indicates a programming error.
		errorMessage := "ETWLogs cannot log the message, because the event provider has not been registered."
		logrus.Error(errorMessage)
		return errors.New(errorMessage)
	}
	m := createLogMessage(etwLogger, msg)
	logger.PutMessage(msg)
	return callEventWriteString(m)
}

// Close closes the logger by unregistering the ETW provider.
func (etwLogger *etwLogs) Close() error {
	unregisterETWProvider()
	return nil
}

func (etwLogger *etwLogs) Name() string {
	return name
}

func createLogMessage(etwLogger *etwLogs, msg *logger.Message) string {
	return fmt.Sprintf("container_name: %s, image_name: %s, container_id: %s, image_id: %s, source: %s, log: %s",
		etwLogger.containerName,
		etwLogger.imageName,
		etwLogger.containerID,
		etwLogger.imageID,
		msg.Source,
		msg.Line)
}

func registerETWProvider() error {
	mu.Lock()
	defer mu.Unlock()
	if refCount == 0 {
		var err error
		if err = callEventRegister(); err != nil {
			return err
		}
	}

	refCount++
	return nil
}

func unregisterETWProvider() {
	mu.Lock()
	defer mu.Unlock()
	if refCount == 1 {
		if callEventUnregister() {
			refCount--
			providerHandle = windows.InvalidHandle
		}
		// Not returning an error if EventUnregister fails, because etwLogs will continue to work
	} else {
		refCount--
	}
}

func callEventRegister() error {
	// The provider's GUID is {a3693192-9ed6-46d2-a981-f8226c8363bd}
	guid := windows.GUID{
		Data1: 0xa3693192,
		Data2: 0x9ed6,
		Data3: 0x46d2,
		Data4: [8]byte{0xa9, 0x81, 0xf8, 0x22, 0x6c, 0x83, 0x63, 0xbd},
	}

	ret, _, _ := procEventRegister.Call(uintptr(unsafe.Pointer(&guid)), 0, 0, uintptr(unsafe.Pointer(&providerHandle)))
	if ret != win32CallSuccess {
		errorMessage := fmt.Sprintf("Failed to register ETW provider. Error: %d", ret)
		logrus.Error(errorMessage)
		return errors.New(errorMessage)
	}
	return nil
}

func callEventWriteString(message string) error {
	utf16message, err := windows.UTF16FromString(message)

	if err != nil {
		return err
	}

	ret, _, _ := procEventWriteString.Call(uintptr(providerHandle), 0, 0, uintptr(unsafe.Pointer(&utf16message[0])))
	if ret != win32CallSuccess {
		errorMessage := fmt.Sprintf("ETWLogs provider failed to log message. Error: %d", ret)
		logrus.Error(errorMessage)
		return errors.New(errorMessage)
	}
	return nil
}

func callEventUnregister() bool {
	ret, _, _ := procEventUnregister.Call(uintptr(providerHandle))
	return ret == win32CallSuccess
}
