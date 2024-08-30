//go:build windows

package jobobject

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"unsafe"

	"github.com/Microsoft/hcnshim/internal/log"
	"github.com/Microsoft/hcnshim/internal/queue"
	"github.com/Microsoft/hcnshim/internal/winapi"
	"github.com/sirupsen/logrus"
	"golang.org/x/sys/windows"
)

var (
	ioInitOnce sync.Once
	initIOErr  error
	// Global iocp handle that will be re-used for every job object
	ioCompletionPort windows.Handle
	// Mapping of job handle to queue to place notifications in.
	jobMap sync.Map
)

// MsgAllProcessesExited is a type representing a message that every process in a job has exited.
type MsgAllProcessesExited struct{}

// MsgUnimplemented represents a message that we are aware of, but that isn't implemented currently.
// This should not be treated as an error.
type MsgUnimplemented struct{}

// pollIOCP polls the io completion port forever.
func pollIOCP(ctx context.Context, iocpHandle windows.Handle) {
	var (
		overlapped uintptr
		code       uint32
		key        uintptr
	)

	for {
		err := windows.GetQueuedCompletionStatus(iocpHandle, &code, &key, (**windows.Overlapped)(unsafe.Pointer(&overlapped)), windows.INFINITE)
		if err != nil {
			log.G(ctx).WithError(err).Error("failed to poll for job object message")
			continue
		}
		if val, ok := jobMap.Load(key); ok {
			msq, ok := val.(*queue.MessageQueue)
			if !ok {
				log.G(ctx).WithField("value", msq).Warn("encountered non queue type in job map")
				continue
			}
			notification, err := parseMessage(code, overlapped)
			if err != nil {
				log.G(ctx).WithFields(logrus.Fields{
					"code":       code,
					"overlapped": overlapped,
				}).Warn("failed to parse job object message")
				continue
			}
			if err := msq.Enqueue(notification); errors.Is(err, queue.ErrQueueClosed) {
				// Write will only return an error when the queue is closed.
				// The only time a queue would ever be closed is when we call `Close` on
				// the job it belongs to which also removes it from the jobMap, so something
				// went wrong here. We can't return as this is reading messages for all jobs
				// so just log it and move on.
				log.G(ctx).WithFields(logrus.Fields{
					"code":       code,
					"overlapped": overlapped,
				}).Warn("tried to write to a closed queue")
				continue
			}
		} else {
			log.G(ctx).Warn("received a message for a job not present in the mapping")
		}
	}
}

func parseMessage(code uint32, overlapped uintptr) (interface{}, error) {
	// Check code and parse out relevant information related to that notification
	// that we care about. For now all we handle is the message that all processes
	// in the job have exited.
	switch code {
	case winapi.JOB_OBJECT_MSG_ACTIVE_PROCESS_ZERO:
		return MsgAllProcessesExited{}, nil
	// Other messages for completeness and a check to make sure that if we fall
	// into the default case that this is a code we don't know how to handle.
	case winapi.JOB_OBJECT_MSG_END_OF_JOB_TIME:
	case winapi.JOB_OBJECT_MSG_END_OF_PROCESS_TIME:
	case winapi.JOB_OBJECT_MSG_ACTIVE_PROCESS_LIMIT:
	case winapi.JOB_OBJECT_MSG_NEW_PROCESS:
	case winapi.JOB_OBJECT_MSG_EXIT_PROCESS:
	case winapi.JOB_OBJECT_MSG_ABNORMAL_EXIT_PROCESS:
	case winapi.JOB_OBJECT_MSG_PROCESS_MEMORY_LIMIT:
	case winapi.JOB_OBJECT_MSG_JOB_MEMORY_LIMIT:
	case winapi.JOB_OBJECT_MSG_NOTIFICATION_LIMIT:
	default:
		return nil, fmt.Errorf("unknown job notification type: %d", code)
	}
	return MsgUnimplemented{}, nil
}

// Assigns an IO completion port to get notified of events for the registered job
// object.
func attachIOCP(job windows.Handle, iocp windows.Handle) error {
	info := winapi.JOBOBJECT_ASSOCIATE_COMPLETION_PORT{
		CompletionKey:  job,
		CompletionPort: iocp,
	}
	_, err := windows.SetInformationJobObject(job, windows.JobObjectAssociateCompletionPortInformation, uintptr(unsafe.Pointer(&info)), uint32(unsafe.Sizeof(info)))
	return err
}
