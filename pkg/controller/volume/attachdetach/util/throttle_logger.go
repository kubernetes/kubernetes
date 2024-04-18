package util

import (
	"context"
	"sync"
	"time"

	"k8s.io/apimachinery/pkg/util/wait"
	"k8s.io/klog/v2"
	"k8s.io/utils/pointer"
)

const defaultInterval = 1 * time.Minute

// ThrottleLogger provides throttling functionality and rate control to the logging output.
// It limits the log repetition for same key in specific time duration.
type ThrottleLogger struct {
	// logger is the underlying logger that messages are written to.
	logger klog.Logger
	// lastLogTimes maps unique keys to the last time a message was logged.
	lastLogTimes map[string]time.Time
	// logInterval is the minimum time between log messages for a given key.
	logInterval time.Duration
	// expireDuration is the maximum time an item of the lastLogTimes map can be idle before it is deleted.
	expireDuration *time.Duration
	// gcPeriod is the time between garbage collection runs.
	gcPeriod *time.Duration
	// mutex locks access to the lastLogTimes map.
	mutex sync.Mutex
}

// Log logs only if the key has not been logged or the gap since last log is more than logInterval.
func (t *ThrottleLogger) Log(level int, uniqueName, msg string, keysAndValues ...interface{}) {
	t.mutex.Lock()
	defer t.mutex.Unlock()

	// Log the message if uniqueName log entry does not exist or if it has not been logged within the logInterval.
	if lastLogTime, ok := t.lastLogTimes[uniqueName]; !ok || time.Since(lastLogTime) > t.logInterval {
		t.logger.V(level).Info(msg, keysAndValues...)
		t.lastLogTimes[uniqueName] = time.Now()
	}
}

// garbageCollectFunc checks and cleans up stale entries in lastLogTimes map.
// If any log key has last log time older than expireDuration then that the entry is removed from map.
func (t *ThrottleLogger) garbageCollectFunc() func(ctx context.Context) {
	return func(ctx context.Context) {
		t.mutex.Lock()
		defer t.mutex.Unlock()

		for key, lastLogTime := range t.lastLogTimes {
			if time.Since(lastLogTime) > t.logInterval {
				delete(t.lastLogTimes, key)
			}
		}
	}
}

// Option is a function that sets some option for ThrottleLogger.
type Option func(*ThrottleLogger)

func WithInterval(interval time.Duration) Option {
	return func(t *ThrottleLogger) {
		t.logInterval = interval
	}
}

func WithExpireDuration(expireDuration time.Duration) Option {
	return func(t *ThrottleLogger) {
		t.expireDuration = pointer.Duration(expireDuration)
	}
}

func WithGCPeriod(gcPeriod time.Duration) Option {
	return func(t *ThrottleLogger) {
		t.gcPeriod = pointer.Duration(gcPeriod)
	}
}

// NewThrottleLogger creates a new ThrottleLogger instance.
// It also starts a separate goroutine to call `garbageCollectFunc` every `gcPeriod` time.
func NewThrottleLogger(ctx context.Context, logger klog.Logger, opts ...Option) *ThrottleLogger {
	tl := &ThrottleLogger{
		logger:       logger,
		lastLogTimes: make(map[string]time.Time),
		logInterval:  defaultInterval,
		mutex:        sync.Mutex{},
	}

	for _, opt := range opts {
		opt(tl)
	}

	// Set default values if not provided.
	if tl.expireDuration == nil {
		tl.expireDuration = pointer.Duration(tl.logInterval * 2)
	}
	if tl.gcPeriod == nil {
		tl.gcPeriod = pointer.Duration(tl.logInterval)
	}

	go wait.UntilWithContext(ctx, tl.garbageCollectFunc(), *tl.gcPeriod)
	return tl
}
