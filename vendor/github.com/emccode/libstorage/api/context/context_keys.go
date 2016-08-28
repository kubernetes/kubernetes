package context

import (
	"sync"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/goof"
)

// Key is the type used as a context key.
type Key int

const (
	_ Key = -1 - iota

	// LoggerKey is a context key.
	LoggerKey

	// HTTPRequestKey is a context key.
	HTTPRequestKey

	// AllInstanceIDsKey is the key for the map[string]*types.InstanceID value
	// that maps all drivers to their instance IDs.
	AllInstanceIDsKey

	// LocalDevicesKey is a context key.
	LocalDevicesKey

	// AllLocalDevicesKey is the key for the map[string]*types.LocalDevices
	// value that maps all drivers to their instance IDs.
	AllLocalDevicesKey

	// AdminTokenKey is the key for the server's admin token.
	AdminTokenKey

	// keyLoggable is the minimum value from which the succeeding keys should
	// be checked when logging.
	keyLoggable

	// ClientKey is a context key.
	ClientKey

	// TaskKey is a context key.
	TaskKey

	// InstanceIDKey is a context key.
	InstanceIDKey

	// ProfileKey is a context key.
	ProfileKey

	// RouteKey is a context key.
	RouteKey

	// ServerKey is a context key.
	ServerKey

	// ServiceKey is an alias for StorageService.
	ServiceKey

	// StorageServiceKey is a context key.
	StorageServiceKey

	// TransactionKey is a context key.
	TransactionKey

	// DriverKey is an alias for StorageDriver.
	DriverKey

	// UserKey is a context key.
	UserKey

	// HostKey is a context key.
	HostKey

	// TLSKey is a context key.
	TLSKey

	// keyEOF should always be the final key
	keyEOF
)

// String returns the name of the context key.
func (k Key) String() string {
	if v, ok := keyNames[k]; ok {
		return v
	}
	return ""
}

var (
	keyNames = map[Key]string{
		TaskKey:           "task",
		InstanceIDKey:     "instanceID",
		ProfileKey:        "profile",
		RouteKey:          "route",
		ServerKey:         "server",
		ServiceKey:        "service",
		StorageServiceKey: "service",
		TransactionKey:    "tx",
		DriverKey:         "storageDriver",
		UserKey:           "user",
		HostKey:           "host",
		TLSKey:            "tls",
	}
)

// CustomKeyTypes is a bitmask used when registering a custom key with the
// context at runtime.
type CustomKeyTypes int

const (
	// CustomLoggerKey indicates a value set for this key should be logged as
	// a member of a log entry's fields collection when using the context's
	// structured logger.
	CustomLoggerKey CustomKeyTypes = 1 << iota

	// CustomHeaderKey indicates a value set for this key should be sent along
	// with HTTP requests as an HTTP header.
	CustomHeaderKey
)

type customKey struct {
	internalID int
	externalID interface{}
	keyBitmask CustomKeyTypes
}

var (
	customKeys    = map[interface{}]*customKey{}
	customKeysRWL = &sync.RWMutex{}
)

func isCustomKey(key interface{}) (int, bool) {
	return isCustomKeyWithLockOpts(key, true)
}

func isCustomKeyWithLockOpts(key interface{}, lock bool) (int, bool) {

	if lock {
		customKeysRWL.RLock()
		defer customKeysRWL.RUnlock()
	}

	if v, ok := customKeys[key]; ok {
		return v.internalID, true
	}
	return 0, false
}

// RegisterCustomKey registers a custom key with the context package.
func RegisterCustomKey(key interface{}, mask CustomKeyTypes) error {

	customKeysRWL.Lock()
	defer customKeysRWL.Unlock()

	if _, ok := customKeys[key]; ok {
		return goof.WithField("key", key, "key already registered")
	}

	newCustomKey := &customKey{
		internalID: len(customKeys) + 1,
		externalID: key,
		keyBitmask: mask,
	}

	customKeys[newCustomKey.externalID] = newCustomKey

	log.WithFields(log.Fields{
		"internalID": newCustomKey.internalID,
		"externalID": newCustomKey.externalID,
		"keyBitmask": newCustomKey.keyBitmask,
	}).Info("registered custom context key")

	return nil
}

// CustomHeaderKeys returns a channel on which can be received all the
// registered, custom header keys.
func CustomHeaderKeys() <-chan interface{} {
	c := make(chan interface{})

	go func() {
		customKeysRWL.RLock()
		defer customKeysRWL.RUnlock()

		for _, v := range customKeys {
			if (v.keyBitmask & CustomHeaderKey) == CustomHeaderKey {
				c <- v.externalID
			}
		}

		close(c)
	}()

	return c
}

// CustomLoggerKeys returns a channel on which can be received all the
// registered, custom logger keys.
func CustomLoggerKeys() <-chan interface{} {

	c := make(chan interface{})

	go func() {
		customKeysRWL.RLock()
		defer customKeysRWL.RUnlock()

		for _, v := range customKeys {
			if (v.keyBitmask & CustomLoggerKey) == CustomLoggerKey {
				c <- v.externalID
			}
		}

		close(c)
	}()

	return c
}
