package hcs

import (
	"sync"
	"syscall"

	"github.com/Microsoft/hcsshim/internal/interop"
	"github.com/sirupsen/logrus"
)

var (
	nextCallback    uintptr
	callbackMap     = map[uintptr]*notifcationWatcherContext{}
	callbackMapLock = sync.RWMutex{}

	notificationWatcherCallback = syscall.NewCallback(notificationWatcher)

	// Notifications for HCS_SYSTEM handles
	hcsNotificationSystemExited                      hcsNotification = 0x00000001
	hcsNotificationSystemCreateCompleted             hcsNotification = 0x00000002
	hcsNotificationSystemStartCompleted              hcsNotification = 0x00000003
	hcsNotificationSystemPauseCompleted              hcsNotification = 0x00000004
	hcsNotificationSystemResumeCompleted             hcsNotification = 0x00000005
	hcsNotificationSystemCrashReport                 hcsNotification = 0x00000006
	hcsNotificationSystemSiloJobCreated              hcsNotification = 0x00000007
	hcsNotificationSystemSaveCompleted               hcsNotification = 0x00000008
	hcsNotificationSystemRdpEnhancedModeStateChanged hcsNotification = 0x00000009
	hcsNotificationSystemShutdownFailed              hcsNotification = 0x0000000A
	hcsNotificationSystemGetPropertiesCompleted      hcsNotification = 0x0000000B
	hcsNotificationSystemModifyCompleted             hcsNotification = 0x0000000C
	hcsNotificationSystemCrashInitiated              hcsNotification = 0x0000000D
	hcsNotificationSystemGuestConnectionClosed       hcsNotification = 0x0000000E

	// Notifications for HCS_PROCESS handles
	hcsNotificationProcessExited hcsNotification = 0x00010000

	// Common notifications
	hcsNotificationInvalid           hcsNotification = 0x00000000
	hcsNotificationServiceDisconnect hcsNotification = 0x01000000
)

type hcsNotification uint32
type notificationChannel chan error

type notifcationWatcherContext struct {
	channels notificationChannels
	handle   hcsCallback
}

type notificationChannels map[hcsNotification]notificationChannel

func newChannels() notificationChannels {
	channels := make(notificationChannels)

	channels[hcsNotificationSystemExited] = make(notificationChannel, 1)
	channels[hcsNotificationSystemCreateCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemStartCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemPauseCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemResumeCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationProcessExited] = make(notificationChannel, 1)
	channels[hcsNotificationServiceDisconnect] = make(notificationChannel, 1)
	channels[hcsNotificationSystemCrashReport] = make(notificationChannel, 1)
	channels[hcsNotificationSystemSiloJobCreated] = make(notificationChannel, 1)
	channels[hcsNotificationSystemSaveCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemRdpEnhancedModeStateChanged] = make(notificationChannel, 1)
	channels[hcsNotificationSystemShutdownFailed] = make(notificationChannel, 1)
	channels[hcsNotificationSystemGetPropertiesCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemModifyCompleted] = make(notificationChannel, 1)
	channels[hcsNotificationSystemCrashInitiated] = make(notificationChannel, 1)
	channels[hcsNotificationSystemGuestConnectionClosed] = make(notificationChannel, 1)

	return channels
}

func closeChannels(channels notificationChannels) {
	for _, c := range channels {
		close(c)
	}
}

func notificationWatcher(notificationType hcsNotification, callbackNumber uintptr, notificationStatus uintptr, notificationData *uint16) uintptr {
	var result error
	if int32(notificationStatus) < 0 {
		result = interop.Win32FromHresult(notificationStatus)
	}

	callbackMapLock.RLock()
	context := callbackMap[callbackNumber]
	callbackMapLock.RUnlock()

	if context == nil {
		return 0
	}

	if channel, ok := context.channels[notificationType]; ok {
		channel <- result
	} else {
		logrus.WithFields(logrus.Fields{
			"notification-type": notificationType,
		}).Warn("Received a callback of an unsupported type")
	}

	return 0
}
