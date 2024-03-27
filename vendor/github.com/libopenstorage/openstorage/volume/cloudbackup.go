package volume

import (
	"fmt"
	"time"

	"github.com/cenkalti/backoff"
	"github.com/libopenstorage/openstorage/api"
)

var (
	cloudsnapBackoffInitialInterval   = 5 * time.Second
	cloudsnapBackoffDefaultMultiplier = float64(1)
	cloudsnapBackoffMaxInterval       = 2 * time.Minute
	// Wait indefinitely
	cloudsnapBackoffMaxElapsedTime = 0 * time.Second
)

func CloudBackupWaitForCompletion(
	cl CloudBackupDriver,
	taskId string,
	opType api.CloudBackupOpType,
) error {
	cloudsnapBackoff := backoff.NewExponentialBackOff()
	cloudsnapBackoff.InitialInterval = cloudsnapBackoffInitialInterval
	cloudsnapBackoff.Multiplier = cloudsnapBackoffDefaultMultiplier
	cloudsnapBackoff.MaxInterval = cloudsnapBackoffMaxInterval
	cloudsnapBackoff.MaxElapsedTime = cloudsnapBackoffMaxElapsedTime

	var opError error
	err := backoff.Retry(func() error {
		response, err := cl.CloudBackupStatus(&api.CloudBackupStatusRequest{
			ID: taskId,
		})
		if err != nil {
			return err
		}
		csStatus, present := response.Statuses[taskId]
		if !present {
			opError = fmt.Errorf("failed to get cloudsnap status for volume: %s", taskId)
			return nil
		}

		err = fmt.Errorf("CloudBackup operation %v for %v in state %v", opType, taskId, csStatus.Status)
		switch csStatus.Status {
		case api.CloudBackupStatusFailed, api.CloudBackupStatusAborted, api.CloudBackupStatusStopped:
			opError = err
			return nil
		case api.CloudBackupStatusDone:
			opError = nil
			return nil
		case api.CloudBackupStatusNotStarted, api.CloudBackupStatusActive, api.CloudBackupStatusPaused:
			return err
		default:
			opError = err
			return nil
		}
	}, cloudsnapBackoff)
	if err != nil {
		return err
	}

	return opError
}
