package schedulerhints

import (
	"regexp"

	"github.com/gophercloud/gophercloud"
)

// SchedulerHints represents a set of scheduling hints that are passed to the
// OpenStack scheduler.
type SchedulerHints struct {
	// DifferentHost will place the volume on a different back-end that does not
	// host the given volumes.
	DifferentHost []string

	// SameHost will place the volume on a back-end that hosts the given volumes.
	SameHost []string

	// LocalToInstance will place volume on same host on a given instance
	LocalToInstance string

	// Query is a conditional statement that results in back-ends able to
	// host the volume.
	Query string

	// AdditionalProperies are arbitrary key/values that are not validated by nova.
	AdditionalProperties map[string]interface{}
}

// VolumeCreateOptsBuilder allows extensions to add additional parameters to the
// Create request.
type VolumeCreateOptsBuilder interface {
	ToVolumeCreateMap() (map[string]interface{}, error)
}

// CreateOptsBuilder builds the scheduler hints into a serializable format.
type CreateOptsBuilder interface {
	ToVolumeSchedulerHintsCreateMap() (map[string]interface{}, error)
}

// ToVolumeSchedulerHintsMap builds the scheduler hints into a serializable format.
func (opts SchedulerHints) ToVolumeSchedulerHintsCreateMap() (map[string]interface{}, error) {
	sh := make(map[string]interface{})

	uuidRegex, _ := regexp.Compile("^[a-z0-9]{8}-[a-z0-9]{4}-[1-5][a-z0-9]{3}-[a-z0-9]{4}-[a-z0-9]{12}$")

	if len(opts.DifferentHost) > 0 {
		for _, diffHost := range opts.DifferentHost {
			if !uuidRegex.MatchString(diffHost) {
				err := gophercloud.ErrInvalidInput{}
				err.Argument = "schedulerhints.SchedulerHints.DifferentHost"
				err.Value = opts.DifferentHost
				err.Info = "The hosts must be in UUID format."
				return nil, err
			}
		}
		sh["different_host"] = opts.DifferentHost
	}

	if len(opts.SameHost) > 0 {
		for _, sameHost := range opts.SameHost {
			if !uuidRegex.MatchString(sameHost) {
				err := gophercloud.ErrInvalidInput{}
				err.Argument = "schedulerhints.SchedulerHints.SameHost"
				err.Value = opts.SameHost
				err.Info = "The hosts must be in UUID format."
				return nil, err
			}
		}
		sh["same_host"] = opts.SameHost
	}

	if opts.LocalToInstance != "" {
		if !uuidRegex.MatchString(opts.LocalToInstance) {
			err := gophercloud.ErrInvalidInput{}
			err.Argument = "schedulerhints.SchedulerHints.LocalToInstance"
			err.Value = opts.LocalToInstance
			err.Info = "The instance must be in UUID format."
			return nil, err
		}
		sh["local_to_instance"] = opts.LocalToInstance
	}

	if opts.Query != "" {
		sh["query"] = opts.Query
	}

	if opts.AdditionalProperties != nil {
		for k, v := range opts.AdditionalProperties {
			sh[k] = v
		}
	}

	return sh, nil
}

// CreateOptsExt adds a SchedulerHints option to the base CreateOpts.
type CreateOptsExt struct {
	VolumeCreateOptsBuilder

	// SchedulerHints provides a set of hints to the scheduler.
	SchedulerHints CreateOptsBuilder
}

// ToVolumeCreateMap adds the SchedulerHints option to the base volume creation options.
func (opts CreateOptsExt) ToVolumeCreateMap() (map[string]interface{}, error) {
	base, err := opts.VolumeCreateOptsBuilder.ToVolumeCreateMap()
	if err != nil {
		return nil, err
	}

	schedulerHints, err := opts.SchedulerHints.ToVolumeSchedulerHintsCreateMap()
	if err != nil {
		return nil, err
	}

	if len(schedulerHints) == 0 {
		return base, nil
	}

	base["OS-SCH-HNT:scheduler_hints"] = schedulerHints

	return base, nil
}
