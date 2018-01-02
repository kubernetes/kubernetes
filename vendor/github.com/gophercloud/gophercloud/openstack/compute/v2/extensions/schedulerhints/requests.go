package schedulerhints

import (
	"net"
	"regexp"
	"strings"

	"github.com/gophercloud/gophercloud"
	"github.com/gophercloud/gophercloud/openstack/compute/v2/servers"
)

// SchedulerHints represents a set of scheduling hints that are passed to the
// OpenStack scheduler.
type SchedulerHints struct {
	// Group specifies a Server Group to place the instance in.
	Group string

	// DifferentHost will place the instance on a compute node that does not
	// host the given instances.
	DifferentHost []string

	// SameHost will place the instance on a compute node that hosts the given
	// instances.
	SameHost []string

	// Query is a conditional statement that results in compute nodes able to
	// host the instance.
	Query []interface{}

	// TargetCell specifies a cell name where the instance will be placed.
	TargetCell string `json:"target_cell,omitempty"`

	// BuildNearHostIP specifies a subnet of compute nodes to host the instance.
	BuildNearHostIP string

	// AdditionalProperies are arbitrary key/values that are not validated by nova.
	AdditionalProperties map[string]interface{}
}

// CreateOptsBuilder builds the scheduler hints into a serializable format.
type CreateOptsBuilder interface {
	ToServerSchedulerHintsCreateMap() (map[string]interface{}, error)
}

// ToServerSchedulerHintsMap builds the scheduler hints into a serializable format.
func (opts SchedulerHints) ToServerSchedulerHintsCreateMap() (map[string]interface{}, error) {
	sh := make(map[string]interface{})

	uuidRegex, _ := regexp.Compile("^[a-z0-9]{8}-[a-z0-9]{4}-[1-5][a-z0-9]{3}-[a-z0-9]{4}-[a-z0-9]{12}$")

	if opts.Group != "" {
		if !uuidRegex.MatchString(opts.Group) {
			err := gophercloud.ErrInvalidInput{}
			err.Argument = "schedulerhints.SchedulerHints.Group"
			err.Value = opts.Group
			err.Info = "Group must be a UUID"
			return nil, err
		}
		sh["group"] = opts.Group
	}

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

	/*
		Query can be something simple like:
			 [">=", "$free_ram_mb", 1024]

			Or more complex like:
				['and',
					['>=', '$free_ram_mb', 1024],
					['>=', '$free_disk_mb', 200 * 1024]
				]

		Because of the possible complexity, just make sure the length is a minimum of 3.
	*/
	if len(opts.Query) > 0 {
		if len(opts.Query) < 3 {
			err := gophercloud.ErrInvalidInput{}
			err.Argument = "schedulerhints.SchedulerHints.Query"
			err.Value = opts.Query
			err.Info = "Must be a conditional statement in the format of [op,variable,value]"
			return nil, err
		}
		sh["query"] = opts.Query
	}

	if opts.TargetCell != "" {
		sh["target_cell"] = opts.TargetCell
	}

	if opts.BuildNearHostIP != "" {
		if _, _, err := net.ParseCIDR(opts.BuildNearHostIP); err != nil {
			err := gophercloud.ErrInvalidInput{}
			err.Argument = "schedulerhints.SchedulerHints.BuildNearHostIP"
			err.Value = opts.BuildNearHostIP
			err.Info = "Must be a valid subnet in the form 192.168.1.1/24"
			return nil, err
		}
		ipParts := strings.Split(opts.BuildNearHostIP, "/")
		sh["build_near_host_ip"] = ipParts[0]
		sh["cidr"] = "/" + ipParts[1]
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
	servers.CreateOptsBuilder

	// SchedulerHints provides a set of hints to the scheduler.
	SchedulerHints CreateOptsBuilder
}

// ToServerCreateMap adds the SchedulerHints option to the base server creation options.
func (opts CreateOptsExt) ToServerCreateMap() (map[string]interface{}, error) {
	base, err := opts.CreateOptsBuilder.ToServerCreateMap()
	if err != nil {
		return nil, err
	}

	schedulerHints, err := opts.SchedulerHints.ToServerSchedulerHintsCreateMap()
	if err != nil {
		return nil, err
	}

	if len(schedulerHints) == 0 {
		return base, nil
	}

	base["os:scheduler_hints"] = schedulerHints

	return base, nil
}
