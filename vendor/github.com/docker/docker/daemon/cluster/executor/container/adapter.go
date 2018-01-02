package container

import (
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"runtime"
	"strings"
	"syscall"
	"time"

	"github.com/docker/distribution/reference"
	"github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/api/types/events"
	containerpkg "github.com/docker/docker/container"
	"github.com/docker/docker/daemon/cluster/convert"
	executorpkg "github.com/docker/docker/daemon/cluster/executor"
	"github.com/docker/docker/pkg/system"
	"github.com/docker/libnetwork"
	"github.com/docker/swarmkit/agent/exec"
	"github.com/docker/swarmkit/api"
	"github.com/docker/swarmkit/log"
	gogotypes "github.com/gogo/protobuf/types"
	"github.com/opencontainers/go-digest"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
	"golang.org/x/time/rate"
)

// containerAdapter conducts remote operations for a container. All calls
// are mostly naked calls to the client API, seeded with information from
// containerConfig.
type containerAdapter struct {
	backend      executorpkg.Backend
	container    *containerConfig
	dependencies exec.DependencyGetter
}

func newContainerAdapter(b executorpkg.Backend, task *api.Task, dependencies exec.DependencyGetter) (*containerAdapter, error) {
	ctnr, err := newContainerConfig(task)
	if err != nil {
		return nil, err
	}

	return &containerAdapter{
		container:    ctnr,
		backend:      b,
		dependencies: dependencies,
	}, nil
}

func (c *containerAdapter) pullImage(ctx context.Context) error {
	spec := c.container.spec()

	// Skip pulling if the image is referenced by image ID.
	if _, err := digest.Parse(spec.Image); err == nil {
		return nil
	}

	// Skip pulling if the image is referenced by digest and already
	// exists locally.
	named, err := reference.ParseNormalizedNamed(spec.Image)
	if err == nil {
		if _, ok := named.(reference.Canonical); ok {
			_, err := c.backend.LookupImage(spec.Image)
			if err == nil {
				return nil
			}
		}
	}

	// if the image needs to be pulled, the auth config will be retrieved and updated
	var encodedAuthConfig string
	if spec.PullOptions != nil {
		encodedAuthConfig = spec.PullOptions.RegistryAuth
	}

	authConfig := &types.AuthConfig{}
	if encodedAuthConfig != "" {
		if err := json.NewDecoder(base64.NewDecoder(base64.URLEncoding, strings.NewReader(encodedAuthConfig))).Decode(authConfig); err != nil {
			logrus.Warnf("invalid authconfig: %v", err)
		}
	}

	pr, pw := io.Pipe()
	metaHeaders := map[string][]string{}
	go func() {
		// TODO @jhowardmsft LCOW Support: This will need revisiting as
		// the stack is built up to include LCOW support for swarm.
		platform := runtime.GOOS
		if system.LCOWSupported() {
			platform = "linux"
		}
		err := c.backend.PullImage(ctx, c.container.image(), "", platform, metaHeaders, authConfig, pw)
		pw.CloseWithError(err)
	}()

	dec := json.NewDecoder(pr)
	dec.UseNumber()
	m := map[string]interface{}{}
	spamLimiter := rate.NewLimiter(rate.Every(time.Second), 1)

	lastStatus := ""
	for {
		if err := dec.Decode(&m); err != nil {
			if err == io.EOF {
				break
			}
			return err
		}
		l := log.G(ctx)
		// limit pull progress logs unless the status changes
		if spamLimiter.Allow() || lastStatus != m["status"] {
			// if we have progress details, we have everything we need
			if progress, ok := m["progressDetail"].(map[string]interface{}); ok {
				// first, log the image and status
				l = l.WithFields(logrus.Fields{
					"image":  c.container.image(),
					"status": m["status"],
				})
				// then, if we have progress, log the progress
				if progress["current"] != nil && progress["total"] != nil {
					l = l.WithFields(logrus.Fields{
						"current": progress["current"],
						"total":   progress["total"],
					})
				}
			}
			l.Debug("pull in progress")
		}
		// sometimes, we get no useful information at all, and add no fields
		if status, ok := m["status"].(string); ok {
			lastStatus = status
		}
	}

	// if the final stream object contained an error, return it
	if errMsg, ok := m["error"]; ok {
		return fmt.Errorf("%v", errMsg)
	}
	return nil
}

func (c *containerAdapter) createNetworks(ctx context.Context) error {
	for _, network := range c.container.networks() {
		ncr, err := c.container.networkCreateRequest(network)
		if err != nil {
			return err
		}

		if err := c.backend.CreateManagedNetwork(ncr); err != nil { // todo name missing
			if _, ok := err.(libnetwork.NetworkNameError); ok {
				continue
			}

			return err
		}
	}

	return nil
}

func (c *containerAdapter) removeNetworks(ctx context.Context) error {
	for _, nid := range c.container.networks() {
		if err := c.backend.DeleteManagedNetwork(nid); err != nil {
			switch err.(type) {
			case *libnetwork.ActiveEndpointsError:
				continue
			case libnetwork.ErrNoSuchNetwork:
				continue
			default:
				log.G(ctx).Errorf("network %s remove failed: %v", nid, err)
				return err
			}
		}
	}

	return nil
}

func (c *containerAdapter) networkAttach(ctx context.Context) error {
	config := c.container.createNetworkingConfig(c.backend)

	var (
		networkName string
		networkID   string
	)

	if config != nil {
		for n, epConfig := range config.EndpointsConfig {
			networkName = n
			networkID = epConfig.NetworkID
			break
		}
	}

	return c.backend.UpdateAttachment(networkName, networkID, c.container.networkAttachmentContainerID(), config)
}

func (c *containerAdapter) waitForDetach(ctx context.Context) error {
	config := c.container.createNetworkingConfig(c.backend)

	var (
		networkName string
		networkID   string
	)

	if config != nil {
		for n, epConfig := range config.EndpointsConfig {
			networkName = n
			networkID = epConfig.NetworkID
			break
		}
	}

	return c.backend.WaitForDetachment(ctx, networkName, networkID, c.container.taskID(), c.container.networkAttachmentContainerID())
}

func (c *containerAdapter) create(ctx context.Context) error {
	var cr containertypes.ContainerCreateCreatedBody
	var err error
	if cr, err = c.backend.CreateManagedContainer(types.ContainerCreateConfig{
		Name:       c.container.name(),
		Config:     c.container.config(),
		HostConfig: c.container.hostConfig(),
		// Use the first network in container create
		NetworkingConfig: c.container.createNetworkingConfig(c.backend),
	}); err != nil {
		return err
	}

	// Docker daemon currently doesn't support multiple networks in container create
	// Connect to all other networks
	nc := c.container.connectNetworkingConfig(c.backend)

	if nc != nil {
		for n, ep := range nc.EndpointsConfig {
			if err := c.backend.ConnectContainerToNetwork(cr.ID, n, ep); err != nil {
				return err
			}
		}
	}

	container := c.container.task.Spec.GetContainer()
	if container == nil {
		return errors.New("unable to get container from task spec")
	}

	if err := c.backend.SetContainerDependencyStore(cr.ID, c.dependencies); err != nil {
		return err
	}

	// configure secrets
	secretRefs := convert.SecretReferencesFromGRPC(container.Secrets)
	if err := c.backend.SetContainerSecretReferences(cr.ID, secretRefs); err != nil {
		return err
	}

	configRefs := convert.ConfigReferencesFromGRPC(container.Configs)
	if err := c.backend.SetContainerConfigReferences(cr.ID, configRefs); err != nil {
		return err
	}

	if err := c.backend.UpdateContainerServiceConfig(cr.ID, c.container.serviceConfig()); err != nil {
		return err
	}

	return nil
}

// checkMounts ensures that the provided mounts won't have any host-specific
// problems at start up. For example, we disallow bind mounts without an
// existing path, which slightly different from the container API.
func (c *containerAdapter) checkMounts() error {
	spec := c.container.spec()
	for _, mount := range spec.Mounts {
		switch mount.Type {
		case api.MountTypeBind:
			if _, err := os.Stat(mount.Source); os.IsNotExist(err) {
				return fmt.Errorf("invalid bind mount source, source path not found: %s", mount.Source)
			}
		}
	}

	return nil
}

func (c *containerAdapter) start(ctx context.Context) error {
	if err := c.checkMounts(); err != nil {
		return err
	}

	return c.backend.ContainerStart(c.container.name(), nil, "", "")
}

func (c *containerAdapter) inspect(ctx context.Context) (types.ContainerJSON, error) {
	cs, err := c.backend.ContainerInspectCurrent(c.container.name(), false)
	if ctx.Err() != nil {
		return types.ContainerJSON{}, ctx.Err()
	}
	if err != nil {
		return types.ContainerJSON{}, err
	}
	return *cs, nil
}

// events issues a call to the events API and returns a channel with all
// events. The stream of events can be shutdown by cancelling the context.
func (c *containerAdapter) events(ctx context.Context) <-chan events.Message {
	log.G(ctx).Debugf("waiting on events")
	buffer, l := c.backend.SubscribeToEvents(time.Time{}, time.Time{}, c.container.eventFilter())
	eventsq := make(chan events.Message, len(buffer))

	for _, event := range buffer {
		eventsq <- event
	}

	go func() {
		defer c.backend.UnsubscribeFromEvents(l)

		for {
			select {
			case ev := <-l:
				jev, ok := ev.(events.Message)
				if !ok {
					log.G(ctx).Warnf("unexpected event message: %q", ev)
					continue
				}
				select {
				case eventsq <- jev:
				case <-ctx.Done():
					return
				}
			case <-ctx.Done():
				return
			}
		}
	}()

	return eventsq
}

func (c *containerAdapter) wait(ctx context.Context) (<-chan containerpkg.StateStatus, error) {
	return c.backend.ContainerWait(ctx, c.container.nameOrID(), containerpkg.WaitConditionNotRunning)
}

func (c *containerAdapter) shutdown(ctx context.Context) error {
	// Default stop grace period to nil (daemon will use the stopTimeout of the container)
	var stopgrace *int
	spec := c.container.spec()
	if spec.StopGracePeriod != nil {
		stopgraceValue := int(spec.StopGracePeriod.Seconds)
		stopgrace = &stopgraceValue
	}
	return c.backend.ContainerStop(c.container.name(), stopgrace)
}

func (c *containerAdapter) terminate(ctx context.Context) error {
	return c.backend.ContainerKill(c.container.name(), uint64(syscall.SIGKILL))
}

func (c *containerAdapter) remove(ctx context.Context) error {
	return c.backend.ContainerRm(c.container.name(), &types.ContainerRmConfig{
		RemoveVolume: true,
		ForceRemove:  true,
	})
}

func (c *containerAdapter) createVolumes(ctx context.Context) error {
	// Create plugin volumes that are embedded inside a Mount
	for _, mount := range c.container.task.Spec.GetContainer().Mounts {
		if mount.Type != api.MountTypeVolume {
			continue
		}

		if mount.VolumeOptions == nil {
			continue
		}

		if mount.VolumeOptions.DriverConfig == nil {
			continue
		}

		req := c.container.volumeCreateRequest(&mount)

		// Check if this volume exists on the engine
		if _, err := c.backend.VolumeCreate(req.Name, req.Driver, req.DriverOpts, req.Labels); err != nil {
			// TODO(amitshukla): Today, volume create through the engine api does not return an error
			// when the named volume with the same parameters already exists.
			// It returns an error if the driver name is different - that is a valid error
			return err
		}

	}

	return nil
}

func (c *containerAdapter) activateServiceBinding() error {
	return c.backend.ActivateContainerServiceBinding(c.container.name())
}

func (c *containerAdapter) deactivateServiceBinding() error {
	return c.backend.DeactivateContainerServiceBinding(c.container.name())
}

func (c *containerAdapter) logs(ctx context.Context, options api.LogSubscriptionOptions) (<-chan *backend.LogMessage, error) {
	apiOptions := &types.ContainerLogsOptions{
		Follow: options.Follow,

		// Always say yes to Timestamps and Details. we make the decision
		// of whether to return these to the user or not way higher up the
		// stack.
		Timestamps: true,
		Details:    true,
	}

	if options.Since != nil {
		since, err := gogotypes.TimestampFromProto(options.Since)
		if err != nil {
			return nil, err
		}
		// print since as this formatted string because the docker container
		// logs interface expects it like this.
		// see github.com/docker/docker/api/types/time.ParseTimestamps
		apiOptions.Since = fmt.Sprintf("%d.%09d", since.Unix(), int64(since.Nanosecond()))
	}

	if options.Tail < 0 {
		// See protobuf documentation for details of how this works.
		apiOptions.Tail = fmt.Sprint(-options.Tail - 1)
	} else if options.Tail > 0 {
		return nil, errors.New("tail relative to start of logs not supported via docker API")
	}

	if len(options.Streams) == 0 {
		// empty == all
		apiOptions.ShowStdout, apiOptions.ShowStderr = true, true
	} else {
		for _, stream := range options.Streams {
			switch stream {
			case api.LogStreamStdout:
				apiOptions.ShowStdout = true
			case api.LogStreamStderr:
				apiOptions.ShowStderr = true
			}
		}
	}
	msgs, err := c.backend.ContainerLogs(ctx, c.container.name(), apiOptions)
	if err != nil {
		return nil, err
	}
	return msgs, nil
}

// todo: typed/wrapped errors
func isContainerCreateNameConflict(err error) bool {
	return strings.Contains(err.Error(), "Conflict. The name")
}

func isUnknownContainer(err error) bool {
	return strings.Contains(err.Error(), "No such container:")
}

func isStoppedContainer(err error) bool {
	return strings.Contains(err.Error(), "is already stopped")
}
