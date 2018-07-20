package cluster

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/docker/distribution/reference"
	apitypes "github.com/docker/docker/api/types"
	"github.com/docker/docker/api/types/backend"
	types "github.com/docker/docker/api/types/swarm"
	timetypes "github.com/docker/docker/api/types/time"
	"github.com/docker/docker/daemon/cluster/convert"
	runconfigopts "github.com/docker/docker/runconfig/opts"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
	"golang.org/x/net/context"
)

// GetServices returns all services of a managed swarm cluster.
func (c *Cluster) GetServices(options apitypes.ServiceListOptions) ([]types.Service, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	if !state.IsActiveManager() {
		return nil, c.errNoManager(state)
	}

	// We move the accepted filter check here as "mode" filter
	// is processed in the daemon, not in SwarmKit. So it might
	// be good to have accepted file check in the same file as
	// the filter processing (in the for loop below).
	accepted := map[string]bool{
		"name":    true,
		"id":      true,
		"label":   true,
		"mode":    true,
		"runtime": true,
	}
	if err := options.Filters.Validate(accepted); err != nil {
		return nil, err
	}

	if len(options.Filters.Get("runtime")) == 0 {
		// Default to using the container runtime filter
		options.Filters.Add("runtime", string(types.RuntimeContainer))
	}

	filters := &swarmapi.ListServicesRequest_Filters{
		NamePrefixes: options.Filters.Get("name"),
		IDPrefixes:   options.Filters.Get("id"),
		Labels:       runconfigopts.ConvertKVStringsToMap(options.Filters.Get("label")),
		Runtimes:     options.Filters.Get("runtime"),
	}

	ctx, cancel := c.getRequestContext()
	defer cancel()

	r, err := state.controlClient.ListServices(
		ctx,
		&swarmapi.ListServicesRequest{Filters: filters})
	if err != nil {
		return nil, err
	}

	services := make([]types.Service, 0, len(r.Services))

	for _, service := range r.Services {
		if options.Filters.Contains("mode") {
			var mode string
			switch service.Spec.GetMode().(type) {
			case *swarmapi.ServiceSpec_Global:
				mode = "global"
			case *swarmapi.ServiceSpec_Replicated:
				mode = "replicated"
			}

			if !options.Filters.ExactMatch("mode", mode) {
				continue
			}
		}
		svcs, err := convert.ServiceFromGRPC(*service)
		if err != nil {
			return nil, err
		}
		services = append(services, svcs)
	}

	return services, nil
}

// GetService returns a service based on an ID or name.
func (c *Cluster) GetService(input string, insertDefaults bool) (types.Service, error) {
	var service *swarmapi.Service
	if err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		s, err := getService(ctx, state.controlClient, input, insertDefaults)
		if err != nil {
			return err
		}
		service = s
		return nil
	}); err != nil {
		return types.Service{}, err
	}
	svc, err := convert.ServiceFromGRPC(*service)
	if err != nil {
		return types.Service{}, err
	}
	return svc, nil
}

// CreateService creates a new service in a managed swarm cluster.
func (c *Cluster) CreateService(s types.ServiceSpec, encodedAuth string, queryRegistry bool) (*apitypes.ServiceCreateResponse, error) {
	var resp *apitypes.ServiceCreateResponse
	err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		err := c.populateNetworkID(ctx, state.controlClient, &s)
		if err != nil {
			return err
		}

		serviceSpec, err := convert.ServiceSpecToGRPC(s)
		if err != nil {
			return convertError{err}
		}

		resp = &apitypes.ServiceCreateResponse{}

		switch serviceSpec.Task.Runtime.(type) {
		// handle other runtimes here
		case *swarmapi.TaskSpec_Generic:
			switch serviceSpec.Task.GetGeneric().Kind {
			case string(types.RuntimePlugin):
				info, _ := c.config.Backend.SystemInfo()
				if !info.ExperimentalBuild {
					return fmt.Errorf("runtime type %q only supported in experimental", types.RuntimePlugin)
				}
				if s.TaskTemplate.PluginSpec == nil {
					return errors.New("plugin spec must be set")
				}

			default:
				return fmt.Errorf("unsupported runtime type: %q", serviceSpec.Task.GetGeneric().Kind)
			}

			r, err := state.controlClient.CreateService(ctx, &swarmapi.CreateServiceRequest{Spec: &serviceSpec})
			if err != nil {
				return err
			}

			resp.ID = r.Service.ID
		case *swarmapi.TaskSpec_Container:
			ctnr := serviceSpec.Task.GetContainer()
			if ctnr == nil {
				return errors.New("service does not use container tasks")
			}
			if encodedAuth != "" {
				ctnr.PullOptions = &swarmapi.ContainerSpec_PullOptions{RegistryAuth: encodedAuth}
			}

			// retrieve auth config from encoded auth
			authConfig := &apitypes.AuthConfig{}
			if encodedAuth != "" {
				authReader := strings.NewReader(encodedAuth)
				dec := json.NewDecoder(base64.NewDecoder(base64.URLEncoding, authReader))
				if err := dec.Decode(authConfig); err != nil {
					logrus.Warnf("invalid authconfig: %v", err)
				}
			}

			// pin image by digest for API versions < 1.30
			// TODO(nishanttotla): The check on "DOCKER_SERVICE_PREFER_OFFLINE_IMAGE"
			// should be removed in the future. Since integration tests only use the
			// latest API version, so this is no longer required.
			if os.Getenv("DOCKER_SERVICE_PREFER_OFFLINE_IMAGE") != "1" && queryRegistry {
				digestImage, err := c.imageWithDigestString(ctx, ctnr.Image, authConfig)
				if err != nil {
					logrus.Warnf("unable to pin image %s to digest: %s", ctnr.Image, err.Error())
					// warning in the client response should be concise
					resp.Warnings = append(resp.Warnings, digestWarning(ctnr.Image))

				} else if ctnr.Image != digestImage {
					logrus.Debugf("pinning image %s by digest: %s", ctnr.Image, digestImage)
					ctnr.Image = digestImage

				} else {
					logrus.Debugf("creating service using supplied digest reference %s", ctnr.Image)

				}

				// Replace the context with a fresh one.
				// If we timed out while communicating with the
				// registry, then "ctx" will already be expired, which
				// would cause UpdateService below to fail. Reusing
				// "ctx" could make it impossible to create a service
				// if the registry is slow or unresponsive.
				var cancel func()
				ctx, cancel = c.getRequestContext()
				defer cancel()
			}

			r, err := state.controlClient.CreateService(ctx, &swarmapi.CreateServiceRequest{Spec: &serviceSpec})
			if err != nil {
				return err
			}

			resp.ID = r.Service.ID
		}
		return nil
	})

	return resp, err
}

// UpdateService updates existing service to match new properties.
func (c *Cluster) UpdateService(serviceIDOrName string, version uint64, spec types.ServiceSpec, flags apitypes.ServiceUpdateOptions, queryRegistry bool) (*apitypes.ServiceUpdateResponse, error) {
	var resp *apitypes.ServiceUpdateResponse

	err := c.lockedManagerAction(func(ctx context.Context, state nodeState) error {

		err := c.populateNetworkID(ctx, state.controlClient, &spec)
		if err != nil {
			return err
		}

		serviceSpec, err := convert.ServiceSpecToGRPC(spec)
		if err != nil {
			return convertError{err}
		}

		currentService, err := getService(ctx, state.controlClient, serviceIDOrName, false)
		if err != nil {
			return err
		}

		resp = &apitypes.ServiceUpdateResponse{}

		switch serviceSpec.Task.Runtime.(type) {
		case *swarmapi.TaskSpec_Generic:
			switch serviceSpec.Task.GetGeneric().Kind {
			case string(types.RuntimePlugin):
				if spec.TaskTemplate.PluginSpec == nil {
					return errors.New("plugin spec must be set")
				}
			}
		case *swarmapi.TaskSpec_Container:
			newCtnr := serviceSpec.Task.GetContainer()
			if newCtnr == nil {
				return errors.New("service does not use container tasks")
			}

			encodedAuth := flags.EncodedRegistryAuth
			if encodedAuth != "" {
				newCtnr.PullOptions = &swarmapi.ContainerSpec_PullOptions{RegistryAuth: encodedAuth}
			} else {
				// this is needed because if the encodedAuth isn't being updated then we
				// shouldn't lose it, and continue to use the one that was already present
				var ctnr *swarmapi.ContainerSpec
				switch flags.RegistryAuthFrom {
				case apitypes.RegistryAuthFromSpec, "":
					ctnr = currentService.Spec.Task.GetContainer()
				case apitypes.RegistryAuthFromPreviousSpec:
					if currentService.PreviousSpec == nil {
						return errors.New("service does not have a previous spec")
					}
					ctnr = currentService.PreviousSpec.Task.GetContainer()
				default:
					return errors.New("unsupported registryAuthFrom value")
				}
				if ctnr == nil {
					return errors.New("service does not use container tasks")
				}
				newCtnr.PullOptions = ctnr.PullOptions
				// update encodedAuth so it can be used to pin image by digest
				if ctnr.PullOptions != nil {
					encodedAuth = ctnr.PullOptions.RegistryAuth
				}
			}

			// retrieve auth config from encoded auth
			authConfig := &apitypes.AuthConfig{}
			if encodedAuth != "" {
				if err := json.NewDecoder(base64.NewDecoder(base64.URLEncoding, strings.NewReader(encodedAuth))).Decode(authConfig); err != nil {
					logrus.Warnf("invalid authconfig: %v", err)
				}
			}

			// pin image by digest for API versions < 1.30
			// TODO(nishanttotla): The check on "DOCKER_SERVICE_PREFER_OFFLINE_IMAGE"
			// should be removed in the future. Since integration tests only use the
			// latest API version, so this is no longer required.
			if os.Getenv("DOCKER_SERVICE_PREFER_OFFLINE_IMAGE") != "1" && queryRegistry {
				digestImage, err := c.imageWithDigestString(ctx, newCtnr.Image, authConfig)
				if err != nil {
					logrus.Warnf("unable to pin image %s to digest: %s", newCtnr.Image, err.Error())
					// warning in the client response should be concise
					resp.Warnings = append(resp.Warnings, digestWarning(newCtnr.Image))
				} else if newCtnr.Image != digestImage {
					logrus.Debugf("pinning image %s by digest: %s", newCtnr.Image, digestImage)
					newCtnr.Image = digestImage
				} else {
					logrus.Debugf("updating service using supplied digest reference %s", newCtnr.Image)
				}

				// Replace the context with a fresh one.
				// If we timed out while communicating with the
				// registry, then "ctx" will already be expired, which
				// would cause UpdateService below to fail. Reusing
				// "ctx" could make it impossible to update a service
				// if the registry is slow or unresponsive.
				var cancel func()
				ctx, cancel = c.getRequestContext()
				defer cancel()
			}
		}

		var rollback swarmapi.UpdateServiceRequest_Rollback
		switch flags.Rollback {
		case "", "none":
			rollback = swarmapi.UpdateServiceRequest_NONE
		case "previous":
			rollback = swarmapi.UpdateServiceRequest_PREVIOUS
		default:
			return fmt.Errorf("unrecognized rollback option %s", flags.Rollback)
		}

		_, err = state.controlClient.UpdateService(
			ctx,
			&swarmapi.UpdateServiceRequest{
				ServiceID: currentService.ID,
				Spec:      &serviceSpec,
				ServiceVersion: &swarmapi.Version{
					Index: version,
				},
				Rollback: rollback,
			},
		)
		return err
	})
	return resp, err
}

// RemoveService removes a service from a managed swarm cluster.
func (c *Cluster) RemoveService(input string) error {
	return c.lockedManagerAction(func(ctx context.Context, state nodeState) error {
		service, err := getService(ctx, state.controlClient, input, false)
		if err != nil {
			return err
		}

		_, err = state.controlClient.RemoveService(ctx, &swarmapi.RemoveServiceRequest{ServiceID: service.ID})
		return err
	})
}

// ServiceLogs collects service logs and writes them back to `config.OutStream`
func (c *Cluster) ServiceLogs(ctx context.Context, selector *backend.LogSelector, config *apitypes.ContainerLogsOptions) (<-chan *backend.LogMessage, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()

	state := c.currentNodeState()
	if !state.IsActiveManager() {
		return nil, c.errNoManager(state)
	}

	swarmSelector, err := convertSelector(ctx, state.controlClient, selector)
	if err != nil {
		return nil, errors.Wrap(err, "error making log selector")
	}

	// set the streams we'll use
	stdStreams := []swarmapi.LogStream{}
	if config.ShowStdout {
		stdStreams = append(stdStreams, swarmapi.LogStreamStdout)
	}
	if config.ShowStderr {
		stdStreams = append(stdStreams, swarmapi.LogStreamStderr)
	}

	// Get tail value squared away - the number of previous log lines we look at
	var tail int64
	// in ContainerLogs, if the tail value is ANYTHING non-integer, we just set
	// it to -1 (all). i don't agree with that, but i also think no tail value
	// should be legitimate. if you don't pass tail, we assume you want "all"
	if config.Tail == "all" || config.Tail == "" {
		// tail of 0 means send all logs on the swarmkit side
		tail = 0
	} else {
		t, err := strconv.Atoi(config.Tail)
		if err != nil {
			return nil, errors.New("tail value must be a positive integer or \"all\"")
		}
		if t < 0 {
			return nil, errors.New("negative tail values not supported")
		}
		// we actually use negative tail in swarmkit to represent messages
		// backwards starting from the beginning. also, -1 means no logs. so,
		// basically, for api compat with docker container logs, add one and
		// flip the sign. we error above if you try to negative tail, which
		// isn't supported by docker (and would error deeper in the stack
		// anyway)
		//
		// See the logs protobuf for more information
		tail = int64(-(t + 1))
	}

	// get the since value - the time in the past we're looking at logs starting from
	var sinceProto *gogotypes.Timestamp
	if config.Since != "" {
		s, n, err := timetypes.ParseTimestamps(config.Since, 0)
		if err != nil {
			return nil, errors.Wrap(err, "could not parse since timestamp")
		}
		since := time.Unix(s, n)
		sinceProto, err = gogotypes.TimestampProto(since)
		if err != nil {
			return nil, errors.Wrap(err, "could not parse timestamp to proto")
		}
	}

	stream, err := state.logsClient.SubscribeLogs(ctx, &swarmapi.SubscribeLogsRequest{
		Selector: swarmSelector,
		Options: &swarmapi.LogSubscriptionOptions{
			Follow:  config.Follow,
			Streams: stdStreams,
			Tail:    tail,
			Since:   sinceProto,
		},
	})
	if err != nil {
		return nil, err
	}

	messageChan := make(chan *backend.LogMessage, 1)
	go func() {
		defer close(messageChan)
		for {
			// Check the context before doing anything.
			select {
			case <-ctx.Done():
				return
			default:
			}
			subscribeMsg, err := stream.Recv()
			if err == io.EOF {
				return
			}
			// if we're not io.EOF, push the message in and return
			if err != nil {
				select {
				case <-ctx.Done():
				case messageChan <- &backend.LogMessage{Err: err}:
				}
				return
			}

			for _, msg := range subscribeMsg.Messages {
				// make a new message
				m := new(backend.LogMessage)
				m.Attrs = make([]backend.LogAttr, 0, len(msg.Attrs)+3)
				// add the timestamp, adding the error if it fails
				m.Timestamp, err = gogotypes.TimestampFromProto(msg.Timestamp)
				if err != nil {
					m.Err = err
				}

				nodeKey := contextPrefix + ".node.id"
				serviceKey := contextPrefix + ".service.id"
				taskKey := contextPrefix + ".task.id"

				// copy over all of the details
				for _, d := range msg.Attrs {
					switch d.Key {
					case nodeKey, serviceKey, taskKey:
						// we have the final say over context details (in case there
						// is a conflict (if the user added a detail with a context's
						// key for some reason))
					default:
						m.Attrs = append(m.Attrs, backend.LogAttr{Key: d.Key, Value: d.Value})
					}
				}
				m.Attrs = append(m.Attrs,
					backend.LogAttr{Key: nodeKey, Value: msg.Context.NodeID},
					backend.LogAttr{Key: serviceKey, Value: msg.Context.ServiceID},
					backend.LogAttr{Key: taskKey, Value: msg.Context.TaskID},
				)

				switch msg.Stream {
				case swarmapi.LogStreamStdout:
					m.Source = "stdout"
				case swarmapi.LogStreamStderr:
					m.Source = "stderr"
				}
				m.Line = msg.Data

				// there could be a case where the reader stops accepting
				// messages and the context is canceled. we need to check that
				// here, or otherwise we risk blocking forever on the message
				// send.
				select {
				case <-ctx.Done():
					return
				case messageChan <- m:
				}
			}
		}
	}()
	return messageChan, nil
}

// convertSelector takes a backend.LogSelector, which contains raw names that
// may or may not be valid, and converts them to an api.LogSelector proto. It
// returns an error if something fails
func convertSelector(ctx context.Context, cc swarmapi.ControlClient, selector *backend.LogSelector) (*swarmapi.LogSelector, error) {
	// don't rely on swarmkit to resolve IDs, do it ourselves
	swarmSelector := &swarmapi.LogSelector{}
	for _, s := range selector.Services {
		service, err := getService(ctx, cc, s, false)
		if err != nil {
			return nil, err
		}
		c := service.Spec.Task.GetContainer()
		if c == nil {
			return nil, errors.New("logs only supported on container tasks")
		}
		swarmSelector.ServiceIDs = append(swarmSelector.ServiceIDs, service.ID)
	}
	for _, t := range selector.Tasks {
		task, err := getTask(ctx, cc, t)
		if err != nil {
			return nil, err
		}
		c := task.Spec.GetContainer()
		if c == nil {
			return nil, errors.New("logs only supported on container tasks")
		}
		swarmSelector.TaskIDs = append(swarmSelector.TaskIDs, task.ID)
	}
	return swarmSelector, nil
}

// imageWithDigestString takes an image such as name or name:tag
// and returns the image pinned to a digest, such as name@sha256:34234
func (c *Cluster) imageWithDigestString(ctx context.Context, image string, authConfig *apitypes.AuthConfig) (string, error) {
	ref, err := reference.ParseAnyReference(image)
	if err != nil {
		return "", err
	}
	namedRef, ok := ref.(reference.Named)
	if !ok {
		if _, ok := ref.(reference.Digested); ok {
			return image, nil
		}
		return "", errors.Errorf("unknown image reference format: %s", image)
	}
	// only query registry if not a canonical reference (i.e. with digest)
	if _, ok := namedRef.(reference.Canonical); !ok {
		namedRef = reference.TagNameOnly(namedRef)

		taggedRef, ok := namedRef.(reference.NamedTagged)
		if !ok {
			return "", errors.Errorf("image reference not tagged: %s", image)
		}

		repo, _, err := c.config.Backend.GetRepository(ctx, taggedRef, authConfig)
		if err != nil {
			return "", err
		}
		dscrptr, err := repo.Tags(ctx).Get(ctx, taggedRef.Tag())
		if err != nil {
			return "", err
		}

		namedDigestedRef, err := reference.WithDigest(taggedRef, dscrptr.Digest)
		if err != nil {
			return "", err
		}
		// return familiar form until interface updated to return type
		return reference.FamiliarString(namedDigestedRef), nil
	}
	// reference already contains a digest, so just return it
	return reference.FamiliarString(ref), nil
}

// digestWarning constructs a formatted warning string
// using the image name that could not be pinned by digest. The
// formatting is hardcoded, but could me made smarter in the future
func digestWarning(image string) string {
	return fmt.Sprintf("image %s could not be accessed on a registry to record\nits digest. Each node will access %s independently,\npossibly leading to different nodes running different\nversions of the image.\n", image, image)
}
