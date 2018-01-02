package daemon

import (
	"context"
	"strconv"
	"strings"
	"time"

	"github.com/docker/docker/api/types/events"
	"github.com/docker/docker/api/types/filters"
	"github.com/docker/docker/container"
	daemonevents "github.com/docker/docker/daemon/events"
	"github.com/docker/libnetwork"
	swarmapi "github.com/docker/swarmkit/api"
	gogotypes "github.com/gogo/protobuf/types"
	"github.com/sirupsen/logrus"
)

var (
	clusterEventAction = map[swarmapi.WatchActionKind]string{
		swarmapi.WatchActionKindCreate: "create",
		swarmapi.WatchActionKindUpdate: "update",
		swarmapi.WatchActionKindRemove: "remove",
	}
)

// LogContainerEvent generates an event related to a container with only the default attributes.
func (daemon *Daemon) LogContainerEvent(container *container.Container, action string) {
	daemon.LogContainerEventWithAttributes(container, action, map[string]string{})
}

// LogContainerEventWithAttributes generates an event related to a container with specific given attributes.
func (daemon *Daemon) LogContainerEventWithAttributes(container *container.Container, action string, attributes map[string]string) {
	copyAttributes(attributes, container.Config.Labels)
	if container.Config.Image != "" {
		attributes["image"] = container.Config.Image
	}
	attributes["name"] = strings.TrimLeft(container.Name, "/")

	actor := events.Actor{
		ID:         container.ID,
		Attributes: attributes,
	}
	daemon.EventsService.Log(action, events.ContainerEventType, actor)
}

// LogImageEvent generates an event related to an image with only the default attributes.
func (daemon *Daemon) LogImageEvent(imageID, refName, action string) {
	daemon.LogImageEventWithAttributes(imageID, refName, action, map[string]string{})
}

// LogImageEventWithAttributes generates an event related to an image with specific given attributes.
func (daemon *Daemon) LogImageEventWithAttributes(imageID, refName, action string, attributes map[string]string) {
	img, err := daemon.GetImage(imageID)
	if err == nil && img.Config != nil {
		// image has not been removed yet.
		// it could be missing if the event is `delete`.
		copyAttributes(attributes, img.Config.Labels)
	}
	if refName != "" {
		attributes["name"] = refName
	}
	actor := events.Actor{
		ID:         imageID,
		Attributes: attributes,
	}

	daemon.EventsService.Log(action, events.ImageEventType, actor)
}

// LogPluginEvent generates an event related to a plugin with only the default attributes.
func (daemon *Daemon) LogPluginEvent(pluginID, refName, action string) {
	daemon.LogPluginEventWithAttributes(pluginID, refName, action, map[string]string{})
}

// LogPluginEventWithAttributes generates an event related to a plugin with specific given attributes.
func (daemon *Daemon) LogPluginEventWithAttributes(pluginID, refName, action string, attributes map[string]string) {
	attributes["name"] = refName
	actor := events.Actor{
		ID:         pluginID,
		Attributes: attributes,
	}
	daemon.EventsService.Log(action, events.PluginEventType, actor)
}

// LogVolumeEvent generates an event related to a volume.
func (daemon *Daemon) LogVolumeEvent(volumeID, action string, attributes map[string]string) {
	actor := events.Actor{
		ID:         volumeID,
		Attributes: attributes,
	}
	daemon.EventsService.Log(action, events.VolumeEventType, actor)
}

// LogNetworkEvent generates an event related to a network with only the default attributes.
func (daemon *Daemon) LogNetworkEvent(nw libnetwork.Network, action string) {
	daemon.LogNetworkEventWithAttributes(nw, action, map[string]string{})
}

// LogNetworkEventWithAttributes generates an event related to a network with specific given attributes.
func (daemon *Daemon) LogNetworkEventWithAttributes(nw libnetwork.Network, action string, attributes map[string]string) {
	attributes["name"] = nw.Name()
	attributes["type"] = nw.Type()
	actor := events.Actor{
		ID:         nw.ID(),
		Attributes: attributes,
	}
	daemon.EventsService.Log(action, events.NetworkEventType, actor)
}

// LogDaemonEventWithAttributes generates an event related to the daemon itself with specific given attributes.
func (daemon *Daemon) LogDaemonEventWithAttributes(action string, attributes map[string]string) {
	if daemon.EventsService != nil {
		if info, err := daemon.SystemInfo(); err == nil && info.Name != "" {
			attributes["name"] = info.Name
		}
		actor := events.Actor{
			ID:         daemon.ID,
			Attributes: attributes,
		}
		daemon.EventsService.Log(action, events.DaemonEventType, actor)
	}
}

// SubscribeToEvents returns the currently record of events, a channel to stream new events from, and a function to cancel the stream of events.
func (daemon *Daemon) SubscribeToEvents(since, until time.Time, filter filters.Args) ([]events.Message, chan interface{}) {
	ef := daemonevents.NewFilter(filter)
	return daemon.EventsService.SubscribeTopic(since, until, ef)
}

// UnsubscribeFromEvents stops the event subscription for a client by closing the
// channel where the daemon sends events to.
func (daemon *Daemon) UnsubscribeFromEvents(listener chan interface{}) {
	daemon.EventsService.Evict(listener)
}

// copyAttributes guarantees that labels are not mutated by event triggers.
func copyAttributes(attributes, labels map[string]string) {
	if labels == nil {
		return
	}
	for k, v := range labels {
		attributes[k] = v
	}
}

// ProcessClusterNotifications gets changes from store and add them to event list
func (daemon *Daemon) ProcessClusterNotifications(ctx context.Context, watchStream chan *swarmapi.WatchMessage) {
	for {
		select {
		case <-ctx.Done():
			return
		case message, ok := <-watchStream:
			if !ok {
				logrus.Debug("cluster event channel has stopped")
				return
			}
			daemon.generateClusterEvent(message)
		}
	}
}

func (daemon *Daemon) generateClusterEvent(msg *swarmapi.WatchMessage) {
	for _, event := range msg.Events {
		if event.Object == nil {
			logrus.Errorf("event without object: %v", event)
			continue
		}
		switch v := event.Object.GetObject().(type) {
		case *swarmapi.Object_Node:
			daemon.logNodeEvent(event.Action, v.Node, event.OldObject.GetNode())
		case *swarmapi.Object_Service:
			daemon.logServiceEvent(event.Action, v.Service, event.OldObject.GetService())
		case *swarmapi.Object_Network:
			daemon.logNetworkEvent(event.Action, v.Network, event.OldObject.GetNetwork())
		case *swarmapi.Object_Secret:
			daemon.logSecretEvent(event.Action, v.Secret, event.OldObject.GetSecret())
		case *swarmapi.Object_Config:
			daemon.logConfigEvent(event.Action, v.Config, event.OldObject.GetConfig())
		default:
			logrus.Warnf("unrecognized event: %v", event)
		}
	}
}

func (daemon *Daemon) logNetworkEvent(action swarmapi.WatchActionKind, net *swarmapi.Network, oldNet *swarmapi.Network) {
	attributes := map[string]string{
		"name": net.Spec.Annotations.Name,
	}
	eventTime := eventTimestamp(net.Meta, action)
	daemon.logClusterEvent(action, net.ID, "network", attributes, eventTime)
}

func (daemon *Daemon) logSecretEvent(action swarmapi.WatchActionKind, secret *swarmapi.Secret, oldSecret *swarmapi.Secret) {
	attributes := map[string]string{
		"name": secret.Spec.Annotations.Name,
	}
	eventTime := eventTimestamp(secret.Meta, action)
	daemon.logClusterEvent(action, secret.ID, "secret", attributes, eventTime)
}

func (daemon *Daemon) logConfigEvent(action swarmapi.WatchActionKind, config *swarmapi.Config, oldConfig *swarmapi.Config) {
	attributes := map[string]string{
		"name": config.Spec.Annotations.Name,
	}
	eventTime := eventTimestamp(config.Meta, action)
	daemon.logClusterEvent(action, config.ID, "config", attributes, eventTime)
}

func (daemon *Daemon) logNodeEvent(action swarmapi.WatchActionKind, node *swarmapi.Node, oldNode *swarmapi.Node) {
	name := node.Spec.Annotations.Name
	if name == "" && node.Description != nil {
		name = node.Description.Hostname
	}
	attributes := map[string]string{
		"name": name,
	}
	eventTime := eventTimestamp(node.Meta, action)
	// In an update event, display the changes in attributes
	if action == swarmapi.WatchActionKindUpdate && oldNode != nil {
		if node.Spec.Availability != oldNode.Spec.Availability {
			attributes["availability.old"] = strings.ToLower(oldNode.Spec.Availability.String())
			attributes["availability.new"] = strings.ToLower(node.Spec.Availability.String())
		}
		if node.Role != oldNode.Role {
			attributes["role.old"] = strings.ToLower(oldNode.Role.String())
			attributes["role.new"] = strings.ToLower(node.Role.String())
		}
		if node.Status.State != oldNode.Status.State {
			attributes["state.old"] = strings.ToLower(oldNode.Status.State.String())
			attributes["state.new"] = strings.ToLower(node.Status.State.String())
		}
		// This handles change within manager role
		if node.ManagerStatus != nil && oldNode.ManagerStatus != nil {
			// leader change
			if node.ManagerStatus.Leader != oldNode.ManagerStatus.Leader {
				if node.ManagerStatus.Leader {
					attributes["leader.old"] = "false"
					attributes["leader.new"] = "true"
				} else {
					attributes["leader.old"] = "true"
					attributes["leader.new"] = "false"
				}
			}
			if node.ManagerStatus.Reachability != oldNode.ManagerStatus.Reachability {
				attributes["reachability.old"] = strings.ToLower(oldNode.ManagerStatus.Reachability.String())
				attributes["reachability.new"] = strings.ToLower(node.ManagerStatus.Reachability.String())
			}
		}
	}

	daemon.logClusterEvent(action, node.ID, "node", attributes, eventTime)
}

func (daemon *Daemon) logServiceEvent(action swarmapi.WatchActionKind, service *swarmapi.Service, oldService *swarmapi.Service) {
	attributes := map[string]string{
		"name": service.Spec.Annotations.Name,
	}
	eventTime := eventTimestamp(service.Meta, action)

	if action == swarmapi.WatchActionKindUpdate && oldService != nil {
		// check image
		if x, ok := service.Spec.Task.GetRuntime().(*swarmapi.TaskSpec_Container); ok {
			containerSpec := x.Container
			if y, ok := oldService.Spec.Task.GetRuntime().(*swarmapi.TaskSpec_Container); ok {
				oldContainerSpec := y.Container
				if containerSpec.Image != oldContainerSpec.Image {
					attributes["image.old"] = oldContainerSpec.Image
					attributes["image.new"] = containerSpec.Image
				}
			} else {
				// This should not happen.
				logrus.Errorf("service %s runtime changed from %T to %T", service.Spec.Annotations.Name, oldService.Spec.Task.GetRuntime(), service.Spec.Task.GetRuntime())
			}
		}
		// check replicated count change
		if x, ok := service.Spec.GetMode().(*swarmapi.ServiceSpec_Replicated); ok {
			replicas := x.Replicated.Replicas
			if y, ok := oldService.Spec.GetMode().(*swarmapi.ServiceSpec_Replicated); ok {
				oldReplicas := y.Replicated.Replicas
				if replicas != oldReplicas {
					attributes["replicas.old"] = strconv.FormatUint(oldReplicas, 10)
					attributes["replicas.new"] = strconv.FormatUint(replicas, 10)
				}
			} else {
				// This should not happen.
				logrus.Errorf("service %s mode changed from %T to %T", service.Spec.Annotations.Name, oldService.Spec.GetMode(), service.Spec.GetMode())
			}
		}
		if service.UpdateStatus != nil {
			if oldService.UpdateStatus == nil {
				attributes["updatestate.new"] = strings.ToLower(service.UpdateStatus.State.String())
			} else if service.UpdateStatus.State != oldService.UpdateStatus.State {
				attributes["updatestate.old"] = strings.ToLower(oldService.UpdateStatus.State.String())
				attributes["updatestate.new"] = strings.ToLower(service.UpdateStatus.State.String())
			}
		}
	}
	daemon.logClusterEvent(action, service.ID, "service", attributes, eventTime)
}

func (daemon *Daemon) logClusterEvent(action swarmapi.WatchActionKind, id, eventType string, attributes map[string]string, eventTime time.Time) {
	actor := events.Actor{
		ID:         id,
		Attributes: attributes,
	}

	jm := events.Message{
		Action:   clusterEventAction[action],
		Type:     eventType,
		Actor:    actor,
		Scope:    "swarm",
		Time:     eventTime.UTC().Unix(),
		TimeNano: eventTime.UTC().UnixNano(),
	}
	daemon.EventsService.PublishMessage(jm)
}

func eventTimestamp(meta swarmapi.Meta, action swarmapi.WatchActionKind) time.Time {
	var eventTime time.Time
	switch action {
	case swarmapi.WatchActionKindCreate:
		eventTime, _ = gogotypes.TimestampFromProto(meta.CreatedAt)
	case swarmapi.WatchActionKindUpdate:
		eventTime, _ = gogotypes.TimestampFromProto(meta.UpdatedAt)
	case swarmapi.WatchActionKindRemove:
		// There is no timestamp from store message for remove operations.
		// Use current time.
		eventTime = time.Now()
	}
	return eventTime
}
