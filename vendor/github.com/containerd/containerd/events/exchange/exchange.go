package exchange

import (
	"context"
	"strings"
	"time"

	v1 "github.com/containerd/containerd/api/services/events/v1"
	"github.com/containerd/containerd/errdefs"
	"github.com/containerd/containerd/events"
	"github.com/containerd/containerd/filters"
	"github.com/containerd/containerd/identifiers"
	"github.com/containerd/containerd/log"
	"github.com/containerd/containerd/namespaces"
	"github.com/containerd/typeurl"
	goevents "github.com/docker/go-events"
	"github.com/gogo/protobuf/types"
	"github.com/pkg/errors"
	"github.com/sirupsen/logrus"
)

// Exchange broadcasts events
type Exchange struct {
	broadcaster *goevents.Broadcaster
}

// NewExchange returns a new event Exchange
func NewExchange() *Exchange {
	return &Exchange{
		broadcaster: goevents.NewBroadcaster(),
	}
}

// Forward accepts an envelope to be direcly distributed on the exchange.
//
// This is useful when an event is forwaded on behalf of another namespace or
// when the event is propagated on behalf of another publisher.
func (e *Exchange) Forward(ctx context.Context, envelope *v1.Envelope) (err error) {
	if err := validateEnvelope(envelope); err != nil {
		return err
	}

	defer func() {
		logger := log.G(ctx).WithFields(logrus.Fields{
			"topic": envelope.Topic,
			"ns":    envelope.Namespace,
			"type":  envelope.Event.TypeUrl,
		})

		if err != nil {
			logger.WithError(err).Error("error forwarding event")
		} else {
			logger.Debug("event forwarded")
		}
	}()

	return e.broadcaster.Write(envelope)
}

// Publish packages and sends an event. The caller will be considered the
// initial publisher of the event. This means the timestamp will be calculated
// at this point and this method may read from the calling context.
func (e *Exchange) Publish(ctx context.Context, topic string, event events.Event) (err error) {
	var (
		namespace string
		encoded   *types.Any
		envelope  v1.Envelope
	)

	namespace, err = namespaces.NamespaceRequired(ctx)
	if err != nil {
		return errors.Wrapf(err, "failed publishing event")
	}
	if err := validateTopic(topic); err != nil {
		return errors.Wrapf(err, "envelope topic %q", topic)
	}

	encoded, err = typeurl.MarshalAny(event)
	if err != nil {
		return err
	}

	envelope.Timestamp = time.Now().UTC()
	envelope.Namespace = namespace
	envelope.Topic = topic
	envelope.Event = encoded

	defer func() {
		logger := log.G(ctx).WithFields(logrus.Fields{
			"topic": envelope.Topic,
			"ns":    envelope.Namespace,
			"type":  envelope.Event.TypeUrl,
		})

		if err != nil {
			logger.WithError(err).Error("error publishing event")
		} else {
			logger.Debug("event published")
		}
	}()

	return e.broadcaster.Write(&envelope)
}

// Subscribe to events on the exchange. Events are sent through the returned
// channel ch. If an error is encountered, it will be sent on channel errs and
// errs will be closed. To end the subscription, cancel the provided context.
//
// Zero or more filters may be provided as strings. Only events that match
// *any* of the provided filters will be sent on the channel. The filters use
// the standard containerd filters package syntax.
func (e *Exchange) Subscribe(ctx context.Context, fs ...string) (ch <-chan *v1.Envelope, errs <-chan error) {
	var (
		evch                  = make(chan *v1.Envelope)
		errq                  = make(chan error, 1)
		channel               = goevents.NewChannel(0)
		queue                 = goevents.NewQueue(channel)
		dst     goevents.Sink = queue
	)

	closeAll := func() {
		defer close(errq)
		defer e.broadcaster.Remove(dst)
		defer queue.Close()
		defer channel.Close()
	}

	ch = evch
	errs = errq

	if len(fs) > 0 {
		filter, err := filters.ParseAll(fs...)
		if err != nil {
			errq <- errors.Wrapf(err, "failed parsing subscription filters")
			closeAll()
			return
		}

		dst = goevents.NewFilter(queue, goevents.MatcherFunc(func(gev goevents.Event) bool {
			return filter.Match(adapt(gev))
		}))
	}

	e.broadcaster.Add(dst)

	go func() {
		defer closeAll()

		var err error
	loop:
		for {
			select {
			case ev := <-channel.C:
				env, ok := ev.(*v1.Envelope)
				if !ok {
					// TODO(stevvooe): For the most part, we are well protected
					// from this condition. Both Forward and Publish protect
					// from this.
					err = errors.Errorf("invalid envelope encountered %#v; please file a bug", ev)
					break
				}

				select {
				case evch <- env:
				case <-ctx.Done():
					break loop
				}
			case <-ctx.Done():
				break loop
			}
		}

		if err == nil {
			if cerr := ctx.Err(); cerr != context.Canceled {
				err = cerr
			}
		}

		errq <- err
	}()

	return
}

func validateTopic(topic string) error {
	if topic == "" {
		return errors.Wrap(errdefs.ErrInvalidArgument, "must not be empty")
	}

	if topic[0] != '/' {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "must start with '/'")
	}

	if len(topic) == 1 {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "must have at least one component")
	}

	components := strings.Split(topic[1:], "/")
	for _, component := range components {
		if err := identifiers.Validate(component); err != nil {
			return errors.Wrapf(err, "failed validation on component %q", component)
		}
	}

	return nil
}

func validateEnvelope(envelope *v1.Envelope) error {
	if err := namespaces.Validate(envelope.Namespace); err != nil {
		return errors.Wrapf(err, "event envelope has invalid namespace")
	}

	if err := validateTopic(envelope.Topic); err != nil {
		return errors.Wrapf(err, "envelope topic %q", envelope.Topic)
	}

	if envelope.Timestamp.IsZero() {
		return errors.Wrapf(errdefs.ErrInvalidArgument, "timestamp must be set on forwarded event")
	}

	return nil
}

func adapt(ev interface{}) filters.Adaptor {
	if adaptor, ok := ev.(filters.Adaptor); ok {
		return adaptor
	}

	return filters.AdapterFunc(func(fieldpath []string) (string, bool) {
		return "", false
	})
}
