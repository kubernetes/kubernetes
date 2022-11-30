package workqueue

type QueueConfig struct {
	metricsProvider MetricsProvider
}

// QueueOption is an interface for applying queue configuration options.
type QueueOption interface {
	apply(*QueueConfig)
}

type optionFunc func(*QueueConfig)

func (fn optionFunc) apply(config *QueueConfig) {
	fn(config)
}

var _ QueueOption = optionFunc(nil)

// NewConfig creates a new QueueConfig and applies all the given options.
func NewConfig(opts ...QueueOption) *QueueConfig {
	config := &QueueConfig{}
	for _, o := range opts {
		o.apply(config)
	}
	return config
}

// WithMetricsProvider allows specifying a metrics provider to use for the queue
// instead of the global provider.
func WithMetricsProvider(provider MetricsProvider) QueueOption {
	return optionFunc(func(config *QueueConfig) {
		config.metricsProvider = provider
	})
}
