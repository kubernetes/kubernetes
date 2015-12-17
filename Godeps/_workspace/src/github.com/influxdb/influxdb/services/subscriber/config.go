package subscriber

// Config represents a configuration of the subscriber service.
type Config struct {
	// Whether to enable to Subscriber service
	Enabled bool `toml:"enabled"`
}

// NewConfig returns a new instance of a subscriber config.
func NewConfig() Config {
	return Config{Enabled: true}
}
