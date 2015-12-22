package admin

const (
	// DefaultBindAddress is the default bind address for the HTTP server.
	DefaultBindAddress = ":8083"
)

// Config represents the configuration for the admin service.
type Config struct {
	Enabled          bool   `toml:"enabled"`
	BindAddress      string `toml:"bind-address"`
	HTTPSEnabled     bool   `toml:"https-enabled"`
	HTTPSCertificate string `toml:"https-certificate"`
}

// NewConfig returns an instance of Config with defaults.
func NewConfig() Config {
	return Config{
		BindAddress:      DefaultBindAddress,
		HTTPSEnabled:     false,
		HTTPSCertificate: "/etc/ssl/influxdb.pem",
	}
}
