package httpd

// Config represents a configuration for a HTTP service.
type Config struct {
	Enabled          bool   `toml:"enabled"`
	BindAddress      string `toml:"bind-address"`
	AuthEnabled      bool   `toml:"auth-enabled"`
	LogEnabled       bool   `toml:"log-enabled"`
	WriteTracing     bool   `toml:"write-tracing"`
	PprofEnabled     bool   `toml:"pprof-enabled"`
	HTTPSEnabled     bool   `toml:"https-enabled"`
	HTTPSCertificate string `toml:"https-certificate"`
}

// NewConfig returns a new Config with default settings.
func NewConfig() Config {
	return Config{
		Enabled:          true,
		BindAddress:      ":8086",
		LogEnabled:       true,
		HTTPSEnabled:     false,
		HTTPSCertificate: "/etc/ssl/influxdb.pem",
	}
}
