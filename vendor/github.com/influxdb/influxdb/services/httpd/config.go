package httpd

type Config struct {
	Enabled          bool   `toml:"enabled"`
	BindAddress      string `toml:"bind-address"`
	AuthEnabled      bool   `toml:"auth-enabled"`
	LogEnabled       bool   `toml:"log-enabled"`
	WriteTracing     bool   `toml:"write-tracing"`
	PprofEnabled     bool   `toml:"pprof-enabled"`
	HttpsEnabled     bool   `toml:"https-enabled"`
	HttpsCertificate string `toml:"https-certificate"`
}

func NewConfig() Config {
	return Config{
		Enabled:          true,
		BindAddress:      ":8086",
		LogEnabled:       true,
		HttpsEnabled:     false,
		HttpsCertificate: "/etc/ssl/influxdb.pem",
	}
}
