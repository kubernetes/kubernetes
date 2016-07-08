package types

// ConfigKey is a configuration key.
type ConfigKey string

// String returns the string-representation of the ConfigKey.
func (k ConfigKey) String() string {
	return string(k)
}

const (
	// ConfigRoot is a config key.
	ConfigRoot = "libstorage"

	// ConfigServer is a config key.
	ConfigServer = ConfigRoot + ".server"

	// ConfigClient is a config key.
	ConfigClient = ConfigRoot + ".client"

	// ConfigClientType is a config key.
	ConfigClientType = ConfigClient + ".type"

	// ConfigHost is a config key.
	ConfigHost = ConfigRoot + ".host"

	// ConfigEmbedded is a config key.
	ConfigEmbedded = ConfigRoot + ".embedded"

	// ConfigService is a config key.
	ConfigService = ConfigRoot + ".service"

	// ConfigOSDriver is a config key.
	ConfigOSDriver = ConfigRoot + ".os.driver"

	// ConfigStorageDriver is a config key.
	ConfigStorageDriver = ConfigRoot + ".storage.driver"

	// ConfigIntegrationDriver is a config key.
	ConfigIntegrationDriver = ConfigRoot + ".integration.driver"

	// ConfigLogging is a config key.
	ConfigLogging = ConfigRoot + ".logging"

	// ConfigLogLevel is a config key.
	ConfigLogLevel = ConfigLogging + ".level"

	// ConfigLogStdout is a config key.
	ConfigLogStdout = ConfigLogging + ".stdout"

	// ConfigLogStderr is a config key.
	ConfigLogStderr = ConfigLogging + ".stderr"

	// ConfigLogHTTPRequests is a config key.
	ConfigLogHTTPRequests = ConfigLogging + ".httpRequests"

	// ConfigLogHTTPResponses is a config key.
	ConfigLogHTTPResponses = ConfigLogging + ".httpResponses"

	// ConfigHTTPDisableKeepAlive is a config key.
	ConfigHTTPDisableKeepAlive = ConfigRoot + ".http.disableKeepAlive"

	// ConfigHTTPWriteTimeout is a config key.
	ConfigHTTPWriteTimeout = ConfigRoot + ".http.writeTimeout"

	// ConfigHTTPReadTimeout is a config key.
	ConfigHTTPReadTimeout = ConfigRoot + ".http.readTimeout"

	// ConfigServices is a config key.
	ConfigServices = ConfigServer + ".services"

	// ConfigServerAutoEndpointMode is a config key.
	ConfigServerAutoEndpointMode = ConfigServer + ".autoEndpointMode"

	// ConfigEndpoints is a config key.
	ConfigEndpoints = ConfigServer + ".endpoints"

	// ConfigExecutorPath is a config key.
	ConfigExecutorPath = ConfigRoot + ".executor.path"

	// ConfigExecutorNoDownload is a config key.
	ConfigExecutorNoDownload = ConfigRoot + ".executor.disableDownload"

	// ConfigClientCacheInstanceID is a config key.
	ConfigClientCacheInstanceID = ConfigClient + ".cache.instanceID"

	// ConfigTLS is a config key.
	ConfigTLS = ConfigRoot + ".tls"

	// ConfigTLSDisabled is a config key.
	ConfigTLSDisabled = ConfigTLS + ".disabled"

	// ConfigTLSServerName is a config key.
	ConfigTLSServerName = ConfigTLS + ".serverName"

	// ConfigTLSClientCertRequired is a config key.
	ConfigTLSClientCertRequired = ConfigTLS + ".clientCertRequired"

	// ConfigTLSTrustedCertsFile is a config key.
	ConfigTLSTrustedCertsFile = ConfigTLS + ".trustedCertsFile"

	// ConfigTLSCertFile is a config key.
	ConfigTLSCertFile = ConfigTLS + ".certFile"

	// ConfigTLSKeyFile is a config key.
	ConfigTLSKeyFile = ConfigTLS + ".keyFile"

	// ConfigDeviceAttachTimeout is a config key.
	ConfigDeviceAttachTimeout = ConfigRoot + ".device.attachTimeout"

	// ConfigDeviceScanType is a config key.
	ConfigDeviceScanType = ConfigRoot + ".device.scanType"
)
