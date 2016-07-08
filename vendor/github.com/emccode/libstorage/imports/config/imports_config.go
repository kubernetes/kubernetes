package config

import (
	"os"
	"runtime"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gofig"

	"github.com/emccode/libstorage/api/types"
)

const (
	logStdoutDesc = "The file to which to log os.Stdout"
	logStderrDesc = "The file to which to log os.Stderr"
)

func init() {
	gofig.LogGetAndSet = false
	gofig.LogSecureKey = false
	gofig.LogFlattenEnvVars = false

	logLevelSz := os.Getenv("LIBSTORAGE_LOGGING_LEVEL")
	logLevel, err := log.ParseLevel(logLevelSz)
	if err != nil {
		logLevel = log.WarnLevel
	}
	log.SetLevel(logLevel)

	r := gofig.NewRegistration("libStorage")

	rk := func(
		keyType gofig.KeyType,
		defaultVal interface{},
		description string,
		keyVal types.ConfigKey,
		args ...interface{}) {

		if args == nil {
			args = []interface{}{keyVal}
		} else {
			args = append([]interface{}{keyVal}, args...)
		}

		r.Key(keyType, "", defaultVal, description, args...)
	}

	defaultAEM := types.UnixEndpoint.String()
	defaultOSDriver := runtime.GOOS
	defaultStorageDriver := types.LibStorageDriverName
	defaultIntDriver := "docker"
	defaultLogLevel := logLevel.String()
	defaultClientType := types.IntegrationClient.String()

	rk(gofig.String, "", "", types.ConfigHost)
	rk(gofig.String, "", "", types.ConfigService)
	rk(gofig.String, defaultAEM, "", types.ConfigServerAutoEndpointMode)
	rk(gofig.String, defaultOSDriver, "", types.ConfigOSDriver)
	rk(gofig.String, defaultStorageDriver, "", types.ConfigStorageDriver)
	rk(gofig.String, defaultIntDriver, "", types.ConfigIntegrationDriver)
	rk(gofig.String, defaultClientType, "", types.ConfigClientType)
	rk(gofig.String, defaultLogLevel, "", types.ConfigLogLevel)
	rk(gofig.String, "", logStdoutDesc, types.ConfigLogStderr)
	rk(gofig.String, "", logStderrDesc, types.ConfigLogStdout)
	rk(gofig.Bool, false, "", types.ConfigLogHTTPRequests)
	rk(gofig.Bool, false, "", types.ConfigLogHTTPResponses)
	rk(gofig.Bool, false, "", types.ConfigHTTPDisableKeepAlive)
	rk(gofig.Int, 300, "", types.ConfigHTTPWriteTimeout)
	rk(gofig.Int, 300, "", types.ConfigHTTPReadTimeout)
	rk(gofig.String, types.LSX.String(), "", types.ConfigExecutorPath)
	rk(gofig.Bool, false, "", types.ConfigExecutorNoDownload)
	rk(gofig.Bool, false, "", types.ConfigIgVolOpsMountPreempt)
	rk(gofig.Bool, false, "", types.ConfigIgVolOpsCreateDisable)
	rk(gofig.Bool, false, "", types.ConfigIgVolOpsRemoveDisable)
	rk(gofig.Bool, false, "", types.ConfigIgVolOpsUnmountIgnoreUsed)
	rk(gofig.Bool, true, "", types.ConfigIgVolOpsPathCache)
	rk(gofig.String, "30m", "", types.ConfigClientCacheInstanceID)
	rk(gofig.String, "30s", "", types.ConfigDeviceAttachTimeout)
	rk(gofig.Int, 0, "", types.ConfigDeviceScanType)
	rk(gofig.Bool, false, "", types.ConfigEmbedded)

	gofig.Register(r)
}
