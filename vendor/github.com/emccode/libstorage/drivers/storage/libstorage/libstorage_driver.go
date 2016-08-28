package libstorage

import (
	"crypto/tls"
	"net"
	"net/http"
	"time"

	log "github.com/Sirupsen/logrus"
	"github.com/akutz/gofig"
	"github.com/akutz/gotil"

	apiclient "github.com/emccode/libstorage/api/client"
	"github.com/emccode/libstorage/api/context"
	"github.com/emccode/libstorage/api/types"
	"github.com/emccode/libstorage/api/utils"
)

var (
	// EnableInstanceIDHeaders is a flag indicating whether or not the
	// client will automatically send the instance ID header(s) along with
	// storage-related API requests. The default is enabled.
	EnableInstanceIDHeaders = true

	// EnableLocalDevicesHeaders is a flag indicating whether or not the
	// client will automatically send the local devices header(s) along with
	// storage-related API requests. The default is enabled.
	EnableLocalDevicesHeaders = true
)

type driver struct {
	client
}

func newDriver() types.StorageDriver {
	return &driver{}
}

func (d *driver) Init(ctx types.Context, config gofig.Config) error {
	logFields := log.Fields{}

	addr := config.GetString(types.ConfigHost)
	d.ctx = ctx.WithValue(context.HostKey, addr)
	d.ctx.Debug("got configured host address")

	proto, lAddr, err := gotil.ParseAddress(addr)
	if err != nil {
		return err
	}

	tlsConfig, err := utils.ParseTLSConfig(
		config, logFields, "libstorage.client")
	if err != nil {
		return err
	}

	host := getHost(proto, lAddr, tlsConfig)
	lsxPath := config.GetString(types.ConfigExecutorPath)
	cliType := types.ParseClientType(config.GetString(types.ConfigClientType))
	disableKeepAlive := config.GetBool(types.ConfigHTTPDisableKeepAlive)

	logFields["host"] = host
	logFields["lsxPath"] = lsxPath
	logFields["clientType"] = cliType
	logFields["disableKeepAlive"] = disableKeepAlive

	httpTransport := &http.Transport{
		Dial: func(string, string) (net.Conn, error) {
			if tlsConfig == nil {
				return net.Dial(proto, lAddr)
			}
			return tls.Dial(proto, lAddr, tlsConfig)
		},
		DisableKeepAlives: disableKeepAlive,
	}

	apiClient := apiclient.New(host, httpTransport)
	logReq := config.GetBool(types.ConfigLogHTTPRequests)
	logRes := config.GetBool(types.ConfigLogHTTPResponses)
	apiClient.LogRequests(logReq)
	apiClient.LogResponses(logRes)

	logFields["enableInstanceIDHeaders"] = EnableInstanceIDHeaders
	logFields["enableLocalDevicesHeaders"] = EnableLocalDevicesHeaders
	logFields["logRequests"] = logReq
	logFields["logResponses"] = logRes

	d.client = client{
		APIClient:    apiClient,
		ctx:          ctx,
		config:       config,
		clientType:   cliType,
		serviceCache: &lss{Store: utils.NewStore()},
	}

	if d.clientType == types.IntegrationClient {

		newIIDCache := utils.NewStore
		dur, err := time.ParseDuration(
			config.GetString(types.ConfigClientCacheInstanceID))
		if err != nil {
			logFields["iidCacheDuration"] = dur.String()
			newIIDCache = func() types.Store {
				return utils.NewTTLStore(dur, true)
			}
		}

		d.lsxCache = &lss{Store: utils.NewStore()}
		d.instanceIDCache = &lss{Store: newIIDCache()}
	}

	d.ctx.WithFields(logFields).Info("created libStorage client")

	if err := d.dial(ctx); err != nil {
		return err
	}

	d.ctx.Info("successefully dialed libStorage server")
	return nil
}
