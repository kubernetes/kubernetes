package libstorage

import (
	"crypto/md5"
	"crypto/tls"
	"fmt"
	"io"
	"os"

	"github.com/akutz/gofig"
	"github.com/akutz/goof"

	"github.com/emccode/libstorage/api/context"
	"github.com/emccode/libstorage/api/types"
	"github.com/emccode/libstorage/api/utils"
)

type client struct {
	types.APIClient
	ctx             types.Context
	config          gofig.Config
	clientType      types.ClientType
	serviceCache    *lss
	lsxCache        *lss
	instanceIDCache types.Store
}

func (c *client) isController() bool {
	return c.clientType == types.ControllerClient
}

func (c *client) dial(ctx types.Context) error {

	ctx.WithField("path", lsxMutex).Info("lsx lock file path")

	svcInfos, err := c.Services(ctx)
	if err != nil {
		return err
	}

	// controller clients do not have any additional dialer logic
	if c.isController() {
		return nil
	}

	store := utils.NewStore()
	c.ctx = c.ctx.WithValue(context.ServerKey, c.ServerName())

	if !c.config.GetBool(types.ConfigExecutorNoDownload) {

		ctx.Info("initializing executors cache")
		if _, err := c.Executors(ctx); err != nil {
			return err
		}

		if err := c.updateExecutor(ctx); err != nil {
			return err
		}
	}

	for service, _ := range svcInfos {
		ctx := c.ctx.WithValue(context.ServiceKey, service)
		ctx.Info("initializing instance ID cache")
		if _, err := c.InstanceID(ctx, store); err != nil {
			return err
		}
	}

	return nil
}

func getHost(proto, lAddr string, tlsConfig *tls.Config) string {
	if tlsConfig != nil && tlsConfig.ServerName != "" {
		return tlsConfig.ServerName
	} else if proto == "unix" {
		return "libstorage-server"
	} else {
		return lAddr
	}
}

func (c *client) getServiceInfo(service string) (*types.ServiceInfo, error) {

	if si := c.serviceCache.GetServiceInfo(service); si != nil {
		return si, nil
	}
	return nil, goof.WithField("name", service, "unknown service")
}

func (c *client) updateExecutor(ctx types.Context) error {

	if c.isController() {
		return utils.NewUnsupportedForClientTypeError(
			c.clientType, "updateExecutor")
	}

	ctx.Debug("updating executor")

	lsxi := c.lsxCache.GetExecutorInfo(types.LSX.Name())
	if lsxi == nil {
		return goof.WithField("lsx", types.LSX, "unknown executor")
	}

	ctx.Debug("waiting on executor lock")
	if err := c.lsxMutexWait(); err != nil {
		return err
	}
	defer func() {
		ctx.Debug("signalling executor lock")
		if err := c.lsxMutexSignal(); err != nil {
			panic(err)
		}
	}()

	if !types.LSX.Exists() {
		return c.downloadExecutor(ctx)
	}

	checksum, err := c.getExecutorChecksum(ctx)
	if err != nil {
		return err
	}

	if lsxi.MD5Checksum != checksum {
		return c.downloadExecutor(ctx)
	}

	return nil
}

func (c *client) getExecutorChecksum(ctx types.Context) (string, error) {

	if c.isController() {
		return "", utils.NewUnsupportedForClientTypeError(
			c.clientType, "getExecutorChecksum")
	}

	ctx.Debug("getting executor checksum")

	f, err := os.Open(types.LSX.String())
	if err != nil {
		return "", err
	}
	defer f.Close()

	h := md5.New()
	buf := make([]byte, 1024)
	for {
		n, err := f.Read(buf)
		if err == io.EOF {
			break
		}
		if err != nil {
			return "", err
		}
		if _, err := h.Write(buf[:n]); err != nil {
			return "", err
		}
	}

	return fmt.Sprintf("%x", h.Sum(nil)), nil
}

func (c *client) downloadExecutor(ctx types.Context) error {

	if c.isController() {
		return utils.NewUnsupportedForClientTypeError(
			c.clientType, "downloadExecutor")
	}

	ctx.Debug("downloading executor")

	f, err := os.OpenFile(
		types.LSX.String(),
		os.O_CREATE|os.O_RDWR|os.O_TRUNC,
		0755)
	if err != nil {
		return err
	}

	defer f.Close()

	rdr, err := c.APIClient.ExecutorGet(ctx, types.LSX.Name())
	if _, err := io.Copy(f, rdr); err != nil {
		return err
	}

	if err := f.Sync(); err != nil {
		return err
	}

	return nil
}
