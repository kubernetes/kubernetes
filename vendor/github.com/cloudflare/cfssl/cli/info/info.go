// Package info implements the info command.
package info

import (
	"encoding/json"
	"fmt"

	"github.com/cloudflare/cfssl/api/client"
	"github.com/cloudflare/cfssl/cli"
	"github.com/cloudflare/cfssl/cli/sign"
	"github.com/cloudflare/cfssl/errors"
	"github.com/cloudflare/cfssl/helpers"
	"github.com/cloudflare/cfssl/info"

	goerr "errors"
)

var infoUsageTxt = `cfssl info -- get info about a remote signer

Usage:

Get info about a remote signer:
cfssl info -remote remote_host [-label label] [-profile profile] [-label label] 

Flags:
`

var infoFlags = []string{"remote", "label", "profile", "config"}

func getInfoFromRemote(c cli.Config) (resp *info.Resp, err error) {
	req := new(info.Req)
	req.Label = c.Label
	req.Profile = c.Profile

	cert, err := helpers.LoadClientCertificate(c.MutualTLSCertFile, c.MutualTLSKeyFile)
	if err != nil {
		return
	}
	remoteCAs, err := helpers.LoadPEMCertPool(c.TLSRemoteCAs)
	if err != nil {
		return
	}
	serv := client.NewServerTLS(c.Remote, helpers.CreateTLSConfig(remoteCAs, cert))
	reqJSON, _ := json.Marshal(req)
	resp, err = serv.Info(reqJSON)
	if err != nil {
		return
	}

	_, err = helpers.ParseCertificatePEM([]byte(resp.Certificate))
	if err != nil {
		return
	}

	return
}

func getInfoFromConfig(c cli.Config) (resp *info.Resp, err error) {
	s, err := sign.SignerFromConfig(c)
	if err != nil {
		return
	}

	req := new(info.Req)
	req.Label = c.Label
	req.Profile = c.Profile

	resp, err = s.Info(*req)
	if err != nil {
		return
	}

	return
}

func infoMain(args []string, c cli.Config) (err error) {
	if len(args) > 0 {
		return goerr.New("argument is provided but not defined; please refer to the usage by flag -h.")
	}

	var resp *info.Resp

	if c.Remote != "" {
		resp, err = getInfoFromRemote(c)
		if err != nil {
			return
		}
	} else if c.CFG != nil {
		resp, err = getInfoFromConfig(c)
		if err != nil {
			return
		}
	} else {
		return goerr.New("Either -remote or -config must be given. Refer to cfssl info -h for usage.")
	}

	respJSON, err := json.Marshal(resp)
	if err != nil {
		return errors.NewBadRequest(err)
	}
	fmt.Print(string(respJSON))
	return nil
}

// Command assembles the definition of Command 'info'
var Command = &cli.Command{
	UsageText: infoUsageTxt,
	Flags:     infoFlags,
	Main:      infoMain,
}
