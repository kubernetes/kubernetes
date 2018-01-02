// Copyright © 2015 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

// Package remote integrates the remote features of Viper.
package remote

import (
	"bytes"
	"github.com/spf13/viper"
	crypt "github.com/xordataexchange/crypt/config"
	"io"
	"os"
)

type remoteConfigProvider struct{}

func (rc remoteConfigProvider) Get(rp viper.RemoteProvider) (io.Reader, error) {
	cm, err := getConfigManager(rp)
	if err != nil {
		return nil, err
	}
	b, err := cm.Get(rp.Path())
	if err != nil {
		return nil, err
	}
	return bytes.NewReader(b), nil
}

func (rc remoteConfigProvider) Watch(rp viper.RemoteProvider) (io.Reader, error) {
	cm, err := getConfigManager(rp)
	if err != nil {
		return nil, err
	}
	resp := <-cm.Watch(rp.Path(), nil)
	err = resp.Error
	if err != nil {
		return nil, err
	}

	return bytes.NewReader(resp.Value), nil
}

func getConfigManager(rp viper.RemoteProvider) (crypt.ConfigManager, error) {

	var cm crypt.ConfigManager
	var err error

	if rp.SecretKeyring() != "" {
		kr, err := os.Open(rp.SecretKeyring())
		defer kr.Close()
		if err != nil {
			return nil, err
		}
		if rp.Provider() == "etcd" {
			cm, err = crypt.NewEtcdConfigManager([]string{rp.Endpoint()}, kr)
		} else {
			cm, err = crypt.NewConsulConfigManager([]string{rp.Endpoint()}, kr)
		}
	} else {
		if rp.Provider() == "etcd" {
			cm, err = crypt.NewStandardEtcdConfigManager([]string{rp.Endpoint()})
		} else {
			cm, err = crypt.NewStandardConsulConfigManager([]string{rp.Endpoint()})
		}
	}
	if err != nil {
		return nil, err
	}
	return cm, nil

}

func init() {
	viper.RemoteConfig = &remoteConfigProvider{}
}
