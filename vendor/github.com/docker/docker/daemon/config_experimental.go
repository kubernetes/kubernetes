// +build experimental

package daemon

import flag "github.com/docker/docker/pkg/mflag"

func (config *Config) attachExperimentalFlags() {
	flag.StringVar(&config.DefaultNetwork, []string{"-default-network"}, "", "Set default network")
	flag.StringVar(&config.NetworkKVStore, []string{"-kv-store"}, "", "Set KV Store configuration")
}
