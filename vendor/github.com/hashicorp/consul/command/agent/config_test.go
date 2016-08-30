package agent

import (
	"bytes"
	"encoding/base64"
	"io/ioutil"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"strings"
	"testing"
	"time"

	"github.com/hashicorp/consul/lib"
)

func TestConfigEncryptBytes(t *testing.T) {
	// Test with some input
	src := []byte("abc")
	c := &Config{
		EncryptKey: base64.StdEncoding.EncodeToString(src),
	}

	result, err := c.EncryptBytes()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !bytes.Equal(src, result) {
		t.Fatalf("bad: %#v", result)
	}

	// Test with no input
	c = &Config{}
	result, err = c.EncryptBytes()
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(result) > 0 {
		t.Fatalf("bad: %#v", result)
	}
}

func TestDecodeConfig(t *testing.T) {
	// Basics
	input := `{"data_dir": "/tmp/", "log_level": "debug"}`
	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.DataDir != "/tmp/" {
		t.Fatalf("bad: %#v", config)
	}

	if config.LogLevel != "debug" {
		t.Fatalf("bad: %#v", config)
	}

	// Without a protocol
	input = `{"node_name": "foo", "datacenter": "dc2"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.NodeName != "foo" {
		t.Fatalf("bad: %#v", config)
	}

	if config.Datacenter != "dc2" {
		t.Fatalf("bad: %#v", config)
	}

	if config.SkipLeaveOnInt != DefaultConfig().SkipLeaveOnInt {
		t.Fatalf("bad: %#v", config)
	}

	if config.LeaveOnTerm != DefaultConfig().LeaveOnTerm {
		t.Fatalf("bad: %#v", config)
	}

	// Server bootstrap
	input = `{"server": true, "bootstrap": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.Server {
		t.Fatalf("bad: %#v", config)
	}

	if !config.Bootstrap {
		t.Fatalf("bad: %#v", config)
	}

	// Expect bootstrap
	input = `{"server": true, "bootstrap_expect": 3}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.Server {
		t.Fatalf("bad: %#v", config)
	}

	if config.BootstrapExpect != 3 {
		t.Fatalf("bad: %#v", config)
	}

	// DNS setup
	input = `{"ports": {"dns": 8500}, "recursors": ["8.8.8.8","8.8.4.4"], "recursor":"127.0.0.1", "domain": "foobar"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Ports.DNS != 8500 {
		t.Fatalf("bad: %#v", config)
	}

	if len(config.DNSRecursors) != 3 {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSRecursors[0] != "8.8.8.8" {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSRecursors[1] != "8.8.4.4" {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSRecursors[2] != "127.0.0.1" {
		t.Fatalf("bad: %#v", config)
	}

	if config.Domain != "foobar" {
		t.Fatalf("bad: %#v", config)
	}

	// RPC configs
	input = `{"ports": {"http": 1234, "https": 1243, "rpc": 8100}, "client_addr": "0.0.0.0"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.ClientAddr != "0.0.0.0" {
		t.Fatalf("bad: %#v", config)
	}

	if config.Ports.HTTP != 1234 {
		t.Fatalf("bad: %#v", config)
	}

	if config.Ports.HTTPS != 1243 {
		t.Fatalf("bad: %#v", config)
	}

	if config.Ports.RPC != 8100 {
		t.Fatalf("bad: %#v", config)
	}

	// Serf configs
	input = `{"ports": {"serf_lan": 1000, "serf_wan": 2000}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Ports.SerfLan != 1000 {
		t.Fatalf("bad: %#v", config)
	}

	if config.Ports.SerfWan != 2000 {
		t.Fatalf("bad: %#v", config)
	}

	// Server addrs
	input = `{"ports": {"server": 8000}, "bind_addr": "127.0.0.2", "advertise_addr": "127.0.0.3"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.BindAddr != "127.0.0.2" {
		t.Fatalf("bad: %#v", config)
	}

	if config.AdvertiseAddr != "127.0.0.3" {
		t.Fatalf("bad: %#v", config)
	}

	if config.AdvertiseAddrWan != "" {
		t.Fatalf("bad: %#v", config)
	}

	if config.Ports.Server != 8000 {
		t.Fatalf("bad: %#v", config)
	}

	// Advertise address for wan
	input = `{"advertise_addr_wan": "127.0.0.5"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.AdvertiseAddr != "" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AdvertiseAddrWan != "127.0.0.5" {
		t.Fatalf("bad: %#v", config)
	}

	// Advertise addresses for serflan
	input = `{"advertise_addrs": {"serf_lan": "127.0.0.5:1234"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.AdvertiseAddrs.SerfLanRaw != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AdvertiseAddrs.SerfLan.String() != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}

	// Advertise addresses for serfwan
	input = `{"advertise_addrs": {"serf_wan": "127.0.0.5:1234"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.AdvertiseAddrs.SerfWanRaw != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AdvertiseAddrs.SerfWan.String() != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}

	// Advertise addresses for rpc
	input = `{"advertise_addrs": {"rpc": "127.0.0.5:1234"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.AdvertiseAddrs.RPCRaw != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AdvertiseAddrs.RPC.String() != "127.0.0.5:1234" {
		t.Fatalf("bad: %#v", config)
	}

	// WAN address translation disabled by default
	config, err = DecodeConfig(bytes.NewReader([]byte(`{}`)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.TranslateWanAddrs != false {
		t.Fatalf("bad: %#v", config)
	}

	// WAN address translation
	input = `{"translate_wan_addrs": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.TranslateWanAddrs != true {
		t.Fatalf("bad: %#v", config)
	}

	// leave_on_terminate
	input = `{"leave_on_terminate": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.LeaveOnTerm != true {
		t.Fatalf("bad: %#v", config)
	}

	// skip_leave_on_interrupt
	input = `{"skip_leave_on_interrupt": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.SkipLeaveOnInt != true {
		t.Fatalf("bad: %#v", config)
	}

	// enable_debug
	input = `{"enable_debug": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.EnableDebug != true {
		t.Fatalf("bad: %#v", config)
	}

	// TLS
	input = `{"verify_incoming": true, "verify_outgoing": true, "verify_server_hostname": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.VerifyIncoming != true {
		t.Fatalf("bad: %#v", config)
	}

	if config.VerifyOutgoing != true {
		t.Fatalf("bad: %#v", config)
	}

	if config.VerifyServerHostname != true {
		t.Fatalf("bad: %#v", config)
	}

	// TLS keys
	input = `{"ca_file": "my/ca/file", "cert_file": "my.cert", "key_file": "key.pem", "server_name": "example.com"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.CAFile != "my/ca/file" {
		t.Fatalf("bad: %#v", config)
	}
	if config.CertFile != "my.cert" {
		t.Fatalf("bad: %#v", config)
	}
	if config.KeyFile != "key.pem" {
		t.Fatalf("bad: %#v", config)
	}
	if config.ServerName != "example.com" {
		t.Fatalf("bad: %#v", config)
	}

	// Start join
	input = `{"start_join": ["1.1.1.1", "2.2.2.2"]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.StartJoin) != 2 {
		t.Fatalf("bad: %#v", config)
	}
	if config.StartJoin[0] != "1.1.1.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.StartJoin[1] != "2.2.2.2" {
		t.Fatalf("bad: %#v", config)
	}

	// Start Join wan
	input = `{"start_join_wan": ["1.1.1.1", "2.2.2.2"]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.StartJoinWan) != 2 {
		t.Fatalf("bad: %#v", config)
	}
	if config.StartJoinWan[0] != "1.1.1.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.StartJoinWan[1] != "2.2.2.2" {
		t.Fatalf("bad: %#v", config)
	}

	// Retry join
	input = `{"retry_join": ["1.1.1.1", "2.2.2.2"]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.RetryJoin) != 2 {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryJoin[0] != "1.1.1.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryJoin[1] != "2.2.2.2" {
		t.Fatalf("bad: %#v", config)
	}

	// Retry interval
	input = `{"retry_interval": "10s"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.RetryIntervalRaw != "10s" {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryInterval.String() != "10s" {
		t.Fatalf("bad: %#v", config)
	}

	// Retry Max
	input = `{"retry_max": 3}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.RetryMaxAttempts != 3 {
		t.Fatalf("bad: %#v", config)
	}

	// Retry Join wan
	input = `{"retry_join_wan": ["1.1.1.1", "2.2.2.2"]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.RetryJoinWan) != 2 {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryJoinWan[0] != "1.1.1.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryJoinWan[1] != "2.2.2.2" {
		t.Fatalf("bad: %#v", config)
	}

	// Retry Interval wan
	input = `{"retry_interval_wan": "10s"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.RetryIntervalWanRaw != "10s" {
		t.Fatalf("bad: %#v", config)
	}
	if config.RetryIntervalWan.String() != "10s" {
		t.Fatalf("bad: %#v", config)
	}

	// Retry Max wan
	input = `{"retry_max_wan": 3}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.RetryMaxAttemptsWan != 3 {
		t.Fatalf("bad: %#v", config)
	}

	// Static UI server
	input = `{"ui": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.EnableUi {
		t.Fatalf("bad: %#v", config)
	}

	// UI Dir
	input = `{"ui_dir": "/opt/consul-ui"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.UiDir != "/opt/consul-ui" {
		t.Fatalf("bad: %#v", config)
	}

	// Pid File
	input = `{"pid_file": "/tmp/consul/pid"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.PidFile != "/tmp/consul/pid" {
		t.Fatalf("bad: %#v", config)
	}

	// Syslog
	input = `{"enable_syslog": true, "syslog_facility": "LOCAL4"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.EnableSyslog {
		t.Fatalf("bad: %#v", config)
	}
	if config.SyslogFacility != "LOCAL4" {
		t.Fatalf("bad: %#v", config)
	}

	// Rejoin
	input = `{"rejoin_after_leave": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.RejoinAfterLeave {
		t.Fatalf("bad: %#v", config)
	}

	// DNS node ttl, max stale
	input = `{"dns_config": {"node_ttl": "5s", "max_stale": "15s", "allow_stale": true}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.DNSConfig.NodeTTL != 5*time.Second {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSConfig.MaxStale != 15*time.Second {
		t.Fatalf("bad: %#v", config)
	}
	if !config.DNSConfig.AllowStale {
		t.Fatalf("bad: %#v", config)
	}

	// DNS service ttl
	input = `{"dns_config": {"service_ttl": {"*": "1s", "api": "10s", "web": "30s"}}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.DNSConfig.ServiceTTL["*"] != time.Second {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSConfig.ServiceTTL["api"] != 10*time.Second {
		t.Fatalf("bad: %#v", config)
	}
	if config.DNSConfig.ServiceTTL["web"] != 30*time.Second {
		t.Fatalf("bad: %#v", config)
	}

	// DNS enable truncate
	input = `{"dns_config": {"enable_truncate": true}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.DNSConfig.EnableTruncate {
		t.Fatalf("bad: %#v", config)
	}

	// DNS only passing
	input = `{"dns_config": {"only_passing": true}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.DNSConfig.OnlyPassing {
		t.Fatalf("bad: %#v", config)
	}

	// CheckUpdateInterval
	input = `{"check_update_interval": "10m"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.CheckUpdateInterval != 10*time.Minute {
		t.Fatalf("bad: %#v", config)
	}

	// ACLs
	input = `{"acl_token": "1234", "acl_datacenter": "dc2",
	"acl_ttl": "60s", "acl_down_policy": "deny",
	"acl_default_policy": "deny", "acl_master_token": "2345"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.ACLToken != "1234" {
		t.Fatalf("bad: %#v", config)
	}
	if config.ACLMasterToken != "2345" {
		t.Fatalf("bad: %#v", config)
	}
	if config.ACLDatacenter != "dc2" {
		t.Fatalf("bad: %#v", config)
	}
	if config.ACLTTL != 60*time.Second {
		t.Fatalf("bad: %#v", config)
	}
	if config.ACLDownPolicy != "deny" {
		t.Fatalf("bad: %#v", config)
	}
	if config.ACLDefaultPolicy != "deny" {
		t.Fatalf("bad: %#v", config)
	}

	// Watches
	input = `{"watches": [{"type":"keyprefix", "prefix":"foo/", "handler":"foobar"}]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.Watches) != 1 {
		t.Fatalf("bad: %#v", config)
	}

	out := config.Watches[0]
	exp := map[string]interface{}{
		"type":    "keyprefix",
		"prefix":  "foo/",
		"handler": "foobar",
	}
	if !reflect.DeepEqual(out, exp) {
		t.Fatalf("bad: %#v", config)
	}

	// remote exec
	input = `{"disable_remote_exec": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.DisableRemoteExec {
		t.Fatalf("bad: %#v", config)
	}

	// stats(d|ite) exec
	input = `{"statsite_addr": "127.0.0.1:7250", "statsd_addr": "127.0.0.1:7251"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Telemetry.StatsiteAddr != "127.0.0.1:7250" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.StatsdAddr != "127.0.0.1:7251" {
		t.Fatalf("bad: %#v", config)
	}

	// dogstatsd
	input = `{"dogstatsd_addr": "127.0.0.1:7254", "dogstatsd_tags":["tag_1:val_1", "tag_2:val_2"]}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.Telemetry.DogStatsdAddr != "127.0.0.1:7254" {
		t.Fatalf("bad: %#v", config)
	}

	if len(config.Telemetry.DogStatsdTags) != 2 {
		t.Fatalf("bad: %#v", config)
	}

	if config.Telemetry.DogStatsdTags[0] != "tag_1:val_1" {
		t.Fatalf("bad: %#v", config)
	}

	if config.Telemetry.DogStatsdTags[1] != "tag_2:val_2" {
		t.Fatalf("bad: %#v", config)
	}

	// Statsite prefix
	input = `{"statsite_prefix": "my_prefix"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.Telemetry.StatsitePrefix != "my_prefix" {
		t.Fatalf("bad: %#v", config)
	}

	// New telemetry
	input = `{"telemetry": { "statsite_prefix": "my_prefix", "statsite_address": "127.0.0.1:7250", "statsd_address":"127.0.0.1:7251", "disable_hostname": true, "dogstatsd_addr": "1.1.1.1:111", "dogstatsd_tags": [ "tag_1:val_1" ] } }`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	if config.Telemetry.StatsitePrefix != "my_prefix" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.StatsiteAddr != "127.0.0.1:7250" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.StatsdAddr != "127.0.0.1:7251" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.DisableHostname != true {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.DogStatsdAddr != "1.1.1.1:111" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Telemetry.DogStatsdTags[0] != "tag_1:val_1" {
		t.Fatalf("bad: %#v", config)
	}

	// Address overrides
	input = `{"addresses": {"dns": "0.0.0.0", "http": "127.0.0.1", "https": "127.0.0.1", "rpc": "127.0.0.1"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Addresses.DNS != "0.0.0.0" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Addresses.HTTP != "127.0.0.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Addresses.HTTPS != "127.0.0.1" {
		t.Fatalf("bad: %#v", config)
	}
	if config.Addresses.RPC != "127.0.0.1" {
		t.Fatalf("bad: %#v", config)
	}

	// Domain socket permissions
	input = `{"unix_sockets": {"user": "500", "group": "500", "mode": "0700"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.UnixSockets.Usr != "500" {
		t.Fatalf("bad: %#v", config)
	}
	if config.UnixSockets.Grp != "500" {
		t.Fatalf("bad: %#v", config)
	}
	if config.UnixSockets.Perms != "0700" {
		t.Fatalf("bad: %#v", config)
	}

	// Disable updates
	input = `{"disable_update_check": true, "disable_anonymous_signature": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if !config.DisableUpdateCheck {
		t.Fatalf("bad: %#v", config)
	}
	if !config.DisableAnonymousSignature {
		t.Fatalf("bad: %#v", config)
	}

	// HTTP API response header fields
	input = `{"http_api_response_headers": {"Access-Control-Allow-Origin": "*", "X-XSS-Protection": "1; mode=block"}}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.HTTPAPIResponseHeaders["Access-Control-Allow-Origin"] != "*" {
		t.Fatalf("bad: %#v", config)
	}

	if config.HTTPAPIResponseHeaders["X-XSS-Protection"] != "1; mode=block" {
		t.Fatalf("bad: %#v", config)
	}

	// Atlas configs
	input = `{
		"atlas_infrastructure": "hashicorp/prod",
		"atlas_token": "abcdefg",
		"atlas_acl_token": "123456789",
		"atlas_join": true,
		"atlas_endpoint": "foo.bar:1111"
}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.AtlasInfrastructure != "hashicorp/prod" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AtlasToken != "abcdefg" {
		t.Fatalf("bad: %#v", config)
	}
	if config.AtlasACLToken != "123456789" {
		t.Fatalf("bad: %#v", config)
	}
	if !config.AtlasJoin {
		t.Fatalf("bad: %#v", config)
	}
	if config.AtlasEndpoint != "foo.bar:1111" {
		t.Fatalf("bad: %#v", config)
	}

	// Coordinate disable
	input = `{"disable_coordinates": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.DisableCoordinates != true {
		t.Fatalf("bad: coordinates not disabled: %#v", config)
	}

	// SessionTTLMin
	input = `{"session_ttl_min": "5s"}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.SessionTTLMin != 5*time.Second {
		t.Fatalf("bad: %s %#v", config.SessionTTLMin.String(), config)
	}

	// Reap
	input = `{"reap": true}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Reap == nil || *config.Reap != true {
		t.Fatalf("bad: reap not enabled: %#v", config)
	}

	input = `{}`
	config, err = DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.Reap != nil {
		t.Fatalf("bad: reap not tri-stated: %#v", config)
	}
}

func TestDecodeConfig_invalidKeys(t *testing.T) {
	input := `{"bad": "no way jose"}`
	_, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err == nil || !strings.Contains(err.Error(), "invalid keys") {
		t.Fatalf("should have rejected invalid config keys")
	}
}

func TestDecodeConfig_Services(t *testing.T) {
	input := `{
		"services": [
			{
				"id": "red0",
				"name": "redis",
				"tags": [
					"master"
				],
				"port": 6000,
				"check": {
					"script": "/bin/check_redis -p 6000",
					"interval": "5s",
					"ttl": "20s"
				},
				"checks": [
					{
						"script": "/bin/check_redis_read",
						"interval": "1m"
					},
					{
						"script": "/bin/check_redis_write",
						"interval": "1m"
					}
				]
			},
			{
				"id": "red1",
				"name": "redis",
				"tags": [
					"delayed",
					"slave"
				],
				"port": 7000,
				"check": {
					"script": "/bin/check_redis -p 7000",
					"interval": "30s",
					"ttl": "60s"
				}
			},
			{
				"id": "es0",
				"name": "elasticsearch",
				"port": 9200,
				"check": {
					"HTTP": "http://localhost:9200/_cluster_health",
					"interval": "10s",
					"timeout": "100ms"
				}
			}
		]
	}`

	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := &Config{
		Services: []*ServiceDefinition{
			&ServiceDefinition{
				Check: CheckType{
					Interval: 5 * time.Second,
					Script:   "/bin/check_redis -p 6000",
					TTL:      20 * time.Second,
				},
				Checks: CheckTypes{
					&CheckType{
						Interval: time.Minute,
						Script:   "/bin/check_redis_read",
					},
					&CheckType{
						Interval: time.Minute,
						Script:   "/bin/check_redis_write",
					},
				},
				ID:   "red0",
				Name: "redis",
				Tags: []string{
					"master",
				},
				Port: 6000,
			},
			&ServiceDefinition{
				Check: CheckType{
					Interval: 30 * time.Second,
					Script:   "/bin/check_redis -p 7000",
					TTL:      60 * time.Second,
				},
				ID:   "red1",
				Name: "redis",
				Tags: []string{
					"delayed",
					"slave",
				},
				Port: 7000,
			},
			&ServiceDefinition{
				Check: CheckType{
					HTTP:     "http://localhost:9200/_cluster_health",
					Interval: 10 * time.Second,
					Timeout:  100 * time.Millisecond,
				},
				ID:   "es0",
				Name: "elasticsearch",
				Port: 9200,
			},
		},
	}

	if !reflect.DeepEqual(config, expected) {
		t.Fatalf("bad: %#v", config)
	}
}

func TestDecodeConfig_Checks(t *testing.T) {
	input := `{
		"checks": [
			{
				"id": "chk1",
				"name": "mem",
				"script": "/bin/check_mem",
				"interval": "5s"
			},
			{
				"id": "chk2",
				"name": "cpu",
				"script": "/bin/check_cpu",
				"interval": "10s"
			},
			{
				"id": "chk3",
				"name": "service:redis:tx",
				"script": "/bin/check_redis_tx",
				"interval": "1m",
				"service_id": "redis"
			},
			{
				"id": "chk4",
				"name": "service:elasticsearch:health",
				"HTTP": "http://localhost:9200/_cluster_health",
				"interval": "10s",
				"timeout": "100ms",
				"service_id": "elasticsearch"
			}
		]
	}`

	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := &Config{
		Checks: []*CheckDefinition{
			&CheckDefinition{
				ID:   "chk1",
				Name: "mem",
				CheckType: CheckType{
					Script:   "/bin/check_mem",
					Interval: 5 * time.Second,
				},
			},
			&CheckDefinition{
				ID:   "chk2",
				Name: "cpu",
				CheckType: CheckType{
					Script:   "/bin/check_cpu",
					Interval: 10 * time.Second,
				},
			},
			&CheckDefinition{
				ID:        "chk3",
				Name:      "service:redis:tx",
				ServiceID: "redis",
				CheckType: CheckType{
					Script:   "/bin/check_redis_tx",
					Interval: time.Minute,
				},
			},
			&CheckDefinition{
				ID:        "chk4",
				Name:      "service:elasticsearch:health",
				ServiceID: "elasticsearch",
				CheckType: CheckType{
					HTTP:     "http://localhost:9200/_cluster_health",
					Interval: 10 * time.Second,
					Timeout:  100 * time.Millisecond,
				},
			},
		},
	}

	if !reflect.DeepEqual(config, expected) {
		t.Fatalf("bad: %#v", config)
	}
}

func TestDecodeConfig_Multiples(t *testing.T) {
	input := `{
		"services": [
			{
				"id": "red0",
				"name": "redis",
				"tags": [
					"master"
				],
				"port": 6000,
				"check": {
					"script": "/bin/check_redis -p 6000",
					"interval": "5s",
					"ttl": "20s"
				}
			}
		],
		"checks": [
			{
				"id": "chk1",
				"name": "mem",
				"script": "/bin/check_mem",
				"interval": "10s"
			}
		]
	}`

	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	expected := &Config{
		Services: []*ServiceDefinition{
			&ServiceDefinition{
				Check: CheckType{
					Interval: 5 * time.Second,
					Script:   "/bin/check_redis -p 6000",
					TTL:      20 * time.Second,
				},
				ID:   "red0",
				Name: "redis",
				Tags: []string{
					"master",
				},
				Port: 6000,
			},
		},
		Checks: []*CheckDefinition{
			&CheckDefinition{
				ID:   "chk1",
				Name: "mem",
				CheckType: CheckType{
					Script:   "/bin/check_mem",
					Interval: 10 * time.Second,
				},
			},
		},
	}

	if !reflect.DeepEqual(config, expected) {
		t.Fatalf("bad: %#v", config)
	}
}

func TestDecodeConfig_Service(t *testing.T) {
	// Basics
	input := `{"service": {"id": "red1", "name": "redis", "tags": ["master"], "port":8000, "check": {"script": "/bin/check_redis", "interval": "10s", "ttl": "15s" }}}`
	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.Services) != 1 {
		t.Fatalf("missing service")
	}

	serv := config.Services[0]
	if serv.ID != "red1" {
		t.Fatalf("bad: %v", serv)
	}

	if serv.Name != "redis" {
		t.Fatalf("bad: %v", serv)
	}

	if !lib.StrContains(serv.Tags, "master") {
		t.Fatalf("bad: %v", serv)
	}

	if serv.Port != 8000 {
		t.Fatalf("bad: %v", serv)
	}

	if serv.Check.Script != "/bin/check_redis" {
		t.Fatalf("bad: %v", serv)
	}

	if serv.Check.Interval != 10*time.Second {
		t.Fatalf("bad: %v", serv)
	}

	if serv.Check.TTL != 15*time.Second {
		t.Fatalf("bad: %v", serv)
	}
}

func TestDecodeConfig_Check(t *testing.T) {
	// Basics
	input := `{"check": {"id": "chk1", "name": "mem", "notes": "foobar", "script": "/bin/check_redis", "interval": "10s", "ttl": "15s", "shell": "/bin/bash", "docker_container_id": "redis" }}`
	config, err := DecodeConfig(bytes.NewReader([]byte(input)))
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if len(config.Checks) != 1 {
		t.Fatalf("missing check")
	}

	chk := config.Checks[0]
	if chk.ID != "chk1" {
		t.Fatalf("bad: %v", chk)
	}

	if chk.Name != "mem" {
		t.Fatalf("bad: %v", chk)
	}

	if chk.Notes != "foobar" {
		t.Fatalf("bad: %v", chk)
	}

	if chk.Script != "/bin/check_redis" {
		t.Fatalf("bad: %v", chk)
	}

	if chk.Interval != 10*time.Second {
		t.Fatalf("bad: %v", chk)
	}

	if chk.TTL != 15*time.Second {
		t.Fatalf("bad: %v", chk)
	}

	if chk.Shell != "/bin/bash" {
		t.Fatalf("bad: %v", chk)
	}

	if chk.DockerContainerID != "redis" {
		t.Fatalf("bad: %v", chk)
	}
}

func TestMergeConfig(t *testing.T) {
	a := &Config{
		Bootstrap:              false,
		BootstrapExpect:        0,
		Datacenter:             "dc1",
		DataDir:                "/tmp/foo",
		Domain:                 "basic",
		LogLevel:               "debug",
		NodeName:               "foo",
		ClientAddr:             "127.0.0.1",
		BindAddr:               "127.0.0.1",
		AdvertiseAddr:          "127.0.0.1",
		Server:                 false,
		LeaveOnTerm:            false,
		SkipLeaveOnInt:         false,
		EnableDebug:            false,
		CheckUpdateIntervalRaw: "8m",
		RetryIntervalRaw:       "10s",
		RetryIntervalWanRaw:    "10s",
		Telemetry: Telemetry{
			DisableHostname: false,
			StatsdAddr:      "nope",
			StatsiteAddr:    "nope",
			StatsitePrefix:  "nope",
			DogStatsdAddr:   "nope",
			DogStatsdTags:   []string{"nope"},
		},
	}

	b := &Config{
		Bootstrap:       true,
		BootstrapExpect: 3,
		Datacenter:      "dc2",
		DataDir:         "/tmp/bar",
		DNSRecursors:    []string{"127.0.0.2:1001"},
		DNSConfig: DNSConfig{
			NodeTTL: 10 * time.Second,
			ServiceTTL: map[string]time.Duration{
				"api": 10 * time.Second,
			},
			AllowStale:     true,
			MaxStale:       30 * time.Second,
			EnableTruncate: true,
		},
		Domain:           "other",
		LogLevel:         "info",
		NodeName:         "baz",
		ClientAddr:       "127.0.0.2",
		BindAddr:         "127.0.0.2",
		AdvertiseAddr:    "127.0.0.2",
		AdvertiseAddrWan: "127.0.0.2",
		Ports: PortConfig{
			DNS:     1,
			HTTP:    2,
			RPC:     3,
			SerfLan: 4,
			SerfWan: 5,
			Server:  6,
			HTTPS:   7,
		},
		Addresses: AddressConfig{
			DNS:   "127.0.0.1",
			HTTP:  "127.0.0.2",
			RPC:   "127.0.0.3",
			HTTPS: "127.0.0.4",
		},
		Server:                 true,
		LeaveOnTerm:            true,
		SkipLeaveOnInt:         true,
		EnableDebug:            true,
		VerifyIncoming:         true,
		VerifyOutgoing:         true,
		CAFile:                 "test/ca.pem",
		CertFile:               "test/cert.pem",
		KeyFile:                "test/key.pem",
		Checks:                 []*CheckDefinition{nil},
		Services:               []*ServiceDefinition{nil},
		StartJoin:              []string{"1.1.1.1"},
		StartJoinWan:           []string{"1.1.1.1"},
		EnableUi:               true,
		UiDir:                  "/opt/consul-ui",
		EnableSyslog:           true,
		RejoinAfterLeave:       true,
		RetryJoin:              []string{"1.1.1.1"},
		RetryIntervalRaw:       "10s",
		RetryInterval:          10 * time.Second,
		RetryJoinWan:           []string{"1.1.1.1"},
		RetryIntervalWanRaw:    "10s",
		RetryIntervalWan:       10 * time.Second,
		CheckUpdateInterval:    8 * time.Minute,
		CheckUpdateIntervalRaw: "8m",
		ACLToken:               "1234",
		ACLMasterToken:         "2345",
		ACLDatacenter:          "dc2",
		ACLTTL:                 15 * time.Second,
		ACLTTLRaw:              "15s",
		ACLDownPolicy:          "deny",
		ACLDefaultPolicy:       "deny",
		Watches: []map[string]interface{}{
			map[string]interface{}{
				"type":    "keyprefix",
				"prefix":  "foo/",
				"handler": "foobar",
			},
		},
		DisableRemoteExec: true,
		Telemetry: Telemetry{
			StatsiteAddr:    "127.0.0.1:7250",
			StatsitePrefix:  "stats_prefix",
			StatsdAddr:      "127.0.0.1:7251",
			DisableHostname: true,
			DogStatsdAddr:   "127.0.0.1:7254",
			DogStatsdTags:   []string{"tag_1:val_1", "tag_2:val_2"},
		},
		DisableUpdateCheck:        true,
		DisableAnonymousSignature: true,
		HTTPAPIResponseHeaders: map[string]string{
			"Access-Control-Allow-Origin": "*",
		},
		UnixSockets: UnixSocketConfig{
			UnixSocketPermissions{
				Usr:   "500",
				Grp:   "500",
				Perms: "0700",
			},
		},
		AtlasInfrastructure: "hashicorp/prod",
		AtlasToken:          "123456789",
		AtlasACLToken:       "abcdefgh",
		AtlasJoin:           true,
		SessionTTLMinRaw:    "1000s",
		SessionTTLMin:       1000 * time.Second,
		AdvertiseAddrs: AdvertiseAddrsConfig{
			SerfLan:    &net.TCPAddr{},
			SerfLanRaw: "127.0.0.5:1231",
			SerfWan:    &net.TCPAddr{},
			SerfWanRaw: "127.0.0.5:1232",
			RPC:        &net.TCPAddr{},
			RPCRaw:     "127.0.0.5:1233",
		},
		Reap: Bool(true),
	}

	c := MergeConfig(a, b)

	if !reflect.DeepEqual(c, b) {
		t.Fatalf("should be equal %#v %#v", c, b)
	}
}

func TestReadConfigPaths_badPath(t *testing.T) {
	_, err := ReadConfigPaths([]string{"/i/shouldnt/exist/ever/rainbows"})
	if err == nil {
		t.Fatal("should have err")
	}
}

func TestReadConfigPaths_file(t *testing.T) {
	tf, err := ioutil.TempFile("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	tf.Write([]byte(`{"node_name":"bar"}`))
	tf.Close()
	defer os.Remove(tf.Name())

	config, err := ReadConfigPaths([]string{tf.Name()})
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.NodeName != "bar" {
		t.Fatalf("bad: %#v", config)
	}
}

func TestReadConfigPaths_dir(t *testing.T) {
	td, err := ioutil.TempDir("", "consul")
	if err != nil {
		t.Fatalf("err: %s", err)
	}
	defer os.RemoveAll(td)

	err = ioutil.WriteFile(filepath.Join(td, "a.json"),
		[]byte(`{"node_name": "bar"}`), 0644)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	err = ioutil.WriteFile(filepath.Join(td, "b.json"),
		[]byte(`{"node_name": "baz"}`), 0644)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// A non-json file, shouldn't be read
	err = ioutil.WriteFile(filepath.Join(td, "c"),
		[]byte(`{"node_name": "bad"}`), 0644)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	// An empty file shouldn't be read
	err = ioutil.WriteFile(filepath.Join(td, "d.json"),
		[]byte{}, 0664)
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	config, err := ReadConfigPaths([]string{td})
	if err != nil {
		t.Fatalf("err: %s", err)
	}

	if config.NodeName != "baz" {
		t.Fatalf("bad: %#v", config)
	}
}

func TestUnixSockets(t *testing.T) {
	path1, ok := unixSocketAddr("unix:///path/to/socket")
	if !ok || path1 != "/path/to/socket" {
		t.Fatalf("bad: %v %v", ok, path1)
	}

	path2, ok := unixSocketAddr("notunix://blah")
	if ok || path2 != "" {
		t.Fatalf("bad: %v %v", ok, path2)
	}
}
