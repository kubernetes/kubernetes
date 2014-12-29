/*
Copyright 2014 Google Inc. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package clientcmd

import (
	"bytes"
	"fmt"
	"io"
	"io/ioutil"
	"os"
	"reflect"
	"strings"
	"testing"

	"github.com/spf13/pflag"

	"github.com/GoogleCloudPlatform/kubernetes/pkg/api/latest"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/client"
	"github.com/GoogleCloudPlatform/kubernetes/pkg/clientauth"
)

func TestSetAllArgumentsOnly(t *testing.T) {
	flags := pflag.NewFlagSet("test-flags", pflag.ContinueOnError)
	clientBuilder := NewBuilder(nil)
	clientBuilder.BindFlags(flags)

	args := argValues{"https://localhost:8080", "v1beta1", "/auth-path", "cert-file", "key-file", "ca-file", "bearer-token", true, true}
	flags.Parse(strings.Split(args.toArguments(), " "))

	castBuilder, ok := clientBuilder.(*builder)
	if !ok {
		t.Errorf("Got unexpected cast result: %#v", castBuilder)
	}

	matchStringArg(args.server, castBuilder.apiserver, t)
	matchStringArg(args.apiVersion, castBuilder.apiVersion, t)
	matchStringArg(args.authPath, castBuilder.authPath, t)
	matchStringArg(args.certFile, castBuilder.cmdAuthInfo.CertFile.Value, t)
	matchStringArg(args.keyFile, castBuilder.cmdAuthInfo.KeyFile.Value, t)
	matchStringArg(args.caFile, castBuilder.cmdAuthInfo.CAFile.Value, t)
	matchStringArg(args.bearerToken, castBuilder.cmdAuthInfo.BearerToken.Value, t)
	matchBoolArg(args.insecure, castBuilder.cmdAuthInfo.Insecure.Value, t)
	matchBoolArg(args.matchApiVersion, castBuilder.matchApiVersion, t)

	clientConfig, err := clientBuilder.Config()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(args.server, clientConfig.Host, t)
	matchStringArg(args.apiVersion, clientConfig.Version, t)
	matchStringArg(args.certFile, clientConfig.CertFile, t)
	matchStringArg(args.keyFile, clientConfig.KeyFile, t)
	matchStringArg(args.caFile, clientConfig.CAFile, t)
	matchStringArg(args.bearerToken, clientConfig.BearerToken, t)
	matchBoolArg(args.insecure, clientConfig.Insecure, t)
}

func TestSetInsecureArgumentsOnly(t *testing.T) {
	flags := pflag.NewFlagSet("test-flags", pflag.ContinueOnError)
	clientBuilder := NewBuilder(nil)
	clientBuilder.BindFlags(flags)

	args := argValues{"http://localhost:8080", "v1beta1", "/auth-path", "cert-file", "key-file", "ca-file", "bearer-token", true, true}
	flags.Parse(strings.Split(args.toArguments(), " "))

	clientConfig, err := clientBuilder.Config()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(args.server, clientConfig.Host, t)
	matchStringArg(args.apiVersion, clientConfig.Version, t)

	// all security related params should be empty in the resulting config even though we set them because we're using http transport
	matchStringArg("", clientConfig.CertFile, t)
	matchStringArg("", clientConfig.KeyFile, t)
	matchStringArg("", clientConfig.CAFile, t)
	matchStringArg("", clientConfig.BearerToken, t)
	matchBoolArg(false, clientConfig.Insecure, t)
}

func TestReadAuthFile(t *testing.T) {
	flags := pflag.NewFlagSet("test-flags", pflag.ContinueOnError)
	clientBuilder := NewBuilder(nil)
	clientBuilder.BindFlags(flags)
	authFileContents := fmt.Sprintf(`{"user": "alfa-user", "password": "bravo-password", "cAFile": "charlie", "certFile": "delta", "keyFile": "echo", "bearerToken": "foxtrot"}`)
	authFile := writeTempAuthFile(authFileContents, t)

	args := argValues{"https://localhost:8080", "v1beta1", authFile, "", "", "", "", true, true}
	flags.Parse(strings.Split(args.toArguments(), " "))

	castBuilder, ok := clientBuilder.(*builder)
	if !ok {
		t.Errorf("Got unexpected cast result: %#v", castBuilder)
	}

	matchStringArg(args.server, castBuilder.apiserver, t)
	matchStringArg(args.apiVersion, castBuilder.apiVersion, t)
	matchStringArg(args.authPath, castBuilder.authPath, t)
	matchStringArg(args.certFile, castBuilder.cmdAuthInfo.CertFile.Value, t)
	matchStringArg(args.keyFile, castBuilder.cmdAuthInfo.KeyFile.Value, t)
	matchStringArg(args.caFile, castBuilder.cmdAuthInfo.CAFile.Value, t)
	matchStringArg(args.bearerToken, castBuilder.cmdAuthInfo.BearerToken.Value, t)
	matchBoolArg(args.insecure, castBuilder.cmdAuthInfo.Insecure.Value, t)
	matchBoolArg(args.matchApiVersion, castBuilder.matchApiVersion, t)

	clientConfig, err := clientBuilder.Config()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(args.server, clientConfig.Host, t)
	matchStringArg(args.apiVersion, clientConfig.Version, t)
	matchStringArg("delta", clientConfig.CertFile, t)
	matchStringArg("echo", clientConfig.KeyFile, t)
	matchStringArg("charlie", clientConfig.CAFile, t)
	matchStringArg("foxtrot", clientConfig.BearerToken, t)
	matchStringArg("alfa-user", clientConfig.Username, t)
	matchStringArg("bravo-password", clientConfig.Password, t)
	matchBoolArg(args.insecure, clientConfig.Insecure, t)
}

func TestAuthFileOverridden(t *testing.T) {
	flags := pflag.NewFlagSet("test-flags", pflag.ContinueOnError)
	clientBuilder := NewBuilder(nil)
	clientBuilder.BindFlags(flags)
	authFileContents := fmt.Sprintf(`{"user": "alfa-user", "password": "bravo-password", "cAFile": "charlie", "certFile": "delta", "keyFile": "echo", "bearerToken": "foxtrot"}`)
	authFile := writeTempAuthFile(authFileContents, t)

	args := argValues{"https://localhost:8080", "v1beta1", authFile, "cert-file", "key-file", "ca-file", "bearer-token", true, true}
	flags.Parse(strings.Split(args.toArguments(), " "))

	castBuilder, ok := clientBuilder.(*builder)
	if !ok {
		t.Errorf("Got unexpected cast result: %#v", castBuilder)
	}

	matchStringArg(args.server, castBuilder.apiserver, t)
	matchStringArg(args.apiVersion, castBuilder.apiVersion, t)
	matchStringArg(args.authPath, castBuilder.authPath, t)
	matchStringArg(args.certFile, castBuilder.cmdAuthInfo.CertFile.Value, t)
	matchStringArg(args.keyFile, castBuilder.cmdAuthInfo.KeyFile.Value, t)
	matchStringArg(args.caFile, castBuilder.cmdAuthInfo.CAFile.Value, t)
	matchStringArg(args.bearerToken, castBuilder.cmdAuthInfo.BearerToken.Value, t)
	matchBoolArg(args.insecure, castBuilder.cmdAuthInfo.Insecure.Value, t)
	matchBoolArg(args.matchApiVersion, castBuilder.matchApiVersion, t)

	clientConfig, err := clientBuilder.Config()
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
	}

	matchStringArg(args.server, clientConfig.Host, t)
	matchStringArg(args.apiVersion, clientConfig.Version, t)
	matchStringArg(args.certFile, clientConfig.CertFile, t)
	matchStringArg(args.keyFile, clientConfig.KeyFile, t)
	matchStringArg(args.caFile, clientConfig.CAFile, t)
	matchStringArg(args.bearerToken, clientConfig.BearerToken, t)
	matchStringArg("alfa-user", clientConfig.Username, t)
	matchStringArg("bravo-password", clientConfig.Password, t)
	matchBoolArg(args.insecure, clientConfig.Insecure, t)
}

func TestUseDefaultArgumentsOnly(t *testing.T) {
	flags := pflag.NewFlagSet("test-flags", pflag.ContinueOnError)
	clientBuilder := NewBuilder(nil)
	clientBuilder.BindFlags(flags)

	flags.Parse(strings.Split("", " "))

	castBuilder, ok := clientBuilder.(*builder)
	if !ok {
		t.Errorf("Got unexpected cast result: %#v", castBuilder)
	}

	matchStringArg("", castBuilder.apiserver, t)
	matchStringArg(latest.Version, castBuilder.apiVersion, t)
	matchStringArg(os.Getenv("HOME")+"/.kubernetes_auth", castBuilder.authPath, t)
	matchStringArg("", castBuilder.cmdAuthInfo.CertFile.Value, t)
	matchStringArg("", castBuilder.cmdAuthInfo.KeyFile.Value, t)
	matchStringArg("", castBuilder.cmdAuthInfo.CAFile.Value, t)
	matchStringArg("", castBuilder.cmdAuthInfo.BearerToken.Value, t)
	matchBoolArg(false, castBuilder.matchApiVersion, t)
}

func TestLoadClientAuthInfoOrPrompt(t *testing.T) {
	loadAuthInfoTests := []struct {
		authData string
		authInfo *clientauth.Info
		r        io.Reader
	}{
		{
			`{"user": "user", "password": "pass"}`,
			&clientauth.Info{User: "user", Password: "pass"},
			nil,
		},
		{
			"", nil, nil,
		},
		{
			"missing",
			&clientauth.Info{User: "user", Password: "pass"},
			bytes.NewBufferString("user\npass"),
		},
	}
	for _, loadAuthInfoTest := range loadAuthInfoTests {
		tt := loadAuthInfoTest
		aifile, err := ioutil.TempFile("", "testAuthInfo")
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if tt.authData != "missing" {
			defer os.Remove(aifile.Name())
			defer aifile.Close()
			_, err = aifile.WriteString(tt.authData)
			if err != nil {
				t.Errorf("Unexpected error: %v", err)
			}
		} else {
			aifile.Close()
			os.Remove(aifile.Name())
		}
		prompter := NewPromptingAuthLoader(tt.r)
		authInfo, err := prompter.LoadAuth(aifile.Name())
		if len(tt.authData) == 0 && tt.authData != "missing" {
			if err == nil {
				t.Error("LoadAuth didn't fail on empty file")
			}
			continue
		}
		if err != nil {
			t.Errorf("Unexpected error: %v", err)
		}
		if !reflect.DeepEqual(authInfo, tt.authInfo) {
			t.Errorf("Expected %#v, got %#v", tt.authInfo, authInfo)
		}
	}
}

func TestOverride(t *testing.T) {
	b := NewBuilder(nil)
	cfg, err := b.Config()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Version != "" {
		t.Errorf("unexpected default config version")
	}

	newCfg, err := b.Override(func(cfg *client.Config) {
		if cfg.Version != "" {
			t.Errorf("unexpected default config version")
		}
		cfg.Version = "test"
	}).Config()

	if newCfg.Version != "test" {
		t.Errorf("unexpected override config version")
	}

	if cfg.Version != "" {
		t.Errorf("original object should not change")
	}

	cfg, err = b.Config()
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if cfg.Version != "" {
		t.Errorf("override should not be persistent")
	}
}

func matchStringArg(expected, got string, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func matchBoolArg(expected, got bool, t *testing.T) {
	if expected != got {
		t.Errorf("Expected %v, got %v", expected, got)
	}
}

func writeTempAuthFile(contents string, t *testing.T) string {
	file, err := ioutil.TempFile("", "testAuthInfo")
	if err != nil {
		t.Errorf("Failed to write config file.  Test cannot continue due to: %v", err)
		return ""
	}
	_, err = file.WriteString(contents)
	if err != nil {
		t.Errorf("Unexpected error: %v", err)
		return ""
	}
	file.Close()

	return file.Name()
}

type argValues struct {
	server          string
	apiVersion      string
	authPath        string
	certFile        string
	keyFile         string
	caFile          string
	bearerToken     string
	insecure        bool
	matchApiVersion bool
}

func (a *argValues) toArguments() string {
	args := ""
	if len(a.server) > 0 {
		args += "--" + FlagApiServer + "=" + a.server + " "
	}
	if len(a.apiVersion) > 0 {
		args += "--" + FlagApiVersion + "=" + a.apiVersion + " "
	}
	if len(a.authPath) > 0 {
		args += "--" + FlagAuthPath + "=" + a.authPath + " "
	}
	if len(a.certFile) > 0 {
		args += "--" + FlagCertFile + "=" + a.certFile + " "
	}
	if len(a.keyFile) > 0 {
		args += "--" + FlagKeyFile + "=" + a.keyFile + " "
	}
	if len(a.caFile) > 0 {
		args += "--" + FlagCAFile + "=" + a.caFile + " "
	}
	if len(a.bearerToken) > 0 {
		args += "--" + FlagBearerToken + "=" + a.bearerToken + " "
	}
	args += "--" + FlagInsecure + "=" + fmt.Sprintf("%v", a.insecure) + " "
	args += "--" + FlagMatchApiVersion + "=" + fmt.Sprintf("%v", a.matchApiVersion) + " "

	return args
}
