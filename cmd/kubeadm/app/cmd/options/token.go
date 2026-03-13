/*
Copyright 2018 The Kubernetes Authors.

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

package options

import (
	"fmt"
	"strings"

	"github.com/spf13/pflag"

	bootstrapapi "k8s.io/cluster-bootstrap/token/api"

	bootstraptokenv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/bootstraptoken/v1"
	kubeadmapiv1 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1beta4"
)

// NewBootstrapTokenOptions creates a new BootstrapTokenOptions object with the default values
func NewBootstrapTokenOptions() *BootstrapTokenOptions {
	bto := &BootstrapTokenOptions{&bootstraptokenv1.BootstrapToken{}, ""}
	bootstraptokenv1.SetDefaults_BootstrapToken(bto.BootstrapToken)
	return bto
}

// BootstrapTokenOptions is a wrapper struct for adding bootstrap token-related flags to a FlagSet
// and applying the parsed flags to a InitConfiguration object later at runtime
// TODO: In the future, we might want to group the flags in a better way than adding them all individually like this
type BootstrapTokenOptions struct {
	*bootstraptokenv1.BootstrapToken
	TokenStr string `datapolicy:"token"`
}

// AddTokenFlag adds the --token flag to the given flagset
func (bto *BootstrapTokenOptions) AddTokenFlag(fs *pflag.FlagSet) {
	fs.StringVar(
		&bto.TokenStr, TokenStr, "",
		"The token to use for establishing bidirectional trust between nodes and control-plane nodes. The format is [a-z0-9]{6}\\.[a-z0-9]{16} - e.g. abcdef.0123456789abcdef",
	)
}

// AddTTLFlag adds the --token-ttl flag to the given flagset
func (bto *BootstrapTokenOptions) AddTTLFlag(fs *pflag.FlagSet) {
	bto.AddTTLFlagWithName(fs, TokenTTL)
}

// AddTTLFlagWithName adds the --token-ttl flag with a custom flag name given flagset
func (bto *BootstrapTokenOptions) AddTTLFlagWithName(fs *pflag.FlagSet, flagName string) {
	fs.DurationVar(
		&bto.TTL.Duration, flagName, bto.TTL.Duration,
		"The duration before the token is automatically deleted (e.g. 1s, 2m, 3h). If set to '0', the token will never expire",
	)
}

// AddUsagesFlag adds the --usages flag to the given flagset
func (bto *BootstrapTokenOptions) AddUsagesFlag(fs *pflag.FlagSet) {
	fs.StringSliceVar(
		&bto.Usages, TokenUsages, bto.Usages,
		fmt.Sprintf("Describes the ways in which this token can be used. You can pass --usages multiple times or provide a comma separated list of options. Valid options: [%s]", strings.Join(bootstraptokenv1.DefaultTokenUsages, ",")),
	)
}

// AddGroupsFlag adds the --groups flag to the given flagset
func (bto *BootstrapTokenOptions) AddGroupsFlag(fs *pflag.FlagSet) {
	fs.StringSliceVar(
		&bto.Groups, TokenGroups, bto.Groups,
		fmt.Sprintf("Extra groups that this token will authenticate as when used for authentication. Must match %q", bootstrapapi.BootstrapGroupPattern),
	)
}

// AddDescriptionFlag adds the --description flag to the given flagset
func (bto *BootstrapTokenOptions) AddDescriptionFlag(fs *pflag.FlagSet) {
	fs.StringVar(
		&bto.Description, TokenDescription, bto.Description,
		"A human friendly description of how this token is used.",
	)
}

// ApplyTo applies the values set internally in the BootstrapTokenOptions object to a InitConfiguration object at runtime
// If --token was specified in the CLI (as a string), it's parsed and validated before it's added to the BootstrapToken object.
func (bto *BootstrapTokenOptions) ApplyTo(cfg *kubeadmapiv1.InitConfiguration) error {
	if len(bto.TokenStr) > 0 {
		var err error
		bto.Token, err = bootstraptokenv1.NewBootstrapTokenString(bto.TokenStr)
		if err != nil {
			return err
		}
	}

	// Set the token specified by the flags as the first and only token to create in case --config is not specified
	cfg.BootstrapTokens = []bootstraptokenv1.BootstrapToken{*bto.BootstrapToken}
	return nil
}
