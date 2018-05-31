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

	bootstrapapi "k8s.io/client-go/tools/bootstrap/token/api"
	kubeadmapiv1alpha2 "k8s.io/kubernetes/cmd/kubeadm/app/apis/kubeadm/v1alpha2"
	kubeadmconstants "k8s.io/kubernetes/cmd/kubeadm/app/constants"
)

func NewBootstrapTokenOptions() *BootstrapTokenOptions {
	bto := &BootstrapTokenOptions{&kubeadmapiv1alpha2.BootstrapToken{}, ""}
	kubeadmapiv1alpha2.SetDefaults_BootstrapToken(bto.BootstrapToken)
	return bto
}

// BootstrapTokenOptions is a wrapper struct for adding bootstrap token-related flags to a FlagSet
// and applying the parsed flags to a MasterConfiguration object later at runtime
// TODO: In the future, we might want to group the flags in a better way than adding them all individually like this
type BootstrapTokenOptions struct {
	*kubeadmapiv1alpha2.BootstrapToken
	TokenStr string
}

func (bto *BootstrapTokenOptions) AddTokenFlag(fs *pflag.FlagSet) {
	fs.StringVar(
		&bto.TokenStr, "token", "",
		"The token to use for establishing bidirectional trust between nodes and masters. The format is [a-z0-9]{6}\\.[a-z0-9]{16} - e.g. abcdef.0123456789abcdef",
	)
}

func (bto *BootstrapTokenOptions) AddTTLFlag(fs *pflag.FlagSet) {
	fs.DurationVar(
		&bto.TTL.Duration, "ttl", bto.TTL.Duration,
		"The duration before the token is automatically deleted (e.g. 1s, 2m, 3h). If set to '0', the token will never expire",
	)
}

func (bto *BootstrapTokenOptions) AddUsagesFlag(fs *pflag.FlagSet) {
	fs.StringSliceVar(
		&bto.Usages, "usages", bto.Usages,
		fmt.Sprintf("Describes the ways in which this token can be used. You can pass --usages multiple times or provide a comma separated list of options. Valid options: [%s]", strings.Join(kubeadmconstants.DefaultTokenUsages, ",")),
	)
}

func (bto *BootstrapTokenOptions) AddGroupsFlag(fs *pflag.FlagSet) {
	fs.StringSliceVar(
		&bto.Groups, "groups", bto.Groups,
		fmt.Sprintf("Extra groups that this token will authenticate as when used for authentication. Must match %q", bootstrapapi.BootstrapGroupPattern),
	)
}

func (bto *BootstrapTokenOptions) AddDescriptionFlag(fs *pflag.FlagSet) {
	fs.StringVar(
		&bto.Description, "description", bto.Description,
		"A human friendly description of how this token is used.",
	)
}

func (bto *BootstrapTokenOptions) ApplyTo(cfg *kubeadmapiv1alpha2.MasterConfiguration) error {
	if len(bto.TokenStr) > 0 {
		var err error
		bto.Token, err = kubeadmapiv1alpha2.NewBootstrapTokenString(bto.TokenStr)
		if err != nil {
			return err
		}
	}

	// Set the token specified by the flags as the first and only token to create in case --config is not specified
	cfg.BootstrapTokens = []kubeadmapiv1alpha2.BootstrapToken{*bto.BootstrapToken}
	return nil
}
