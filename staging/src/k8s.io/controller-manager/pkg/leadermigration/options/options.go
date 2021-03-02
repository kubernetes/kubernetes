/*
Copyright 2021 The Kubernetes Authors.

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

	"github.com/spf13/pflag"
	"k8s.io/controller-manager/config"
	"k8s.io/controller-manager/pkg/leadermigration"
	migrationconfig "k8s.io/controller-manager/pkg/leadermigration/config"
)

// LeaderMigrationOptions is the set of options for Leader Migration,
// which is given to the controller manager through flags
type LeaderMigrationOptions struct {
	// Enabled indicates whether leader migration is enabled through the --enabled-leader-migration flag.
	Enabled bool

	// ControllerMigrationConfig is the path to the file of LeaderMigrationConfiguration type.
	// It can be set with --leader-migration-config flag
	// If the path is "" (default vaule), the default vaule will be used.
	ControllerMigrationConfig string
}

// DefaultLeaderMigrationOptions returns a LeaderMigrationOptions with default values.
func DefaultLeaderMigrationOptions() *LeaderMigrationOptions {
	return &LeaderMigrationOptions{
		Enabled:                   false,
		ControllerMigrationConfig: "",
	}
}

// AddFlags adds all flags related to leader migration to given flag set.
func (o *LeaderMigrationOptions) AddFlags(fs *pflag.FlagSet) {
	if o == nil {
		return
	}
	fs.BoolVar(&o.Enabled, "enable-leader-migration", false, "Whether to enable controller leader migration.")
	fs.StringVar(&o.ControllerMigrationConfig, "leader-migration-config", "",
		"Path to the config file for controller leader migration, "+
			"or empty to use the value that reflects default configuration of the controller manager. "+
			"The config file should be of type LeaderMigrationConfiguration, group controllermanager.config.k8s.io, version v1alpha1.")
}

// ApplyTo applies the options of leader migration to generic configuration.
func (o *LeaderMigrationOptions) ApplyTo(cfg *config.GenericControllerManagerConfiguration) error {
	if o == nil {
		// an nil LeaderMigrationOptions indicates that default options should be used
		// in which case leader migration will be disabled
		cfg.LeaderMigrationEnabled = false
		return nil
	}
	if o.Enabled && !leadermigration.FeatureEnabled() {
		return fmt.Errorf("Leader Migration is not enabled through feature gate")
	}
	cfg.LeaderMigrationEnabled = o.Enabled
	if !cfg.LeaderMigrationEnabled {
		return nil
	}
	if o.ControllerMigrationConfig == "" {
		return fmt.Errorf("--leader-migration-config is required")
	}
	leaderMigrationConfig, err := migrationconfig.ReadLeaderMigrationConfiguration(o.ControllerMigrationConfig)
	if err != nil {
		return err
	}
	errs := migrationconfig.ValidateLeaderMigrationConfiguration(leaderMigrationConfig)
	if len(errs) != 0 {
		return fmt.Errorf("failed to parse leader migration configuration: %v", errs)
	}
	cfg.LeaderMigration = *leaderMigrationConfig
	return nil
}
