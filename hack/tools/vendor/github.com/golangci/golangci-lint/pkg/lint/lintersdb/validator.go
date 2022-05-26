package lintersdb

import (
	"fmt"
	"strings"

	"github.com/golangci/golangci-lint/pkg/config"
)

type Validator struct {
	m *Manager
}

func NewValidator(m *Manager) *Validator {
	return &Validator{
		m: m,
	}
}

func (v Validator) validateLintersNames(cfg *config.Linters) error {
	allNames := append([]string{}, cfg.Enable...)
	allNames = append(allNames, cfg.Disable...)

	var unknownNames []string

	for _, name := range allNames {
		if v.m.GetLinterConfigs(name) == nil {
			unknownNames = append(unknownNames, name)
		}
	}

	if len(unknownNames) > 0 {
		return fmt.Errorf("unknown linters: '%v', run 'golangci-lint help linters' to see the list of supported linters",
			strings.Join(unknownNames, ","))
	}

	return nil
}

func (v Validator) validatePresets(cfg *config.Linters) error {
	allPresets := v.m.allPresetsSet()
	for _, p := range cfg.Presets {
		if !allPresets[p] {
			return fmt.Errorf("no such preset %q: only next presets exist: (%s)",
				p, strings.Join(v.m.AllPresets(), "|"))
		}
	}

	if len(cfg.Presets) != 0 && cfg.EnableAll {
		return fmt.Errorf("--presets is incompatible with --enable-all")
	}

	return nil
}

func (v Validator) validateAllDisableEnableOptions(cfg *config.Linters) error {
	if cfg.EnableAll && cfg.DisableAll {
		return fmt.Errorf("--enable-all and --disable-all options must not be combined")
	}

	if cfg.DisableAll {
		if len(cfg.Enable) == 0 && len(cfg.Presets) == 0 {
			return fmt.Errorf("all linters were disabled, but no one linter was enabled: must enable at least one")
		}

		if len(cfg.Disable) != 0 {
			return fmt.Errorf("can't combine options --disable-all and --disable %s", cfg.Disable[0])
		}
	}

	if cfg.EnableAll && len(cfg.Enable) != 0 && !cfg.Fast {
		return fmt.Errorf("can't combine options --enable-all and --enable %s", cfg.Enable[0])
	}

	return nil
}

func (v Validator) validateDisabledAndEnabledAtOneMoment(cfg *config.Linters) error {
	enabledLintersSet := map[string]bool{}
	for _, name := range cfg.Enable {
		enabledLintersSet[name] = true
	}

	for _, name := range cfg.Disable {
		if enabledLintersSet[name] {
			return fmt.Errorf("linter %q can't be disabled and enabled at one moment", name)
		}
	}

	return nil
}

func (v Validator) validateEnabledDisabledLintersConfig(cfg *config.Linters) error {
	validators := []func(cfg *config.Linters) error{
		v.validateLintersNames,
		v.validatePresets,
		v.validateAllDisableEnableOptions,
		v.validateDisabledAndEnabledAtOneMoment,
	}
	for _, v := range validators {
		if err := v(cfg); err != nil {
			return err
		}
	}

	return nil
}
