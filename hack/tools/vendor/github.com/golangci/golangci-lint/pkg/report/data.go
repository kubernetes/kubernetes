package report

type Warning struct {
	Tag  string `json:",omitempty"`
	Text string
}

type LinterData struct {
	Name             string
	Enabled          bool `json:",omitempty"`
	EnabledByDefault bool `json:",omitempty"`
}

type Data struct {
	Warnings []Warning    `json:",omitempty"`
	Linters  []LinterData `json:",omitempty"`
	Error    string       `json:",omitempty"`
}

func (d *Data) AddLinter(name string, enabled, enabledByDefault bool) {
	d.Linters = append(d.Linters, LinterData{
		Name:             name,
		Enabled:          enabled,
		EnabledByDefault: enabledByDefault,
	})
}
