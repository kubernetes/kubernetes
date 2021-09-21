package types

type ReplacementField struct {
	Replacement `json:",inline,omitempty" yaml:",inline,omitempty"`
	Path        string `json:"path,omitempty" yaml:"path,omitempty"`
}
