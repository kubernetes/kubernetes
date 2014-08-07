package buildconfigapi

import "github.com/GoogleCloudPlatform/kubernetes/pkg/api"

type BuildConfig struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Type         BuildType `json:"type,omitempty" yaml:"type,omitempty"`
	SourceURI    string    `json:"sourceUri,omitempty" yaml:"sourceUri,omitempty"`
	ImageTag     string    `json:"imageTag,omitempty" yaml:"imageTag,omitempty"`
	BuilderImage string    `json:"builderImage,omitempty" yaml:"builderImage,omitempty"`
	SourceRef    string    `json:"sourceRef,omitempty" yaml:"sourceRef,omitempty"`
}

type BuildType string

// BuildList is a collection of Builds.
type BuildConfigList struct {
	api.JSONBase `json:",inline" yaml:",inline"`
	Items        []BuildConfig `json:"items,omitempty" yaml:"items,omitempty"`
}

func init() {
	api.AddKnownTypes("",
		BuildConfig{},
		BuildConfigList{},
	)

	api.AddKnownTypes("v1beta1",
		BuildConfig{},
		BuildConfigList{},
	)
}
