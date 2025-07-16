package flags

import (
	"reflect"

	"github.com/spf13/pflag"
)

type EnvironmentalFlags struct {
	APIGroups            []string
	Architecture         string
	ExternalConnectivity string
	Facts                map[string]string
	FeatureGates         []string
	Network              string
	NetworkStack         string
	OptionalCapabilities []string
	Platform             string
	Topology             string
	Upgrade              string
	Version              string
}

func NewEnvironmentalFlags() *EnvironmentalFlags {
	return &EnvironmentalFlags{}
}

func (f *EnvironmentalFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringArrayVar(&f.APIGroups,
		"api-group",
		f.APIGroups,
		"The API groups supported by this cluster. Since: v1.1")
	fs.StringVar(&f.Architecture,
		"architecture",
		"",
		"The CPU architecture of the target cluster (\"amd64\", \"arm64\"). Since: v1.0")
	fs.StringVar(&f.ExternalConnectivity,
		"external-connectivity",
		"",
		"The External Connectivity of the target cluster (\"Disconnected\", \"Direct\", \"Proxied\"). Since: v1.0")
	fs.StringArrayVar(&f.FeatureGates,
		"feature-gate",
		f.FeatureGates,
		"The feature gates enabled on this cluster. Since: v1.1")
	fs.StringToStringVar(&f.Facts,
		"fact",
		make(map[string]string),
		"Facts advertised by cluster components. Since: v1.0")
	fs.StringVar(&f.Network,
		"network",
		"",
		"The network of the target cluster (\"ovn\", \"sdn\"). Since: v1.0")
	fs.StringVar(&f.NetworkStack,
		"network-stack",
		"",
		"The network stack of the target cluster (\"ipv6\", \"ipv4\", \"dual\"). Since: v1.0")
	fs.StringSliceVar(&f.OptionalCapabilities,
		"optional-capability",
		[]string{},
		"An Optional Capability of the target cluster. Can be passed multiple times. Since: v1.0")
	fs.StringVar(&f.Platform,
		"platform",
		"",
		"The hardware or cloud platform (\"aws\", \"gcp\", \"metal\", ...). Since: v1.0")
	fs.StringVar(&f.Topology,
		"topology",
		"",
		"The target cluster topology (\"ha\", \"microshift\", ...). Since: v1.0")
	fs.StringVar(&f.Upgrade,
		"upgrade",
		"",
		"The upgrade that was performed prior to the test run (\"micro\", \"minor\"). Since: v1.0")
	fs.StringVar(&f.Version,
		"version",
		"",
		"\"major.minor\" version of target cluster. Since: v1.0")
}

func (f *EnvironmentalFlags) IsEmpty() bool {
	v := reflect.ValueOf(*f)

	for i := 0; i < v.NumField(); i++ {
		field := v.Field(i)

		switch field.Kind() {
		case reflect.Slice, reflect.Map:
			if !field.IsNil() && field.Len() > 0 {
				return false
			}
		default:
			if !reflect.DeepEqual(field.Interface(), reflect.Zero(field.Type()).Interface()) {
				return false
			}
		}
	}

	return true
}

// EnvironmentFlagVersions holds the "Since" version metadata for each flag.
var EnvironmentFlagVersions = map[string]string{
	"api-group":             "v1.1",
	"architecture":          "v1.0",
	"external-connectivity": "v1.0",
	"fact":                  "v1.0",
	"feature-gate":          "v1.1",
	"network":               "v1.0",
	"network-stack":         "v1.0",
	"optional-capability":   "v1.0",
	"platform":              "v1.0",
	"topology":              "v1.0",
	"upgrade":               "v1.0",
	"version":               "v1.0",
}
