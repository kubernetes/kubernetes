package flags

import "github.com/spf13/pflag"

type EnvironmentalFlags struct {
	Platform     string
	Network      string
	NetworkStack string
	Upgrade      string
	Topology     string
	Architecture string
	Installer    string
	Facts        map[string]string
	Version      string
}

func NewEnvironmentalFlags() *EnvironmentalFlags {
	return &EnvironmentalFlags{}
}

func (f *EnvironmentalFlags) BindFlags(fs *pflag.FlagSet) {
	fs.StringVar(&f.Platform,
		"platform",
		"",
		"The hardware or cloud platform (\"aws\", \"gcp\", \"metal\", ...). Since: v1.0")
	fs.StringVar(&f.Network,
		"network",
		"",
		"The network of the target cluster (\"ovn\", \"sdn\"). Since: v1.0")
	fs.StringVar(&f.NetworkStack,
		"network-stack",
		"",
		"The network stack of the target cluster (\"ipv6\", \"ipv4\", \"dual\"). Since: v1.0")
	fs.StringVar(&f.Upgrade,
		"upgrade",
		"",
		"The upgrade that was performed prior to the test run (\"micro\", \"minor\"). Since: v1.0")
	fs.StringVar(&f.Topology,
		"topology",
		"",
		"The target cluster topology (\"ha\", \"microshift\", ...). Since: v1.0")
	fs.StringVar(&f.Architecture,
		"architecture",
		"",
		"The CPU architecture of the target cluster (\"amd64\", \"arm64\"). Since: v1.0")
	fs.StringVar(&f.Installer,
		"installer",
		"",
		"The installer used to create the cluster (\"ipi\", \"upi\", \"assisted\", ...). Since: v1.0")
	fs.StringToStringVar(&f.Facts,
		"fact",
		make(map[string]string),
		"Facts advertised by cluster components. Since: v1.0")
	fs.StringVar(&f.Version,
		"version",
		"",
		"\"major.minor\" version of target cluster. Since: v1.0")
}

func (f *EnvironmentalFlags) IsEmpty() bool {
	return f.Platform == "" &&
		f.Network == "" &&
		f.NetworkStack == "" &&
		f.Upgrade == "" &&
		f.Topology == "" &&
		f.Architecture == "" &&
		f.Installer == "" &&
		len(f.Facts) == 0 &&
		f.Version == ""
}

// EnvironmentFlagVersions holds the "Since" version metadata for each flag.
var EnvironmentFlagVersions = map[string]string{
	"platform":      "v1.0",
	"network":       "v1.0",
	"network-stack": "v1.0",
	"upgrade":       "v1.0",
	"topology":      "v1.0",
	"architecture":  "v1.0",
	"installer":     "v1.0",
	"fact":          "v1.0",
	"version":       "v1.0",
}
