package stress

import (
	"flag"
	"fmt"
	"strings"

	"github.com/BurntSushi/toml"
)

// Config is a struct for the Stress test configuration
type Config struct {
	Provision Provision `toml:"provision"`
	Write     Write     `toml:"write"`
	Read      Read      `toml:"read"`
}

// Provision is a struct that contains the configuration
// parameters for all implemented Provisioner's.
type Provision struct {
	Basic BasicProvisioner `toml:"basic"`
}

// Write is a struct that contains the configuration
// parameters for the stress test Writer.
type Write struct {
	PointGenerators PointGenerators `toml:"point_generator"`
	InfluxClients   InfluxClients   `toml:"influx_client"`
}

// PointGenerators is a struct that contains the configuration
// parameters for all implemented PointGenerator's.
type PointGenerators struct {
	Basic *BasicPointGenerator `toml:"basic"`
}

// InfluxClients is a struct that contains the configuration
// parameters for all implemented InfluxClient's.
type InfluxClients struct {
	Basic BasicClient `toml:"basic"`
}

// Read is a struct that contains the configuration
// parameters for the stress test Reader.
type Read struct {
	QueryGenerators QueryGenerators `toml:"query_generator"`
	QueryClients    QueryClients    `toml:"query_client"`
}

// QueryGenerators is a struct that contains the configuration
// parameters for all implemented QueryGenerator's.
type QueryGenerators struct {
	Basic BasicQuery `toml:"basic"`
}

// QueryClients is a struct that contains the configuration
// parameters for all implemented QueryClient's.
type QueryClients struct {
	Basic BasicQueryClient `toml:"basic"`
}

// NewConfig returns a pointer to a Config
func NewConfig(s string) (*Config, error) {
	var c *Config
	var err error

	if s == "" {
		c, err = BasicStress()
	} else {
		c, err = DecodeFile(s)
	}

	return c, err
}

// DecodeFile takes a file path for a toml config file
// and returns a pointer to a Config Struct.
func DecodeFile(s string) (*Config, error) {
	t := &Config{}

	// Decode the toml file
	if _, err := toml.DecodeFile(s, t); err != nil {
		return nil, err
	}

	return t, nil
}

// DecodeConfig takes a file path for a toml config file
// and returns a pointer to a Config Struct.
func DecodeConfig(s string) (*Config, error) {
	t := &Config{}

	// Decode the toml file
	if _, err := toml.Decode(s, t); err != nil {
		return nil, err
	}

	return t, nil
}

type outputConfig struct {
	tags            map[string]string
	addr            string
	database        string
	retentionPolicy string
}

func (t *outputConfig) SetParams(addr, db, rp string) {
	t.addr = addr
	t.database = db
	t.retentionPolicy = rp
}

func NewOutputConfig() *outputConfig {
	var o outputConfig
	tags := make(map[string]string)
	o.tags = tags
	database := flag.String("database", "stress", "name of database where the response times will persist")
	retentionPolicy := flag.String("retention-policy", "", "name of the retention policy where the response times will persist")
	address := flag.String("addr", "http://localhost:8086", "IP address and port of database where response times will persist (e.g., localhost:8086)")
	flag.Var(&o, "tags", "A comma seperated list of tags")
	flag.Parse()

	o.SetParams(*address, *database, *retentionPolicy)

	return &o

}

func (t *outputConfig) String() string {
	var s string
	for k, v := range t.tags {
		s += fmt.Sprintf("%v=%v ", k, v)
	}
	return fmt.Sprintf("%v %v %v %v", s, t.database, t.retentionPolicy, t.addr)
}

func (t *outputConfig) Set(value string) error {
	for _, s := range strings.Split(value, ",") {
		tags := strings.Split(s, "=")
		t.tags[tags[0]] = tags[1]
	}
	return nil
}
