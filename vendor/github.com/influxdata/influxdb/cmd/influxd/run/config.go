package run

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"log"
	"os"
	"os/user"
	"path/filepath"
	"reflect"
	"regexp"
	"strconv"
	"strings"
	"time"

	"github.com/BurntSushi/toml"
	"github.com/influxdata/influxdb/coordinator"
	"github.com/influxdata/influxdb/monitor"
	"github.com/influxdata/influxdb/services/admin"
	"github.com/influxdata/influxdb/services/collectd"
	"github.com/influxdata/influxdb/services/continuous_querier"
	"github.com/influxdata/influxdb/services/graphite"
	"github.com/influxdata/influxdb/services/httpd"
	"github.com/influxdata/influxdb/services/meta"
	"github.com/influxdata/influxdb/services/opentsdb"
	"github.com/influxdata/influxdb/services/precreator"
	"github.com/influxdata/influxdb/services/retention"
	"github.com/influxdata/influxdb/services/subscriber"
	"github.com/influxdata/influxdb/services/udp"
	"github.com/influxdata/influxdb/tsdb"
)

const (
	// DefaultBindAddress is the default address for various RPC services.
	DefaultBindAddress = ":8088"
)

// Config represents the configuration format for the influxd binary.
type Config struct {
	Meta        *meta.Config       `toml:"meta"`
	Data        tsdb.Config        `toml:"data"`
	Coordinator coordinator.Config `toml:"coordinator"`
	Retention   retention.Config   `toml:"retention"`
	Precreator  precreator.Config  `toml:"shard-precreation"`

	Admin          admin.Config      `toml:"admin"`
	Monitor        monitor.Config    `toml:"monitor"`
	Subscriber     subscriber.Config `toml:"subscriber"`
	HTTPD          httpd.Config      `toml:"http"`
	GraphiteInputs []graphite.Config `toml:"graphite"`
	CollectdInputs []collectd.Config `toml:"collectd"`
	OpenTSDBInputs []opentsdb.Config `toml:"opentsdb"`
	UDPInputs      []udp.Config      `toml:"udp"`

	ContinuousQuery continuous_querier.Config `toml:"continuous_queries"`

	// Server reporting
	ReportingDisabled bool `toml:"reporting-disabled"`

	// BindAddress is the address that all TCP services use (Raft, Snapshot, Cluster, etc.)
	BindAddress string `toml:"bind-address"`
}

// NewConfig returns an instance of Config with reasonable defaults.
func NewConfig() *Config {
	c := &Config{}
	c.Meta = meta.NewConfig()
	c.Data = tsdb.NewConfig()
	c.Coordinator = coordinator.NewConfig()
	c.Precreator = precreator.NewConfig()

	c.Admin = admin.NewConfig()
	c.Monitor = monitor.NewConfig()
	c.Subscriber = subscriber.NewConfig()
	c.HTTPD = httpd.NewConfig()

	c.GraphiteInputs = []graphite.Config{graphite.NewConfig()}
	c.CollectdInputs = []collectd.Config{collectd.NewConfig()}
	c.OpenTSDBInputs = []opentsdb.Config{opentsdb.NewConfig()}
	c.UDPInputs = []udp.Config{udp.NewConfig()}

	c.ContinuousQuery = continuous_querier.NewConfig()
	c.Retention = retention.NewConfig()
	c.BindAddress = DefaultBindAddress

	return c
}

// NewDemoConfig returns the config that runs when no config is specified.
func NewDemoConfig() (*Config, error) {
	c := NewConfig()

	var homeDir string
	// By default, store meta and data files in current users home directory
	u, err := user.Current()
	if err == nil {
		homeDir = u.HomeDir
	} else if os.Getenv("HOME") != "" {
		homeDir = os.Getenv("HOME")
	} else {
		return nil, fmt.Errorf("failed to determine current user for storage")
	}

	c.Meta.Dir = filepath.Join(homeDir, ".influxdb/meta")
	c.Data.Dir = filepath.Join(homeDir, ".influxdb/data")
	c.Data.WALDir = filepath.Join(homeDir, ".influxdb/wal")

	return c, nil
}

// trimBOM trims the Byte-Order-Marks from the beginning of the file.
// this is for Windows compatability only.
// see https://github.com/influxdata/telegraf/issues/1378
func trimBOM(f []byte) []byte {
	return bytes.TrimPrefix(f, []byte("\xef\xbb\xbf"))
}

// FromTomlFile loads the config from a TOML file.
func (c *Config) FromTomlFile(fpath string) error {
	bs, err := ioutil.ReadFile(fpath)
	if err != nil {
		return err
	}
	bs = trimBOM(bs)
	return c.FromToml(string(bs))
}

// FromToml loads the config from TOML.
func (c *Config) FromToml(input string) error {
	// Replace deprecated [cluster] with [coordinator]
	re := regexp.MustCompile(`(?m)^\s*\[cluster\]`)
	input = re.ReplaceAllStringFunc(input, func(in string) string {
		in = strings.TrimSpace(in)
		out := "[coordinator]"
		log.Printf("deprecated config option %s replaced with %s; %s will not be supported in a future release\n", in, out, in)
		return out
	})

	_, err := toml.Decode(input, c)
	return err
}

// Validate returns an error if the config is invalid.
func (c *Config) Validate() error {
	if err := c.Meta.Validate(); err != nil {
		return err
	}

	if err := c.Data.Validate(); err != nil {
		return err
	}

	if err := c.Monitor.Validate(); err != nil {
		return err
	}

	if err := c.Subscriber.Validate(); err != nil {
		return err
	}

	for _, g := range c.GraphiteInputs {
		if err := g.Validate(); err != nil {
			return fmt.Errorf("invalid graphite config: %v", err)
		}
	}

	return nil
}

// ApplyEnvOverrides apply the environment configuration on top of the config.
func (c *Config) ApplyEnvOverrides() error {
	return c.applyEnvOverrides("INFLUXDB", reflect.ValueOf(c))
}

func (c *Config) applyEnvOverrides(prefix string, spec reflect.Value) error {
	// If we have a pointer, dereference it
	s := spec
	if spec.Kind() == reflect.Ptr {
		s = spec.Elem()
	}

	// Make sure we have struct
	if s.Kind() != reflect.Struct {
		return nil
	}

	typeOfSpec := s.Type()
	for i := 0; i < s.NumField(); i++ {
		f := s.Field(i)
		// Get the toml tag to determine what env var name to use
		configName := typeOfSpec.Field(i).Tag.Get("toml")
		// Replace hyphens with underscores to avoid issues with shells
		configName = strings.Replace(configName, "-", "_", -1)
		fieldKey := typeOfSpec.Field(i).Name

		// Skip any fields that we cannot set
		if f.CanSet() || f.Kind() == reflect.Slice {

			// Use the upper-case prefix and toml name for the env var
			key := strings.ToUpper(configName)
			if prefix != "" {
				key = strings.ToUpper(fmt.Sprintf("%s_%s", prefix, configName))
			}
			value := os.Getenv(key)

			// If the type is s slice, apply to each using the index as a suffix
			// e.g. GRAPHITE_0
			if f.Kind() == reflect.Slice || f.Kind() == reflect.Array {
				for i := 0; i < f.Len(); i++ {
					if err := c.applyEnvOverrides(key, f.Index(i)); err != nil {
						return err
					}
					if err := c.applyEnvOverrides(fmt.Sprintf("%s_%d", key, i), f.Index(i)); err != nil {
						return err
					}
				}
				continue
			}

			// If it's a sub-config, recursively apply
			if f.Kind() == reflect.Struct || f.Kind() == reflect.Ptr {
				if err := c.applyEnvOverrides(key, f); err != nil {
					return err
				}
				continue
			}

			// Skip any fields we don't have a value to set
			if value == "" {
				continue
			}

			switch f.Kind() {
			case reflect.String:
				f.SetString(value)
			case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:

				var intValue int64

				// Handle toml.Duration
				if f.Type().Name() == "Duration" {
					dur, err := time.ParseDuration(value)
					if err != nil {
						return fmt.Errorf("failed to apply %v to %v using type %v and value '%v'", key, fieldKey, f.Type().String(), value)
					}
					intValue = dur.Nanoseconds()
				} else {
					var err error
					intValue, err = strconv.ParseInt(value, 0, f.Type().Bits())
					if err != nil {
						return fmt.Errorf("failed to apply %v to %v using type %v and value '%v'", key, fieldKey, f.Type().String(), value)
					}
				}

				f.SetInt(intValue)
			case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
				var intValue uint64
				var err error
				intValue, err = strconv.ParseUint(value, 0, f.Type().Bits())
				if err != nil {
					return fmt.Errorf("failed to apply %v to %v using type %v and value '%v'", key, fieldKey, f.Type().String(), value)
				}

				f.SetUint(intValue)
			case reflect.Bool:
				boolValue, err := strconv.ParseBool(value)
				if err != nil {
					return fmt.Errorf("failed to apply %v to %v using type %v and value '%v'", key, fieldKey, f.Type().String(), value)

				}
				f.SetBool(boolValue)
			case reflect.Float32, reflect.Float64:
				floatValue, err := strconv.ParseFloat(value, f.Type().Bits())
				if err != nil {
					return fmt.Errorf("failed to apply %v to %v using type %v and value '%v'", key, fieldKey, f.Type().String(), value)

				}
				f.SetFloat(floatValue)
			default:
				if err := c.applyEnvOverrides(key, f); err != nil {
					return err
				}
			}
		}
	}
	return nil
}
