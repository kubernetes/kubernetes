package types

import (
	"bytes"
	"encoding/json"
	"fmt"
	"regexp"
	"sort"

	"github.com/akutz/goof"
)

// LocalDevicesMap is a map of LocalDevices objects.
type LocalDevicesMap map[string]*LocalDevices

// LocalDevices is a wrapper for a map of volume to device mappings.
type LocalDevices struct {

	// Driver is the name of the StorageExecutor that created the map
	// as well as the name of the StorageDriver for which the map is
	// valid.
	Driver string `json:"driver"`

	// DeviceMap is voluem to device mappings.
	DeviceMap map[string]string `json:"deviceMap,omitempty" yaml:"deviceMap,omitempty"`
}

// String returns the string representation of a LocalDevices object.
func (l *LocalDevices) String() string {
	buf, err := l.MarshalText()
	if err != nil {
		panic(err)
	}
	return string(buf)
}

// MarshalText marshals LocalDevices to a text string that adheres to the
// format `DRIVER=VOLUMEID::DEVICEID[,VOLUMEID::DEVICEID,...]`.
func (l *LocalDevices) MarshalText() ([]byte, error) {

	t := &bytes.Buffer{}
	fmt.Fprintf(t, "%s=", l.Driver)

	keys := []string{}

	for k := range l.DeviceMap {
		keys = append(keys, k)
	}

	sort.Sort(byString(keys))

	for _, k := range keys {
		fmt.Fprintf(t, "%s::%s,", k, l.DeviceMap[k])
	}

	if len(l.DeviceMap) > 0 {
		t.Truncate(t.Len() - 1)
	}

	return t.Bytes(), nil
}

var (
	ldRX         = regexp.MustCompile(`^(.+?)=(\S+::\S+(?::\s*,\s*\S+::\S+)*)?$`)
	commaByteSep = []byte{','}
	colonByteSep = []byte{':', ':'}
)

// UnmarshalText unmarshals the data into a an InstanceID provided the data
// adheres to the format described in the MarshalText function.
func (l *LocalDevices) UnmarshalText(value []byte) error {

	m := ldRX.FindSubmatch(value)
	lm := len(m)

	if lm < 3 {
		return goof.WithField("value", string(value), "invalid LocalDevices")
	}

	l.Driver = string(m[1])
	l.DeviceMap = map[string]string{}

	for _, p := range bytes.Split(m[2], commaByteSep) {
		pp := bytes.Split(p, colonByteSep)
		if len(pp) < 2 {
			continue
		}
		l.DeviceMap[string(pp[0])] = string(pp[1])
	}

	return nil
}

// MarshalJSON marshals the InstanceID to JSON.
func (l *LocalDevices) MarshalJSON() ([]byte, error) {
	return json.Marshal(&struct {
		Driver    string            `json:"driver"`
		DeviceMap map[string]string `json:"deviceMap"`
	}{l.Driver, l.DeviceMap})
}

// UnmarshalJSON marshals the InstanceID to JSON.
func (l *LocalDevices) UnmarshalJSON(data []byte) error {

	ldm := &struct {
		Driver    string            `json:"driver"`
		DeviceMap map[string]string `json:"deviceMap"`
	}{}

	if err := json.Unmarshal(data, ldm); err != nil {
		return err
	}

	l.Driver = ldm.Driver
	l.DeviceMap = ldm.DeviceMap

	return nil
}

// MarshalYAML returns the object to marshal to the YAML representation of the
// LocalDevices.
func (l *LocalDevices) MarshalYAML() (interface{}, error) {
	return &struct {
		Driver    string            `json:"driver" yaml:"driver"`
		DeviceMap map[string]string `json:"deviceMap,omitempty" yaml:"deviceMap,omitempty"`
	}{l.Driver, l.DeviceMap}, nil
}

// byString  implements sort.Interface for []string.
type byString []string

func (a byString) Len() int           { return len(a) }
func (a byString) Swap(i, j int)      { a[i], a[j] = a[j], a[i] }
func (a byString) Less(i, j int) bool { return a[i] < a[j] }
