package runconfig

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io/ioutil"
	"testing"
)

func TestNetworkModeTest(t *testing.T) {
	networkModes := map[NetworkMode][]bool{
		// private, bridge, host, container, none, default
		"":                         {true, false, false, false, false, false},
		"something:weird":          {true, false, false, false, false, false},
		"bridge":                   {true, true, false, false, false, false},
		DefaultDaemonNetworkMode(): {true, true, false, false, false, false},
		"host":           {false, false, true, false, false, false},
		"container:name": {false, false, false, true, false, false},
		"none":           {true, false, false, false, true, false},
		"default":        {true, false, false, false, false, true},
	}
	networkModeNames := map[NetworkMode]string{
		"":                         "",
		"something:weird":          "",
		"bridge":                   "bridge",
		DefaultDaemonNetworkMode(): "bridge",
		"host":           "host",
		"container:name": "container",
		"none":           "none",
		"default":        "default",
	}
	for networkMode, state := range networkModes {
		if networkMode.IsPrivate() != state[0] {
			t.Fatalf("NetworkMode.IsPrivate for %v should have been %v but was %v", networkMode, state[0], networkMode.IsPrivate())
		}
		if networkMode.IsBridge() != state[1] {
			t.Fatalf("NetworkMode.IsBridge for %v should have been %v but was %v", networkMode, state[1], networkMode.IsBridge())
		}
		if networkMode.IsHost() != state[2] {
			t.Fatalf("NetworkMode.IsHost for %v should have been %v but was %v", networkMode, state[2], networkMode.IsHost())
		}
		if networkMode.IsContainer() != state[3] {
			t.Fatalf("NetworkMode.IsContainer for %v should have been %v but was %v", networkMode, state[3], networkMode.IsContainer())
		}
		if networkMode.IsNone() != state[4] {
			t.Fatalf("NetworkMode.IsNone for %v should have been %v but was %v", networkMode, state[4], networkMode.IsNone())
		}
		if networkMode.IsDefault() != state[5] {
			t.Fatalf("NetworkMode.IsDefault for %v should have been %v but was %v", networkMode, state[5], networkMode.IsDefault())
		}
		if networkMode.NetworkName() != networkModeNames[networkMode] {
			t.Fatalf("Expected name %v, got %v", networkModeNames[networkMode], networkMode.NetworkName())
		}
	}
}

func TestIpcModeTest(t *testing.T) {
	ipcModes := map[IpcMode][]bool{
		// private, host, container, valid
		"":                         {true, false, false, true},
		"something:weird":          {true, false, false, false},
		":weird":                   {true, false, false, true},
		"host":                     {false, true, false, true},
		"container:name":           {false, false, true, true},
		"container:name:something": {false, false, true, false},
		"container:":               {false, false, true, false},
	}
	for ipcMode, state := range ipcModes {
		if ipcMode.IsPrivate() != state[0] {
			t.Fatalf("IpcMode.IsPrivate for %v should have been %v but was %v", ipcMode, state[0], ipcMode.IsPrivate())
		}
		if ipcMode.IsHost() != state[1] {
			t.Fatalf("IpcMode.IsHost for %v should have been %v but was %v", ipcMode, state[1], ipcMode.IsHost())
		}
		if ipcMode.IsContainer() != state[2] {
			t.Fatalf("IpcMode.IsContainer for %v should have been %v but was %v", ipcMode, state[2], ipcMode.IsContainer())
		}
		if ipcMode.Valid() != state[3] {
			t.Fatalf("IpcMode.Valid for %v should have been %v but was %v", ipcMode, state[3], ipcMode.Valid())
		}
	}
	containerIpcModes := map[IpcMode]string{
		"":                      "",
		"something":             "",
		"something:weird":       "weird",
		"container":             "",
		"container:":            "",
		"container:name":        "name",
		"container:name1:name2": "name1:name2",
	}
	for ipcMode, container := range containerIpcModes {
		if ipcMode.Container() != container {
			t.Fatalf("Expected %v for %v but was %v", container, ipcMode, ipcMode.Container())
		}
	}
}

func TestUTSModeTest(t *testing.T) {
	utsModes := map[UTSMode][]bool{
		// private, host, valid
		"":                {true, false, true},
		"something:weird": {true, false, false},
		"host":            {false, true, true},
		"host:name":       {true, false, true},
	}
	for utsMode, state := range utsModes {
		if utsMode.IsPrivate() != state[0] {
			t.Fatalf("UtsMode.IsPrivate for %v should have been %v but was %v", utsMode, state[0], utsMode.IsPrivate())
		}
		if utsMode.IsHost() != state[1] {
			t.Fatalf("UtsMode.IsHost for %v should have been %v but was %v", utsMode, state[1], utsMode.IsHost())
		}
		if utsMode.Valid() != state[2] {
			t.Fatalf("UtsMode.Valid for %v should have been %v but was %v", utsMode, state[2], utsMode.Valid())
		}
	}
}

func TestPidModeTest(t *testing.T) {
	pidModes := map[PidMode][]bool{
		// private, host, valid
		"":                {true, false, true},
		"something:weird": {true, false, false},
		"host":            {false, true, true},
		"host:name":       {true, false, true},
	}
	for pidMode, state := range pidModes {
		if pidMode.IsPrivate() != state[0] {
			t.Fatalf("PidMode.IsPrivate for %v should have been %v but was %v", pidMode, state[0], pidMode.IsPrivate())
		}
		if pidMode.IsHost() != state[1] {
			t.Fatalf("PidMode.IsHost for %v should have been %v but was %v", pidMode, state[1], pidMode.IsHost())
		}
		if pidMode.Valid() != state[2] {
			t.Fatalf("PidMode.Valid for %v should have been %v but was %v", pidMode, state[2], pidMode.Valid())
		}
	}
}

func TestRestartPolicy(t *testing.T) {
	restartPolicies := map[RestartPolicy][]bool{
		// none, always, failure
		RestartPolicy{}:                {false, false, false},
		RestartPolicy{"something", 0}:  {false, false, false},
		RestartPolicy{"no", 0}:         {true, false, false},
		RestartPolicy{"always", 0}:     {false, true, false},
		RestartPolicy{"on-failure", 0}: {false, false, true},
	}
	for restartPolicy, state := range restartPolicies {
		if restartPolicy.IsNone() != state[0] {
			t.Fatalf("RestartPolicy.IsNone for %v should have been %v but was %v", restartPolicy, state[0], restartPolicy.IsNone())
		}
		if restartPolicy.IsAlways() != state[1] {
			t.Fatalf("RestartPolicy.IsAlways for %v should have been %v but was %v", restartPolicy, state[1], restartPolicy.IsAlways())
		}
		if restartPolicy.IsOnFailure() != state[2] {
			t.Fatalf("RestartPolicy.IsOnFailure for %v should have been %v but was %v", restartPolicy, state[2], restartPolicy.IsOnFailure())
		}
	}
}

func TestLxcConfigMarshalJSON(t *testing.T) {
	lxcConfigs := map[*LxcConfig]string{
		nil:          "",
		&LxcConfig{}: "null",
		&LxcConfig{
			[]KeyValuePair{{"key1", "value1"}},
		}: `[{"Key":"key1","Value":"value1"}]`,
	}

	for lxcconfig, expected := range lxcConfigs {
		data, err := lxcconfig.MarshalJSON()
		if err != nil {
			t.Fatal(err)
		}
		if string(data) != expected {
			t.Fatalf("Expected %v, got %v", expected, string(data))
		}
	}
}

func TestLxcConfigUnmarshalJSON(t *testing.T) {
	keyvaluePairs := map[string][]KeyValuePair{
		"":   {{"key1", "value1"}},
		"[]": {},
		`[{"Key":"key2","Value":"value2"}]`: {{"key2", "value2"}},
	}
	for json, expectedParts := range keyvaluePairs {
		lxcConfig := &LxcConfig{
			[]KeyValuePair{{"key1", "value1"}},
		}
		if err := lxcConfig.UnmarshalJSON([]byte(json)); err != nil {
			t.Fatal(err)
		}

		actualParts := lxcConfig.Slice()
		if len(actualParts) != len(expectedParts) {
			t.Fatalf("Expected %v keyvaluePairs, got %v (%v)", len(expectedParts), len(actualParts), expectedParts)
		}
		for index, part := range actualParts {
			if part != expectedParts[index] {
				t.Fatalf("Expected %v, got %v", expectedParts, actualParts)
				break
			}
		}
	}
}

func TestMergeConfigs(t *testing.T) {
	expectedHostname := "hostname"
	expectedContainerIDFile := "containerIdFile"
	config := &Config{
		Hostname: expectedHostname,
	}
	hostConfig := &HostConfig{
		ContainerIDFile: expectedContainerIDFile,
	}
	containerConfigWrapper := MergeConfigs(config, hostConfig)
	if containerConfigWrapper.Config.Hostname != expectedHostname {
		t.Fatalf("containerConfigWrapper config hostname expected %v got %v", expectedHostname, containerConfigWrapper.Config.Hostname)
	}
	if containerConfigWrapper.InnerHostConfig.ContainerIDFile != expectedContainerIDFile {
		t.Fatalf("containerConfigWrapper hostconfig containerIdfile expected %v got %v", expectedContainerIDFile, containerConfigWrapper.InnerHostConfig.ContainerIDFile)
	}
	if containerConfigWrapper.Cpuset != "" {
		t.Fatalf("Expected empty Cpuset, got %v", containerConfigWrapper.Cpuset)
	}
}

func TestDecodeHostConfig(t *testing.T) {
	fixtures := []struct {
		file string
	}{
		{"fixtures/container_hostconfig_1_14.json"},
		{"fixtures/container_hostconfig_1_19.json"},
	}

	for _, f := range fixtures {
		b, err := ioutil.ReadFile(f.file)
		if err != nil {
			t.Fatal(err)
		}

		c, err := DecodeHostConfig(bytes.NewReader(b))
		if err != nil {
			t.Fatal(fmt.Errorf("Error parsing %s: %v", f, err))
		}

		if c.Privileged != false {
			t.Fatalf("Expected privileged false, found %v\n", c.Privileged)
		}

		if l := len(c.Binds); l != 1 {
			t.Fatalf("Expected 1 bind, found %d\n", l)
		}

		if c.CapAdd.Len() != 1 && c.CapAdd.Slice()[0] != "NET_ADMIN" {
			t.Fatalf("Expected CapAdd NET_ADMIN, got %v", c.CapAdd)
		}

		if c.CapDrop.Len() != 1 && c.CapDrop.Slice()[0] != "NET_ADMIN" {
			t.Fatalf("Expected CapDrop MKNOD, got %v", c.CapDrop)
		}
	}
}

func TestCapListUnmarshalSliceAndString(t *testing.T) {
	var cl *CapList
	cap0, err := json.Marshal([]string{"CAP_SOMETHING"})
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(cap0, &cl); err != nil {
		t.Fatal(err)
	}

	slice := cl.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "CAP_SOMETHING" {
		t.Fatalf("expected `CAP_SOMETHING`, got: %q", slice[0])
	}

	cap1, err := json.Marshal("CAP_SOMETHING")
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(cap1, &cl); err != nil {
		t.Fatal(err)
	}

	slice = cl.Slice()
	if len(slice) != 1 {
		t.Fatalf("expected 1 element after unmarshal: %q", slice)
	}

	if slice[0] != "CAP_SOMETHING" {
		t.Fatalf("expected `CAP_SOMETHING`, got: %q", slice[0])
	}
}
