// +build !solaris

package daemon

import (
	"io/ioutil"
	"os"
	"path/filepath"
	"testing"

	containertypes "github.com/docker/docker/api/types/container"
	"github.com/docker/docker/container"
	_ "github.com/docker/docker/pkg/discovery/memory"
	"github.com/docker/docker/pkg/idtools"
	"github.com/docker/docker/pkg/truncindex"
	"github.com/docker/docker/volume"
	volumedrivers "github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
	"github.com/docker/docker/volume/store"
	"github.com/docker/go-connections/nat"
)

//
// https://github.com/docker/docker/issues/8069
//

func TestGetContainer(t *testing.T) {
	c1 := &container.Container{
		ID:   "5a4ff6a163ad4533d22d69a2b8960bf7fafdcba06e72d2febdba229008b0bf57",
		Name: "tender_bardeen",
	}

	c2 := &container.Container{
		ID:   "3cdbd1aa394fd68559fd1441d6eff2ab7c1e6363582c82febfaa8045df3bd8de",
		Name: "drunk_hawking",
	}

	c3 := &container.Container{
		ID:   "3cdbd1aa394fd68559fd1441d6eff2abfafdcba06e72d2febdba229008b0bf57",
		Name: "3cdbd1aa",
	}

	c4 := &container.Container{
		ID:   "75fb0b800922abdbef2d27e60abcdfaf7fb0698b2a96d22d3354da361a6ff4a5",
		Name: "5a4ff6a163ad4533d22d69a2b8960bf7fafdcba06e72d2febdba229008b0bf57",
	}

	c5 := &container.Container{
		ID:   "d22d69a2b8960bf7fafdcba06e72d2febdba960bf7fafdcba06e72d2f9008b060b",
		Name: "d22d69a2b896",
	}

	store := container.NewMemoryStore()
	store.Add(c1.ID, c1)
	store.Add(c2.ID, c2)
	store.Add(c3.ID, c3)
	store.Add(c4.ID, c4)
	store.Add(c5.ID, c5)

	index := truncindex.NewTruncIndex([]string{})
	index.Add(c1.ID)
	index.Add(c2.ID)
	index.Add(c3.ID)
	index.Add(c4.ID)
	index.Add(c5.ID)

	containersReplica, err := container.NewViewDB()
	if err != nil {
		t.Fatalf("could not create ViewDB: %v", err)
	}

	daemon := &Daemon{
		containers:        store,
		containersReplica: containersReplica,
		idIndex:           index,
	}

	daemon.reserveName(c1.ID, c1.Name)
	daemon.reserveName(c2.ID, c2.Name)
	daemon.reserveName(c3.ID, c3.Name)
	daemon.reserveName(c4.ID, c4.Name)
	daemon.reserveName(c5.ID, c5.Name)

	if container, _ := daemon.GetContainer("3cdbd1aa394fd68559fd1441d6eff2ab7c1e6363582c82febfaa8045df3bd8de"); container != c2 {
		t.Fatal("Should explicitly match full container IDs")
	}

	if container, _ := daemon.GetContainer("75fb0b8009"); container != c4 {
		t.Fatal("Should match a partial ID")
	}

	if container, _ := daemon.GetContainer("drunk_hawking"); container != c2 {
		t.Fatal("Should match a full name")
	}

	// c3.Name is a partial match for both c3.ID and c2.ID
	if c, _ := daemon.GetContainer("3cdbd1aa"); c != c3 {
		t.Fatal("Should match a full name even though it collides with another container's ID")
	}

	if container, _ := daemon.GetContainer("d22d69a2b896"); container != c5 {
		t.Fatal("Should match a container where the provided prefix is an exact match to the its name, and is also a prefix for its ID")
	}

	if _, err := daemon.GetContainer("3cdbd1"); err == nil {
		t.Fatal("Should return an error when provided a prefix that partially matches multiple container ID's")
	}

	if _, err := daemon.GetContainer("nothing"); err == nil {
		t.Fatal("Should return an error when provided a prefix that is neither a name or a partial match to an ID")
	}
}

func initDaemonWithVolumeStore(tmp string) (*Daemon, error) {
	var err error
	daemon := &Daemon{
		repository: tmp,
		root:       tmp,
	}
	daemon.volumes, err = store.New(tmp)
	if err != nil {
		return nil, err
	}

	volumesDriver, err := local.New(tmp, idtools.IDPair{UID: 0, GID: 0})
	if err != nil {
		return nil, err
	}
	volumedrivers.Register(volumesDriver, volumesDriver.Name())

	return daemon, nil
}

func TestValidContainerNames(t *testing.T) {
	invalidNames := []string{"-rm", "&sdfsfd", "safd%sd"}
	validNames := []string{"word-word", "word_word", "1weoid"}

	for _, name := range invalidNames {
		if validContainerNamePattern.MatchString(name) {
			t.Fatalf("%q is not a valid container name and was returned as valid.", name)
		}
	}

	for _, name := range validNames {
		if !validContainerNamePattern.MatchString(name) {
			t.Fatalf("%q is a valid container name and was returned as invalid.", name)
		}
	}
}

func TestContainerInitDNS(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-container-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	containerID := "d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e"
	containerPath := filepath.Join(tmp, containerID)
	if err := os.MkdirAll(containerPath, 0755); err != nil {
		t.Fatal(err)
	}

	config := `{"State":{"Running":true,"Paused":false,"Restarting":false,"OOMKilled":false,"Dead":false,"Pid":2464,"ExitCode":0,
"Error":"","StartedAt":"2015-05-26T16:48:53.869308965Z","FinishedAt":"0001-01-01T00:00:00Z"},
"ID":"d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e","Created":"2015-05-26T16:48:53.7987917Z","Path":"top",
"Args":[],"Config":{"Hostname":"d59df5276e7b","Domainname":"","User":"","Memory":0,"MemorySwap":0,"CpuShares":0,"Cpuset":"",
"AttachStdin":false,"AttachStdout":false,"AttachStderr":false,"PortSpecs":null,"ExposedPorts":null,"Tty":true,"OpenStdin":true,
"StdinOnce":false,"Env":null,"Cmd":["top"],"Image":"ubuntu:latest","Volumes":null,"WorkingDir":"","Entrypoint":null,
"NetworkDisabled":false,"MacAddress":"","OnBuild":null,"Labels":{}},"Image":"07f8e8c5e66084bef8f848877857537ffe1c47edd01a93af27e7161672ad0e95",
"NetworkSettings":{"IPAddress":"172.17.0.1","IPPrefixLen":16,"MacAddress":"02:42:ac:11:00:01","LinkLocalIPv6Address":"fe80::42:acff:fe11:1",
"LinkLocalIPv6PrefixLen":64,"GlobalIPv6Address":"","GlobalIPv6PrefixLen":0,"Gateway":"172.17.42.1","IPv6Gateway":"","Bridge":"docker0","Ports":{}},
"ResolvConfPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/resolv.conf",
"HostnamePath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hostname",
"HostsPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hosts",
"LogPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e-json.log",
"Name":"/ubuntu","Driver":"aufs","MountLabel":"","ProcessLabel":"","AppArmorProfile":"","RestartCount":0,
"UpdateDns":false,"Volumes":{},"VolumesRW":{},"AppliedVolumesFrom":null}`

	// Container struct only used to retrieve path to config file
	container := &container.Container{Root: containerPath}
	configPath, err := container.ConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	if err = ioutil.WriteFile(configPath, []byte(config), 0644); err != nil {
		t.Fatal(err)
	}

	hostConfig := `{"Binds":[],"ContainerIDFile":"","Memory":0,"MemorySwap":0,"CpuShares":0,"CpusetCpus":"",
"Privileged":false,"PortBindings":{},"Links":null,"PublishAllPorts":false,"Dns":null,"DnsOptions":null,"DnsSearch":null,"ExtraHosts":null,"VolumesFrom":null,
"Devices":[],"NetworkMode":"bridge","IpcMode":"","PidMode":"","CapAdd":null,"CapDrop":null,"RestartPolicy":{"Name":"no","MaximumRetryCount":0},
"SecurityOpt":null,"ReadonlyRootfs":false,"Ulimits":null,"LogConfig":{"Type":"","Config":null},"CgroupParent":""}`

	hostConfigPath, err := container.HostConfigPath()
	if err != nil {
		t.Fatal(err)
	}
	if err = ioutil.WriteFile(hostConfigPath, []byte(hostConfig), 0644); err != nil {
		t.Fatal(err)
	}

	daemon, err := initDaemonWithVolumeStore(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer volumedrivers.Unregister(volume.DefaultDriverName)

	c, err := daemon.load(containerID)
	if err != nil {
		t.Fatal(err)
	}

	if c.HostConfig.DNS == nil {
		t.Fatal("Expected container DNS to not be nil")
	}

	if c.HostConfig.DNSSearch == nil {
		t.Fatal("Expected container DNSSearch to not be nil")
	}

	if c.HostConfig.DNSOptions == nil {
		t.Fatal("Expected container DNSOptions to not be nil")
	}
}

func newPortNoError(proto, port string) nat.Port {
	p, _ := nat.NewPort(proto, port)
	return p
}

func TestMerge(t *testing.T) {
	volumesImage := make(map[string]struct{})
	volumesImage["/test1"] = struct{}{}
	volumesImage["/test2"] = struct{}{}
	portsImage := make(nat.PortSet)
	portsImage[newPortNoError("tcp", "1111")] = struct{}{}
	portsImage[newPortNoError("tcp", "2222")] = struct{}{}
	configImage := &containertypes.Config{
		ExposedPorts: portsImage,
		Env:          []string{"VAR1=1", "VAR2=2"},
		Volumes:      volumesImage,
	}

	portsUser := make(nat.PortSet)
	portsUser[newPortNoError("tcp", "2222")] = struct{}{}
	portsUser[newPortNoError("tcp", "3333")] = struct{}{}
	volumesUser := make(map[string]struct{})
	volumesUser["/test3"] = struct{}{}
	configUser := &containertypes.Config{
		ExposedPorts: portsUser,
		Env:          []string{"VAR2=3", "VAR3=3"},
		Volumes:      volumesUser,
	}

	if err := merge(configUser, configImage); err != nil {
		t.Error(err)
	}

	if len(configUser.ExposedPorts) != 3 {
		t.Fatalf("Expected 3 ExposedPorts, 1111, 2222 and 3333, found %d", len(configUser.ExposedPorts))
	}
	for portSpecs := range configUser.ExposedPorts {
		if portSpecs.Port() != "1111" && portSpecs.Port() != "2222" && portSpecs.Port() != "3333" {
			t.Fatalf("Expected 1111 or 2222 or 3333, found %s", portSpecs)
		}
	}
	if len(configUser.Env) != 3 {
		t.Fatalf("Expected 3 env var, VAR1=1, VAR2=3 and VAR3=3, found %d", len(configUser.Env))
	}
	for _, env := range configUser.Env {
		if env != "VAR1=1" && env != "VAR2=3" && env != "VAR3=3" {
			t.Fatalf("Expected VAR1=1 or VAR2=3 or VAR3=3, found %s", env)
		}
	}

	if len(configUser.Volumes) != 3 {
		t.Fatalf("Expected 3 volumes, /test1, /test2 and /test3, found %d", len(configUser.Volumes))
	}
	for v := range configUser.Volumes {
		if v != "/test1" && v != "/test2" && v != "/test3" {
			t.Fatalf("Expected /test1 or /test2 or /test3, found %s", v)
		}
	}

	ports, _, err := nat.ParsePortSpecs([]string{"0000"})
	if err != nil {
		t.Error(err)
	}
	configImage2 := &containertypes.Config{
		ExposedPorts: ports,
	}

	if err := merge(configUser, configImage2); err != nil {
		t.Error(err)
	}

	if len(configUser.ExposedPorts) != 4 {
		t.Fatalf("Expected 4 ExposedPorts, 0000, 1111, 2222 and 3333, found %d", len(configUser.ExposedPorts))
	}
	for portSpecs := range configUser.ExposedPorts {
		if portSpecs.Port() != "0" && portSpecs.Port() != "1111" && portSpecs.Port() != "2222" && portSpecs.Port() != "3333" {
			t.Fatalf("Expected %q or %q or %q or %q, found %s", 0, 1111, 2222, 3333, portSpecs)
		}
	}
}
