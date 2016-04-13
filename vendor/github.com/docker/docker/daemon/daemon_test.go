package daemon

import (
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"path/filepath"
	"testing"

	"github.com/docker/docker/pkg/graphdb"
	"github.com/docker/docker/pkg/stringid"
	"github.com/docker/docker/pkg/truncindex"
	"github.com/docker/docker/volume"
	"github.com/docker/docker/volume/drivers"
	"github.com/docker/docker/volume/local"
)

//
// https://github.com/docker/docker/issues/8069
//

func TestGet(t *testing.T) {
	c1 := &Container{
		CommonContainer: CommonContainer{
			ID:   "5a4ff6a163ad4533d22d69a2b8960bf7fafdcba06e72d2febdba229008b0bf57",
			Name: "tender_bardeen",
		},
	}

	c2 := &Container{
		CommonContainer: CommonContainer{
			ID:   "3cdbd1aa394fd68559fd1441d6eff2ab7c1e6363582c82febfaa8045df3bd8de",
			Name: "drunk_hawking",
		},
	}

	c3 := &Container{
		CommonContainer: CommonContainer{
			ID:   "3cdbd1aa394fd68559fd1441d6eff2abfafdcba06e72d2febdba229008b0bf57",
			Name: "3cdbd1aa",
		},
	}

	c4 := &Container{
		CommonContainer: CommonContainer{
			ID:   "75fb0b800922abdbef2d27e60abcdfaf7fb0698b2a96d22d3354da361a6ff4a5",
			Name: "5a4ff6a163ad4533d22d69a2b8960bf7fafdcba06e72d2febdba229008b0bf57",
		},
	}

	c5 := &Container{
		CommonContainer: CommonContainer{
			ID:   "d22d69a2b8960bf7fafdcba06e72d2febdba960bf7fafdcba06e72d2f9008b060b",
			Name: "d22d69a2b896",
		},
	}

	store := &contStore{
		s: map[string]*Container{
			c1.ID: c1,
			c2.ID: c2,
			c3.ID: c3,
			c4.ID: c4,
			c5.ID: c5,
		},
	}

	index := truncindex.NewTruncIndex([]string{})
	index.Add(c1.ID)
	index.Add(c2.ID)
	index.Add(c3.ID)
	index.Add(c4.ID)
	index.Add(c5.ID)

	daemonTestDbPath := path.Join(os.TempDir(), "daemon_test.db")
	graph, err := graphdb.NewSqliteConn(daemonTestDbPath)
	if err != nil {
		t.Fatalf("Failed to create daemon test sqlite database at %s", daemonTestDbPath)
	}
	graph.Set(c1.Name, c1.ID)
	graph.Set(c2.Name, c2.ID)
	graph.Set(c3.Name, c3.ID)
	graph.Set(c4.Name, c4.ID)
	graph.Set(c5.Name, c5.ID)

	daemon := &Daemon{
		containers:     store,
		idIndex:        index,
		containerGraph: graph,
	}

	if container, _ := daemon.Get("3cdbd1aa394fd68559fd1441d6eff2ab7c1e6363582c82febfaa8045df3bd8de"); container != c2 {
		t.Fatal("Should explicitly match full container IDs")
	}

	if container, _ := daemon.Get("75fb0b8009"); container != c4 {
		t.Fatal("Should match a partial ID")
	}

	if container, _ := daemon.Get("drunk_hawking"); container != c2 {
		t.Fatal("Should match a full name")
	}

	// c3.Name is a partial match for both c3.ID and c2.ID
	if c, _ := daemon.Get("3cdbd1aa"); c != c3 {
		t.Fatal("Should match a full name even though it collides with another container's ID")
	}

	if container, _ := daemon.Get("d22d69a2b896"); container != c5 {
		t.Fatal("Should match a container where the provided prefix is an exact match to the it's name, and is also a prefix for it's ID")
	}

	if _, err := daemon.Get("3cdbd1"); err == nil {
		t.Fatal("Should return an error when provided a prefix that partially matches multiple container ID's")
	}

	if _, err := daemon.Get("nothing"); err == nil {
		t.Fatal("Should return an error when provided a prefix that is neither a name or a partial match to an ID")
	}

	os.Remove(daemonTestDbPath)
}

func TestLoadWithVolume(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	containerId := "d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e"
	containerPath := filepath.Join(tmp, containerId)
	if err := os.MkdirAll(containerPath, 0755); err != nil {
		t.Fatal(err)
	}

	hostVolumeId := stringid.GenerateRandomID()
	vfsPath := filepath.Join(tmp, "vfs", "dir", hostVolumeId)
	volumePath := filepath.Join(tmp, "volumes", hostVolumeId)

	if err := os.MkdirAll(vfsPath, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(volumePath, 0755); err != nil {
		t.Fatal(err)
	}

	content := filepath.Join(vfsPath, "helo")
	if err := ioutil.WriteFile(content, []byte("HELO"), 0644); err != nil {
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
"LinkLocalIPv6PrefixLen":64,"GlobalIPv6Address":"","GlobalIPv6PrefixLen":0,"Gateway":"172.17.42.1","IPv6Gateway":"","Bridge":"docker0","PortMapping":null,"Ports":{}},
"ResolvConfPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/resolv.conf",
"HostnamePath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hostname",
"HostsPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hosts",
"LogPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e-json.log",
"Name":"/ubuntu","Driver":"aufs","ExecDriver":"native-0.2","MountLabel":"","ProcessLabel":"","AppArmorProfile":"","RestartCount":0,
"UpdateDns":false,"Volumes":{"/vol1":"%s"},"VolumesRW":{"/vol1":true},"AppliedVolumesFrom":null}`

	cfg := fmt.Sprintf(config, vfsPath)
	if err = ioutil.WriteFile(filepath.Join(containerPath, "config.json"), []byte(cfg), 0644); err != nil {
		t.Fatal(err)
	}

	hostConfig := `{"Binds":[],"ContainerIDFile":"","LxcConf":[],"Memory":0,"MemorySwap":0,"CpuShares":0,"CpusetCpus":"",
"Privileged":false,"PortBindings":{},"Links":null,"PublishAllPorts":false,"Dns":null,"DnsSearch":null,"ExtraHosts":null,"VolumesFrom":null,
"Devices":[],"NetworkMode":"bridge","IpcMode":"","PidMode":"","CapAdd":null,"CapDrop":null,"RestartPolicy":{"Name":"no","MaximumRetryCount":0},
"SecurityOpt":null,"ReadonlyRootfs":false,"Ulimits":null,"LogConfig":{"Type":"","Config":null},"CgroupParent":""}`
	if err = ioutil.WriteFile(filepath.Join(containerPath, "hostconfig.json"), []byte(hostConfig), 0644); err != nil {
		t.Fatal(err)
	}

	daemon, err := initDaemonForVolumesTest(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer volumedrivers.Unregister(volume.DefaultDriverName)

	c, err := daemon.load(containerId)
	if err != nil {
		t.Fatal(err)
	}

	err = daemon.verifyVolumesInfo(c)
	if err != nil {
		t.Fatal(err)
	}

	if len(c.MountPoints) != 1 {
		t.Fatalf("Expected 1 volume mounted, was 0\n")
	}

	m := c.MountPoints["/vol1"]
	if m.Name != hostVolumeId {
		t.Fatalf("Expected mount name to be %s, was %s\n", hostVolumeId, m.Name)
	}

	if m.Destination != "/vol1" {
		t.Fatalf("Expected mount destination /vol1, was %s\n", m.Destination)
	}

	if !m.RW {
		t.Fatalf("Expected mount point to be RW but it was not\n")
	}

	if m.Driver != volume.DefaultDriverName {
		t.Fatalf("Expected mount driver local, was %s\n", m.Driver)
	}

	newVolumeContent := filepath.Join(volumePath, local.VolumeDataPathName, "helo")
	b, err := ioutil.ReadFile(newVolumeContent)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "HELO" {
		t.Fatalf("Expected HELO, was %s\n", string(b))
	}
}

func TestLoadWithBindMount(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	containerId := "d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e"
	containerPath := filepath.Join(tmp, containerId)
	if err = os.MkdirAll(containerPath, 0755); err != nil {
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
"LinkLocalIPv6PrefixLen":64,"GlobalIPv6Address":"","GlobalIPv6PrefixLen":0,"Gateway":"172.17.42.1","IPv6Gateway":"","Bridge":"docker0","PortMapping":null,"Ports":{}},
"ResolvConfPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/resolv.conf",
"HostnamePath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hostname",
"HostsPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hosts",
"LogPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e-json.log",
"Name":"/ubuntu","Driver":"aufs","ExecDriver":"native-0.2","MountLabel":"","ProcessLabel":"","AppArmorProfile":"","RestartCount":0,
"UpdateDns":false,"Volumes":{"/vol1": "/vol1"},"VolumesRW":{"/vol1":true},"AppliedVolumesFrom":null}`

	if err = ioutil.WriteFile(filepath.Join(containerPath, "config.json"), []byte(config), 0644); err != nil {
		t.Fatal(err)
	}

	hostConfig := `{"Binds":["/vol1:/vol1"],"ContainerIDFile":"","LxcConf":[],"Memory":0,"MemorySwap":0,"CpuShares":0,"CpusetCpus":"",
"Privileged":false,"PortBindings":{},"Links":null,"PublishAllPorts":false,"Dns":null,"DnsSearch":null,"ExtraHosts":null,"VolumesFrom":null,
"Devices":[],"NetworkMode":"bridge","IpcMode":"","PidMode":"","CapAdd":null,"CapDrop":null,"RestartPolicy":{"Name":"no","MaximumRetryCount":0},
"SecurityOpt":null,"ReadonlyRootfs":false,"Ulimits":null,"LogConfig":{"Type":"","Config":null},"CgroupParent":""}`
	if err = ioutil.WriteFile(filepath.Join(containerPath, "hostconfig.json"), []byte(hostConfig), 0644); err != nil {
		t.Fatal(err)
	}

	daemon, err := initDaemonForVolumesTest(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer volumedrivers.Unregister(volume.DefaultDriverName)

	c, err := daemon.load(containerId)
	if err != nil {
		t.Fatal(err)
	}

	err = daemon.verifyVolumesInfo(c)
	if err != nil {
		t.Fatal(err)
	}

	if len(c.MountPoints) != 1 {
		t.Fatalf("Expected 1 volume mounted, was 0\n")
	}

	m := c.MountPoints["/vol1"]
	if m.Name != "" {
		t.Fatalf("Expected empty mount name, was %s\n", m.Name)
	}

	if m.Source != "/vol1" {
		t.Fatalf("Expected mount source /vol1, was %s\n", m.Source)
	}

	if m.Destination != "/vol1" {
		t.Fatalf("Expected mount destination /vol1, was %s\n", m.Destination)
	}

	if !m.RW {
		t.Fatalf("Expected mount point to be RW but it was not\n")
	}
}

func TestLoadWithVolume17RC(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	containerId := "d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e"
	containerPath := filepath.Join(tmp, containerId)
	if err := os.MkdirAll(containerPath, 0755); err != nil {
		t.Fatal(err)
	}

	hostVolumeId := "6a3c03fc4a4e588561a543cc3bdd50089e27bd11bbb0e551e19bf735e2514101"
	volumePath := filepath.Join(tmp, "volumes", hostVolumeId)

	if err := os.MkdirAll(volumePath, 0755); err != nil {
		t.Fatal(err)
	}

	content := filepath.Join(volumePath, "helo")
	if err := ioutil.WriteFile(content, []byte("HELO"), 0644); err != nil {
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
"LinkLocalIPv6PrefixLen":64,"GlobalIPv6Address":"","GlobalIPv6PrefixLen":0,"Gateway":"172.17.42.1","IPv6Gateway":"","Bridge":"docker0","PortMapping":null,"Ports":{}},
"ResolvConfPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/resolv.conf",
"HostnamePath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hostname",
"HostsPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hosts",
"LogPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e-json.log",
"Name":"/ubuntu","Driver":"aufs","ExecDriver":"native-0.2","MountLabel":"","ProcessLabel":"","AppArmorProfile":"","RestartCount":0,
"UpdateDns":false,"MountPoints":{"/vol1":{"Name":"6a3c03fc4a4e588561a543cc3bdd50089e27bd11bbb0e551e19bf735e2514101","Destination":"/vol1","Driver":"local","RW":true,"Source":"","Relabel":""}},"AppliedVolumesFrom":null}`

	if err = ioutil.WriteFile(filepath.Join(containerPath, "config.json"), []byte(config), 0644); err != nil {
		t.Fatal(err)
	}

	hostConfig := `{"Binds":[],"ContainerIDFile":"","LxcConf":[],"Memory":0,"MemorySwap":0,"CpuShares":0,"CpusetCpus":"",
"Privileged":false,"PortBindings":{},"Links":null,"PublishAllPorts":false,"Dns":null,"DnsSearch":null,"ExtraHosts":null,"VolumesFrom":null,
"Devices":[],"NetworkMode":"bridge","IpcMode":"","PidMode":"","CapAdd":null,"CapDrop":null,"RestartPolicy":{"Name":"no","MaximumRetryCount":0},
"SecurityOpt":null,"ReadonlyRootfs":false,"Ulimits":null,"LogConfig":{"Type":"","Config":null},"CgroupParent":""}`
	if err = ioutil.WriteFile(filepath.Join(containerPath, "hostconfig.json"), []byte(hostConfig), 0644); err != nil {
		t.Fatal(err)
	}

	daemon, err := initDaemonForVolumesTest(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer volumedrivers.Unregister(volume.DefaultDriverName)

	c, err := daemon.load(containerId)
	if err != nil {
		t.Fatal(err)
	}

	err = daemon.verifyVolumesInfo(c)
	if err != nil {
		t.Fatal(err)
	}

	if len(c.MountPoints) != 1 {
		t.Fatalf("Expected 1 volume mounted, was 0\n")
	}

	m := c.MountPoints["/vol1"]
	if m.Name != hostVolumeId {
		t.Fatalf("Expected mount name to be %s, was %s\n", hostVolumeId, m.Name)
	}

	if m.Destination != "/vol1" {
		t.Fatalf("Expected mount destination /vol1, was %s\n", m.Destination)
	}

	if !m.RW {
		t.Fatalf("Expected mount point to be RW but it was not\n")
	}

	if m.Driver != volume.DefaultDriverName {
		t.Fatalf("Expected mount driver local, was %s\n", m.Driver)
	}

	newVolumeContent := filepath.Join(volumePath, local.VolumeDataPathName, "helo")
	b, err := ioutil.ReadFile(newVolumeContent)
	if err != nil {
		t.Fatal(err)
	}
	if string(b) != "HELO" {
		t.Fatalf("Expected HELO, was %s\n", string(b))
	}
}

func TestRemoveLocalVolumesFollowingSymlinks(t *testing.T) {
	tmp, err := ioutil.TempDir("", "docker-daemon-test-")
	if err != nil {
		t.Fatal(err)
	}
	defer os.RemoveAll(tmp)

	containerId := "d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e"
	containerPath := filepath.Join(tmp, containerId)
	if err := os.MkdirAll(containerPath, 0755); err != nil {
		t.Fatal(err)
	}

	hostVolumeId := stringid.GenerateRandomID()
	vfsPath := filepath.Join(tmp, "vfs", "dir", hostVolumeId)
	volumePath := filepath.Join(tmp, "volumes", hostVolumeId)

	if err := os.MkdirAll(vfsPath, 0755); err != nil {
		t.Fatal(err)
	}
	if err := os.MkdirAll(volumePath, 0755); err != nil {
		t.Fatal(err)
	}

	content := filepath.Join(vfsPath, "helo")
	if err := ioutil.WriteFile(content, []byte("HELO"), 0644); err != nil {
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
"LinkLocalIPv6PrefixLen":64,"GlobalIPv6Address":"","GlobalIPv6PrefixLen":0,"Gateway":"172.17.42.1","IPv6Gateway":"","Bridge":"docker0","PortMapping":null,"Ports":{}},
"ResolvConfPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/resolv.conf",
"HostnamePath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hostname",
"HostsPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/hosts",
"LogPath":"/var/lib/docker/containers/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e/d59df5276e7b219d510fe70565e0404bc06350e0d4b43fe961f22f339980170e-json.log",
"Name":"/ubuntu","Driver":"aufs","ExecDriver":"native-0.2","MountLabel":"","ProcessLabel":"","AppArmorProfile":"","RestartCount":0,
"UpdateDns":false,"Volumes":{"/vol1":"%s"},"VolumesRW":{"/vol1":true},"AppliedVolumesFrom":null}`

	cfg := fmt.Sprintf(config, vfsPath)
	if err = ioutil.WriteFile(filepath.Join(containerPath, "config.json"), []byte(cfg), 0644); err != nil {
		t.Fatal(err)
	}

	hostConfig := `{"Binds":[],"ContainerIDFile":"","LxcConf":[],"Memory":0,"MemorySwap":0,"CpuShares":0,"CpusetCpus":"",
"Privileged":false,"PortBindings":{},"Links":null,"PublishAllPorts":false,"Dns":null,"DnsSearch":null,"ExtraHosts":null,"VolumesFrom":null,
"Devices":[],"NetworkMode":"bridge","IpcMode":"","PidMode":"","CapAdd":null,"CapDrop":null,"RestartPolicy":{"Name":"no","MaximumRetryCount":0},
"SecurityOpt":null,"ReadonlyRootfs":false,"Ulimits":null,"LogConfig":{"Type":"","Config":null},"CgroupParent":""}`
	if err = ioutil.WriteFile(filepath.Join(containerPath, "hostconfig.json"), []byte(hostConfig), 0644); err != nil {
		t.Fatal(err)
	}

	daemon, err := initDaemonForVolumesTest(tmp)
	if err != nil {
		t.Fatal(err)
	}
	defer volumedrivers.Unregister(volume.DefaultDriverName)

	c, err := daemon.load(containerId)
	if err != nil {
		t.Fatal(err)
	}

	err = daemon.verifyVolumesInfo(c)
	if err != nil {
		t.Fatal(err)
	}

	if len(c.MountPoints) != 1 {
		t.Fatalf("Expected 1 volume mounted, was 0\n")
	}

	m := c.MountPoints["/vol1"]
	v, err := createVolume(m.Name, m.Driver)
	if err != nil {
		t.Fatal(err)
	}

	if err := removeVolume(v); err != nil {
		t.Fatal(err)
	}

	fi, err := os.Stat(vfsPath)
	if err == nil || !os.IsNotExist(err) {
		t.Fatalf("Expected vfs path to not exist: %v - %v\n", fi, err)
	}
}

func initDaemonForVolumesTest(tmp string) (*Daemon, error) {
	daemon := &Daemon{
		repository: tmp,
		root:       tmp,
	}

	volumesDriver, err := local.New(tmp)
	if err != nil {
		return nil, err
	}
	volumedrivers.Register(volumesDriver, volumesDriver.Name())

	return daemon, nil
}
