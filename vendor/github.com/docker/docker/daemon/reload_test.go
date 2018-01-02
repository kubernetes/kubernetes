// +build !solaris

package daemon

import (
	"reflect"
	"sort"
	"testing"
	"time"

	"github.com/docker/docker/daemon/config"
	"github.com/docker/docker/pkg/discovery"
	_ "github.com/docker/docker/pkg/discovery/memory"
	"github.com/docker/docker/registry"
)

func TestDaemonReloadLabels(t *testing.T) {
	daemon := &Daemon{}
	daemon.configStore = &config.Config{
		CommonConfig: config.CommonConfig{
			Labels: []string{"foo:bar"},
		},
	}

	valuesSets := make(map[string]interface{})
	valuesSets["labels"] = "foo:baz"
	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			Labels:    []string{"foo:baz"},
			ValuesSet: valuesSets,
		},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	label := daemon.configStore.Labels[0]
	if label != "foo:baz" {
		t.Fatalf("Expected daemon label `foo:baz`, got %s", label)
	}
}

func TestDaemonReloadAllowNondistributableArtifacts(t *testing.T) {
	daemon := &Daemon{
		configStore: &config.Config{},
	}

	// Initialize daemon with some registries.
	daemon.RegistryService = registry.NewService(registry.ServiceOptions{
		AllowNondistributableArtifacts: []string{
			"127.0.0.0/8",
			"10.10.1.11:5000",
			"10.10.1.22:5000", // This will be removed during reload.
			"docker1.com",
			"docker2.com", // This will be removed during reload.
		},
	})

	registries := []string{
		"127.0.0.0/8",
		"10.10.1.11:5000",
		"10.10.1.33:5000", // This will be added during reload.
		"docker1.com",
		"docker3.com", // This will be added during reload.
	}

	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			ServiceOptions: registry.ServiceOptions{
				AllowNondistributableArtifacts: registries,
			},
			ValuesSet: map[string]interface{}{
				"allow-nondistributable-artifacts": registries,
			},
		},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	actual := []string{}
	serviceConfig := daemon.RegistryService.ServiceConfig()
	for _, value := range serviceConfig.AllowNondistributableArtifactsCIDRs {
		actual = append(actual, value.String())
	}
	for _, value := range serviceConfig.AllowNondistributableArtifactsHostnames {
		actual = append(actual, value)
	}

	sort.Strings(registries)
	sort.Strings(actual)
	if !reflect.DeepEqual(registries, actual) {
		t.Fatalf("expected %v, got %v\n", registries, actual)
	}
}

func TestDaemonReloadMirrors(t *testing.T) {
	daemon := &Daemon{}
	daemon.RegistryService = registry.NewService(registry.ServiceOptions{
		InsecureRegistries: []string{},
		Mirrors: []string{
			"https://mirror.test1.com",
			"https://mirror.test2.com", // this will be removed when reloading
			"https://mirror.test3.com", // this will be removed when reloading
		},
	})

	daemon.configStore = &config.Config{}

	type pair struct {
		valid   bool
		mirrors []string
		after   []string
	}

	loadMirrors := []pair{
		{
			valid:   false,
			mirrors: []string{"10.10.1.11:5000"}, // this mirror is invalid
			after:   []string{},
		},
		{
			valid:   false,
			mirrors: []string{"mirror.test1.com"}, // this mirror is invalid
			after:   []string{},
		},
		{
			valid:   false,
			mirrors: []string{"10.10.1.11:5000", "mirror.test1.com"}, // mirrors are invalid
			after:   []string{},
		},
		{
			valid:   true,
			mirrors: []string{"https://mirror.test1.com", "https://mirror.test4.com"},
			after:   []string{"https://mirror.test1.com/", "https://mirror.test4.com/"},
		},
	}

	for _, value := range loadMirrors {
		valuesSets := make(map[string]interface{})
		valuesSets["registry-mirrors"] = value.mirrors

		newConfig := &config.Config{
			CommonConfig: config.CommonConfig{
				ServiceOptions: registry.ServiceOptions{
					Mirrors: value.mirrors,
				},
				ValuesSet: valuesSets,
			},
		}

		err := daemon.Reload(newConfig)
		if !value.valid && err == nil {
			// mirrors should be invalid, should be a non-nil error
			t.Fatalf("Expected daemon reload error with invalid mirrors: %s, while get nil", value.mirrors)
		}

		if value.valid {
			if err != nil {
				// mirrors should be valid, should be no error
				t.Fatal(err)
			}
			registryService := daemon.RegistryService.ServiceConfig()

			if len(registryService.Mirrors) != len(value.after) {
				t.Fatalf("Expected %d daemon mirrors %s while get %d with %s",
					len(value.after),
					value.after,
					len(registryService.Mirrors),
					registryService.Mirrors)
			}

			dataMap := map[string]struct{}{}

			for _, mirror := range registryService.Mirrors {
				if _, exist := dataMap[mirror]; !exist {
					dataMap[mirror] = struct{}{}
				}
			}

			for _, address := range value.after {
				if _, exist := dataMap[address]; !exist {
					t.Fatalf("Expected %s in daemon mirrors, while get none", address)
				}
			}
		}
	}
}

func TestDaemonReloadInsecureRegistries(t *testing.T) {
	daemon := &Daemon{}
	// initialize daemon with existing insecure registries: "127.0.0.0/8", "10.10.1.11:5000", "10.10.1.22:5000"
	daemon.RegistryService = registry.NewService(registry.ServiceOptions{
		InsecureRegistries: []string{
			"127.0.0.0/8",
			"10.10.1.11:5000",
			"10.10.1.22:5000", // this will be removed when reloading
			"docker1.com",
			"docker2.com", // this will be removed when reloading
		},
	})

	daemon.configStore = &config.Config{}

	insecureRegistries := []string{
		"127.0.0.0/8",     // this will be kept
		"10.10.1.11:5000", // this will be kept
		"10.10.1.33:5000", // this will be newly added
		"docker1.com",     // this will be kept
		"docker3.com",     // this will be newly added
	}

	valuesSets := make(map[string]interface{})
	valuesSets["insecure-registries"] = insecureRegistries

	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			ServiceOptions: registry.ServiceOptions{
				InsecureRegistries: insecureRegistries,
			},
			ValuesSet: valuesSets,
		},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	// After Reload, daemon.RegistryService will be changed which is useful
	// for registry communication in daemon.
	registries := daemon.RegistryService.ServiceConfig()

	// After Reload(), newConfig has come to registries.InsecureRegistryCIDRs and registries.IndexConfigs in daemon.
	// Then collect registries.InsecureRegistryCIDRs in dataMap.
	// When collecting, we need to convert CIDRS into string as a key,
	// while the times of key appears as value.
	dataMap := map[string]int{}
	for _, value := range registries.InsecureRegistryCIDRs {
		if _, ok := dataMap[value.String()]; !ok {
			dataMap[value.String()] = 1
		} else {
			dataMap[value.String()]++
		}
	}

	for _, value := range registries.IndexConfigs {
		if _, ok := dataMap[value.Name]; !ok {
			dataMap[value.Name] = 1
		} else {
			dataMap[value.Name]++
		}
	}

	// Finally compare dataMap with the original insecureRegistries.
	// Each value in insecureRegistries should appear in daemon's insecure registries,
	// and each can only appear exactly ONCE.
	for _, r := range insecureRegistries {
		if value, ok := dataMap[r]; !ok {
			t.Fatalf("Expected daemon insecure registry %s, got none", r)
		} else if value != 1 {
			t.Fatalf("Expected only 1 daemon insecure registry %s, got %d", r, value)
		}
	}

	// assert if "10.10.1.22:5000" is removed when reloading
	if value, ok := dataMap["10.10.1.22:5000"]; ok {
		t.Fatalf("Expected no insecure registry of 10.10.1.22:5000, got %d", value)
	}

	// assert if "docker2.com" is removed when reloading
	if value, ok := dataMap["docker2.com"]; ok {
		t.Fatalf("Expected no insecure registry of docker2.com, got %d", value)
	}
}

func TestDaemonReloadNotAffectOthers(t *testing.T) {
	daemon := &Daemon{}
	daemon.configStore = &config.Config{
		CommonConfig: config.CommonConfig{
			Labels: []string{"foo:bar"},
			Debug:  true,
		},
	}

	valuesSets := make(map[string]interface{})
	valuesSets["labels"] = "foo:baz"
	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			Labels:    []string{"foo:baz"},
			ValuesSet: valuesSets,
		},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	label := daemon.configStore.Labels[0]
	if label != "foo:baz" {
		t.Fatalf("Expected daemon label `foo:baz`, got %s", label)
	}
	debug := daemon.configStore.Debug
	if !debug {
		t.Fatal("Expected debug 'enabled', got 'disabled'")
	}
}

func TestDaemonDiscoveryReload(t *testing.T) {
	daemon := &Daemon{}
	daemon.configStore = &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore:     "memory://127.0.0.1",
			ClusterAdvertise: "127.0.0.1:3333",
		},
	}

	if err := daemon.initDiscovery(daemon.configStore); err != nil {
		t.Fatal(err)
	}

	expected := discovery.Entries{
		&discovery.Entry{Host: "127.0.0.1", Port: "3333"},
	}

	select {
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for discovery")
	case <-daemon.discoveryWatcher.ReadyCh():
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	ch, errCh := daemon.discoveryWatcher.Watch(stopCh)

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("failed to get discovery advertisements in time")
	case e := <-ch:
		if !reflect.DeepEqual(e, expected) {
			t.Fatalf("expected %v, got %v\n", expected, e)
		}
	case e := <-errCh:
		t.Fatal(e)
	}

	valuesSets := make(map[string]interface{})
	valuesSets["cluster-store"] = "memory://127.0.0.1:2222"
	valuesSets["cluster-advertise"] = "127.0.0.1:5555"
	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore:     "memory://127.0.0.1:2222",
			ClusterAdvertise: "127.0.0.1:5555",
			ValuesSet:        valuesSets,
		},
	}

	expected = discovery.Entries{
		&discovery.Entry{Host: "127.0.0.1", Port: "5555"},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	select {
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for discovery")
	case <-daemon.discoveryWatcher.ReadyCh():
	}

	ch, errCh = daemon.discoveryWatcher.Watch(stopCh)

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("failed to get discovery advertisements in time")
	case e := <-ch:
		if !reflect.DeepEqual(e, expected) {
			t.Fatalf("expected %v, got %v\n", expected, e)
		}
	case e := <-errCh:
		t.Fatal(e)
	}
}

func TestDaemonDiscoveryReloadFromEmptyDiscovery(t *testing.T) {
	daemon := &Daemon{}
	daemon.configStore = &config.Config{}

	valuesSet := make(map[string]interface{})
	valuesSet["cluster-store"] = "memory://127.0.0.1:2222"
	valuesSet["cluster-advertise"] = "127.0.0.1:5555"
	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore:     "memory://127.0.0.1:2222",
			ClusterAdvertise: "127.0.0.1:5555",
			ValuesSet:        valuesSet,
		},
	}

	expected := discovery.Entries{
		&discovery.Entry{Host: "127.0.0.1", Port: "5555"},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	select {
	case <-time.After(10 * time.Second):
		t.Fatal("timeout waiting for discovery")
	case <-daemon.discoveryWatcher.ReadyCh():
	}

	stopCh := make(chan struct{})
	defer close(stopCh)
	ch, errCh := daemon.discoveryWatcher.Watch(stopCh)

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("failed to get discovery advertisements in time")
	case e := <-ch:
		if !reflect.DeepEqual(e, expected) {
			t.Fatalf("expected %v, got %v\n", expected, e)
		}
	case e := <-errCh:
		t.Fatal(e)
	}
}

func TestDaemonDiscoveryReloadOnlyClusterAdvertise(t *testing.T) {
	daemon := &Daemon{}
	daemon.configStore = &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterStore: "memory://127.0.0.1",
		},
	}
	valuesSets := make(map[string]interface{})
	valuesSets["cluster-advertise"] = "127.0.0.1:5555"
	newConfig := &config.Config{
		CommonConfig: config.CommonConfig{
			ClusterAdvertise: "127.0.0.1:5555",
			ValuesSet:        valuesSets,
		},
	}
	expected := discovery.Entries{
		&discovery.Entry{Host: "127.0.0.1", Port: "5555"},
	}

	if err := daemon.Reload(newConfig); err != nil {
		t.Fatal(err)
	}

	select {
	case <-daemon.discoveryWatcher.ReadyCh():
	case <-time.After(10 * time.Second):
		t.Fatal("Timeout waiting for discovery")
	}
	stopCh := make(chan struct{})
	defer close(stopCh)
	ch, errCh := daemon.discoveryWatcher.Watch(stopCh)

	select {
	case <-time.After(1 * time.Second):
		t.Fatal("failed to get discovery advertisements in time")
	case e := <-ch:
		if !reflect.DeepEqual(e, expected) {
			t.Fatalf("expected %v, got %v\n", expected, e)
		}
	case e := <-errCh:
		t.Fatal(e)
	}
}
