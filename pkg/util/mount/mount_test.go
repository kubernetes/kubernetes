/*
Copyright 2015 The Kubernetes Authors All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package mount

import (
	"testing"
)

func TestParseMountConfig(t *testing.T) {
	// default config
	cfg, err := ParseMountConfig("", []string{}, false)
	if err != nil {
		t.Errorf("expected MounterMount, got error %v", err)
	}
	if cfg.Mounter != MounterMount {
		t.Errorf("expected MounterMount, got %v", cfg.Mounter)
	}

	// default config, containerized
	cfg, err = ParseMountConfig("", []string{}, true)
	if err != nil {
		t.Errorf("expected MounterNsenter, got error %v", err)
	}
	if cfg.Mounter != MounterNsenter {
		t.Errorf("expected MounterNsenter, got %v", cfg.Mounter)
	}

	// parsing of volumeMounter argument
	cfg, err = ParseMountConfig("mount", []string{}, false)
	if err != nil {
		t.Errorf("expected MounterMount, got error %v", err)
	}
	if cfg.Mounter != MounterMount {
		t.Errorf("expected MounterMount, got %v", cfg.Mounter)
	}

	cfg, err = ParseMountConfig("nsenter", []string{}, false)
	if err != nil {
		t.Errorf("expected MounterNsenter, got error %v", err)
	}
	if cfg.Mounter != MounterNsenter {
		t.Errorf("expected MounterNsenter, got %v", cfg.Mounter)
	}

	cfg, err = ParseMountConfig("container", []string{"fs:name=value:container"}, false)
	if err != nil {
		t.Errorf("expected MounterContainer, got error %v", err)
	}
	if cfg.Mounter != MounterContainer {
		t.Errorf("expected MounterContainer, got %v", cfg.Mounter)
	}

	cfg, err = ParseMountConfig("xxxinvalidxxx", []string{}, false)
	if cfg != nil {
		t.Errorf("expected error and nil config, got %v", cfg)
	}
	if err == nil {
		t.Errorf("expected error, got nil")
	}

	// parsing of mountContainers argument
	// single entry
	val := []string{"fs:name=value:container"}
	cfg, err = ParseMountConfig("container", val, false)
	if err != nil {
		t.Errorf("parsing '%s': expected configuration, got error %v", val, err)
	}
	mounter := cfg.MountContainers["fs"]
	if mounter.Selector.String() != "name=value" || mounter.ContainerName != "container" {
		t.Errorf("parsing '%s': expected selector 'name=value' and container 'container', got %v", val, mounter)
	}

	// multiple entries
	val = []string{"fs1:name1=value1,name11=value11:container1", "fs2:name2=value2:container2", "fs3:name3=value3:container3"}
	cfg, err = ParseMountConfig("container", val, false)
	if err != nil {
		t.Errorf("parsing '%s': expected configuration, got error %v", val, err)
	}
	mounter = cfg.MountContainers["fs1"]
	if mounter.Selector.String() != "name1=value1,name11=value11" || mounter.ContainerName != "container1" {
		t.Errorf("parsing '%s': expected selector 'name1=value1,name11=value11' and container 'container1', got %v", val, mounter)
	}
	mounter = cfg.MountContainers["fs2"]
	if mounter.Selector.String() != "name2=value2" || mounter.ContainerName != "container2" {
		t.Errorf("parsing '%s': expected selector 'name2=value2' and container 'container2', got %v", val, mounter)
	}
	mounter = cfg.MountContainers["fs3"]
	if mounter.Selector.String() != "name3=value3" || mounter.ContainerName != "container3" {
		t.Errorf("parsing '%s': expected selector 'name3=value3' and container 'container3', got %v", val, mounter)
	}

	// invalid entry: too few items
	val = []string{"fs:pod"}
	cfg, err = ParseMountConfig("container", val, false)
	if err == nil {
		t.Errorf("expected error when parsing %s", val)
	}
	// invalid entry: too many items
	val = []string{"fs:name=value:container:junk"}
	cfg, err = ParseMountConfig("container", val, false)
	if err == nil {
		t.Errorf("expected error when parsing %s", val)
	}

}
