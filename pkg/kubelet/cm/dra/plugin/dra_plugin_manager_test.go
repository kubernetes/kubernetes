/*
Copyright 2024 The Kubernetes Authors.

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

package plugin

import (
	"fmt"
	"math/rand/v2"
	"testing"

	"github.com/stretchr/testify/require"
	"k8s.io/kubernetes/test/utils/ktesting"
)

func TestAddSameName(t *testing.T) {
	tCtx := ktesting.Init(t)
	// name will have a random value to avoid conflicts
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, "old.sock", "", defaultClientCallTimeout), "add first plugin")
	p, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get first plugin")

	// Same name, same endpoint -> error.
	require.Error(tCtx, draPlugins.add(driverName, "old.sock", "", defaultClientCallTimeout))

	tCtx.ExpectNoError(draPlugins.add(driverName, "new.sock", "", defaultClientCallTimeout), "add second plugin")
	p2, err := draPlugins.GetPlugin(driverName)
	tCtx.ExpectNoError(err, "get second plugin")
	if p == p2 {
		tCtx.Fatal("expected to get second plugin, got first one again")
	}

	// Remove old plugin.
	draPlugins.remove(p.driverName, p.endpoint)
	plugin, err := draPlugins.GetPlugin(driverName)

	// Now we should have p2 left.
	tCtx.ExpectNoError(err, "get plugin")
	if p2 != plugin {
		tCtx.Fatal("expected to get second plugin again, got something else")
	}
}

func TestDelete(t *testing.T) {
	tCtx := ktesting.Init(t)
	driverName := fmt.Sprintf("dummy-driver-%d", rand.IntN(10000))

	// ensure the plugin we are using is registered
	draPlugins := NewDRAPluginManager(tCtx, nil, nil, 0)
	tCtx.ExpectNoError(draPlugins.add(driverName, "dra.sock", "", defaultClientCallTimeout), "add plugin")
	draPlugins.remove(driverName, "")
}
