/**
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package scheduler

import (
	mesos "github.com/mesos/mesos-go/mesosproto"
	util "github.com/mesos/mesos-go/mesosutil"
	"github.com/stretchr/testify/assert"
	"testing"

	"github.com/mesos/mesos-go/upid"
)

func TestSchedCacheNew(t *testing.T) {
	cache := newSchedCache()
	assert.NotNil(t, cache)
	assert.NotNil(t, cache.savedOffers)
	assert.NotNil(t, cache.savedSlavePids)
}

func TestSchedCachePutOffer(t *testing.T) {
	cache := newSchedCache()

	offer01 := createTestOffer("01")
	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	cache.putOffer(offer01, pid01)

	offer02 := createTestOffer("02")
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)
	cache.putOffer(offer02, pid02)

	assert.Equal(t, len(cache.savedOffers), 2)
	cachedOffer1, ok := cache.savedOffers["test-offer-01"]
	assert.True(t, ok)
	cachedOffer2, ok := cache.savedOffers["test-offer-02"]
	assert.True(t, ok)

	assert.NotNil(t, cachedOffer1.offer)
	assert.Equal(t, "test-offer-01", cachedOffer1.offer.Id.GetValue())
	assert.NotNil(t, cachedOffer2.offer)
	assert.Equal(t, "test-offer-02", cachedOffer2.offer.Id.GetValue())

	assert.NotNil(t, cachedOffer1.slavePid)
	assert.Equal(t, "slave01@127.0.0.1:5050", cachedOffer1.slavePid.String())
	assert.NotNil(t, cachedOffer2.slavePid)
	assert.Equal(t, "slave02@127.0.0.1:5050", cachedOffer2.slavePid.String())

}

func TestSchedCacheGetOffer(t *testing.T) {
	cache := newSchedCache()
	offer01 := createTestOffer("01")
	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	offer02 := createTestOffer("02")
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putOffer(offer01, pid01)
	cache.putOffer(offer02, pid02)

	cachedOffer01 := cache.getOffer(util.NewOfferID("test-offer-01")).offer
	cachedOffer02 := cache.getOffer(util.NewOfferID("test-offer-02")).offer
	assert.NotEqual(t, offer01, cachedOffer02)
	assert.Equal(t, offer01, cachedOffer01)
	assert.Equal(t, offer02, cachedOffer02)

}

func TestSchedCacheContainsOffer(t *testing.T) {
	cache := newSchedCache()
	offer01 := createTestOffer("01")
	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	offer02 := createTestOffer("02")
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putOffer(offer01, pid01)
	cache.putOffer(offer02, pid02)

	assert.True(t, cache.containsOffer(util.NewOfferID("test-offer-01")))
	assert.True(t, cache.containsOffer(util.NewOfferID("test-offer-02")))
	assert.False(t, cache.containsOffer(util.NewOfferID("test-offer-05")))
}

func TestSchedCacheRemoveOffer(t *testing.T) {
	cache := newSchedCache()
	offer01 := createTestOffer("01")
	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	offer02 := createTestOffer("02")
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putOffer(offer01, pid01)
	cache.putOffer(offer02, pid02)
	cache.removeOffer(util.NewOfferID("test-offer-01"))

	assert.Equal(t, 1, len(cache.savedOffers))
	assert.True(t, cache.containsOffer(util.NewOfferID("test-offer-02")))
	assert.False(t, cache.containsOffer(util.NewOfferID("test-offer-01")))
}

func TestSchedCachePutSlavePid(t *testing.T) {
	cache := newSchedCache()

	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)
	pid03, err := upid.Parse("slave03@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putSlavePid(util.NewSlaveID("slave01"), pid01)
	cache.putSlavePid(util.NewSlaveID("slave02"), pid02)
	cache.putSlavePid(util.NewSlaveID("slave03"), pid03)

	assert.Equal(t, len(cache.savedSlavePids), 3)
	cachedSlavePid1, ok := cache.savedSlavePids["slave01"]
	assert.True(t, ok)
	cachedSlavePid2, ok := cache.savedSlavePids["slave02"]
	assert.True(t, ok)
	cachedSlavePid3, ok := cache.savedSlavePids["slave03"]
	assert.True(t, ok)

	assert.True(t, cachedSlavePid1.Equal(pid01))
	assert.True(t, cachedSlavePid2.Equal(pid02))
	assert.True(t, cachedSlavePid3.Equal(pid03))
}

func TestSchedCacheGetSlavePid(t *testing.T) {
	cache := newSchedCache()

	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putSlavePid(util.NewSlaveID("slave01"), pid01)
	cache.putSlavePid(util.NewSlaveID("slave02"), pid02)

	cachedSlavePid1 := cache.getSlavePid(util.NewSlaveID("slave01"))
	cachedSlavePid2 := cache.getSlavePid(util.NewSlaveID("slave02"))

	assert.NotNil(t, cachedSlavePid1)
	assert.NotNil(t, cachedSlavePid2)
	assert.True(t, pid01.Equal(cachedSlavePid1))
	assert.True(t, pid02.Equal(cachedSlavePid2))
	assert.False(t, pid01.Equal(cachedSlavePid2))
}

func TestSchedCacheContainsSlavePid(t *testing.T) {
	cache := newSchedCache()

	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putSlavePid(util.NewSlaveID("slave01"), pid01)
	cache.putSlavePid(util.NewSlaveID("slave02"), pid02)

	assert.True(t, cache.containsSlavePid(util.NewSlaveID("slave01")))
	assert.True(t, cache.containsSlavePid(util.NewSlaveID("slave02")))
	assert.False(t, cache.containsSlavePid(util.NewSlaveID("slave05")))
}

func TestSchedCacheRemoveSlavePid(t *testing.T) {
	cache := newSchedCache()

	pid01, err := upid.Parse("slave01@127.0.0.1:5050")
	assert.NoError(t, err)
	pid02, err := upid.Parse("slave02@127.0.0.1:5050")
	assert.NoError(t, err)

	cache.putSlavePid(util.NewSlaveID("slave01"), pid01)
	cache.putSlavePid(util.NewSlaveID("slave02"), pid02)

	assert.True(t, cache.containsSlavePid(util.NewSlaveID("slave01")))
	assert.True(t, cache.containsSlavePid(util.NewSlaveID("slave02")))
	assert.False(t, cache.containsSlavePid(util.NewSlaveID("slave05")))

	cache.removeSlavePid(util.NewSlaveID("slave01"))
	assert.Equal(t, 1, len(cache.savedSlavePids))
	assert.False(t, cache.containsSlavePid(util.NewSlaveID("slave01")))

}

func createTestOffer(idSuffix string) *mesos.Offer {
	return util.NewOffer(
		util.NewOfferID("test-offer-"+idSuffix),
		util.NewFrameworkID("test-framework-"+idSuffix),
		util.NewSlaveID("test-slave-"+idSuffix),
		"localhost."+idSuffix,
	)
}
