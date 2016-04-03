package zoo

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

const (
	zkurl = "zk://127.0.0.1:2181/mesos"
)

func TestParseZk_single(t *testing.T) {
	hosts, path, err := parseZk(zkurl)
	assert.NoError(t, err)
	assert.Equal(t, 1, len(hosts))
	assert.Equal(t, "/mesos", path)
}

func TestParseZk_multi(t *testing.T) {
	hosts, path, err := parseZk("zk://abc:1,def:2/foo")
	assert.NoError(t, err)
	assert.Equal(t, []string{"abc:1", "def:2"}, hosts)
	assert.Equal(t, "/foo", path)
}

func TestParseZk_multiIP(t *testing.T) {
	hosts, path, err := parseZk("zk://10.186.175.156:2181,10.47.50.94:2181,10.0.92.171:2181/mesos")
	assert.NoError(t, err)
	assert.Equal(t, []string{"10.186.175.156:2181", "10.47.50.94:2181", "10.0.92.171:2181"}, hosts)
	assert.Equal(t, "/mesos", path)
}

func TestMasterDetect_selectTopNode_none(t *testing.T) {
	assert := assert.New(t)
	nodeList := []string{}
	node := selectTopNodePrefix(nodeList, "foo")
	assert.Equal("", node)
}

func TestMasterDetect_selectTopNode_0000x(t *testing.T) {
	assert := assert.New(t)
	nodeList := []string{
		"info_0000000046",
		"info_0000000032",
		"info_0000000058",
		"info_0000000061",
		"info_0000000008",
	}
	node := selectTopNodePrefix(nodeList, nodePrefix)
	assert.Equal("info_0000000008", node)
}

func TestMasterDetect_selectTopNode_mixJson(t *testing.T) {
	assert := assert.New(t)
	nodeList := []string{
		nodePrefix + "0000000046",
		nodePrefix + "0000000032",
		nodeJSONPrefix + "0000000046",
		nodeJSONPrefix + "0000000032",
	}
	node := selectTopNodePrefix(nodeList, nodeJSONPrefix)
	assert.Equal(nodeJSONPrefix+"0000000032", node)

	node = selectTopNodePrefix(nodeList, nodePrefix)
	assert.Equal(nodePrefix+"0000000032", node)
}

func TestMasterDetect_selectTopNode_mixedEntries(t *testing.T) {
	assert := assert.New(t)
	nodeList := []string{
		"info_0000000046",
		"info_0000000032",
		"foo_lskdjfglsdkfsdfgdfg",
		"info_0000000061",
		"log_replicas_fdgwsdfgsdf",
		"bar",
	}
	node := selectTopNodePrefix(nodeList, nodePrefix)
	assert.Equal("info_0000000032", node)
}
