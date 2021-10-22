package misc_tests

import (
	"encoding/json"
	"math/big"
	"testing"

	"github.com/json-iterator/go"
	"github.com/stretchr/testify/require"
	"strings"
)

func Test_decode_TextMarshaler_key_map(t *testing.T) {
	should := require.New(t)
	var val map[*big.Float]string
	should.Nil(jsoniter.UnmarshalFromString(`{"1":"2"}`, &val))
	str, err := jsoniter.MarshalToString(val)
	should.Nil(err)
	should.Equal(`{"1":"2"}`, str)
}

func Test_read_map_with_reader(t *testing.T) {
	should := require.New(t)
	input := `{"branch":"beta","change_log":"add the rows{10}","channel":"fros","create_time":"2017-06-13 16:39:08","firmware_list":"","md5":"80dee2bf7305bcf179582088e29fd7b9","note":{"CoreServices":{"md5":"d26975c0a8c7369f70ed699f2855cc2e","package_name":"CoreServices","version_code":"76","version_name":"1.0.76"},"FrDaemon":{"md5":"6b1f0626673200bc2157422cd2103f5d","package_name":"FrDaemon","version_code":"390","version_name":"1.0.390"},"FrGallery":{"md5":"90d767f0f31bcd3c1d27281ec979ba65","package_name":"FrGallery","version_code":"349","version_name":"1.0.349"},"FrLocal":{"md5":"f15a215b2c070a80a01f07bde4f219eb","package_name":"FrLocal","version_code":"791","version_name":"1.0.791"}},"pack_region_urls":{"CN":"https://s3.cn-north-1.amazonaws.com.cn/xxx-os/ttt_xxx_android_1.5.3.344.393.zip","default":"http://192.168.8.78/ttt_xxx_android_1.5.3.344.393.zip","local":"http://192.168.8.78/ttt_xxx_android_1.5.3.344.393.zip"},"pack_version":"1.5.3.344.393","pack_version_code":393,"region":"all","release_flag":0,"revision":62,"size":38966875,"status":3}`
	reader := strings.NewReader(input)
	decoder := jsoniter.ConfigCompatibleWithStandardLibrary.NewDecoder(reader)
	m1 := map[string]interface{}{}
	should.Nil(decoder.Decode(&m1))
	m2 := map[string]interface{}{}
	should.Nil(json.Unmarshal([]byte(input), &m2))
	should.Equal(m2, m1)
	should.Equal("1.0.76", m1["note"].(map[string]interface{})["CoreServices"].(map[string]interface{})["version_name"])
}

func Test_map_eface_of_eface(t *testing.T) {
	should := require.New(t)
	json := jsoniter.ConfigCompatibleWithStandardLibrary
	output, err := json.MarshalToString(map[interface{}]interface{}{
		"1": 2,
		3:   "4",
	})
	should.NoError(err)
	should.Equal(`{"1":2,"3":"4"}`, output)
}

func Test_encode_nil_map(t *testing.T) {
	should := require.New(t)
	var nilMap map[string]string
	output, err := jsoniter.MarshalToString(nilMap)
	should.NoError(err)
	should.Equal(`null`, output)
}
