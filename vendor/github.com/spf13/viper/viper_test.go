// Copyright © 2014 Steve Francia <spf@spf13.com>.
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file.

package viper

import (
	"bytes"
	"fmt"
	"io/ioutil"
	"os"
	"path"
	"reflect"
	"sort"
	"strings"
	"testing"
	"time"

	"github.com/spf13/pflag"
	"github.com/stretchr/testify/assert"
)

var yamlExample = []byte(`Hacker: true
name: steve
hobbies:
- skateboarding
- snowboarding
- go
clothing:
  jacket: leather
  trousers: denim
  pants:
    size: large
age: 35
eyes : brown
beard: true
`)

var yamlExampleWithExtras = []byte(`Existing: true
Bogus: true
`)

type testUnmarshalExtra struct {
	Existing bool
}

var tomlExample = []byte(`
title = "TOML Example"

[owner]
organization = "MongoDB"
Bio = "MongoDB Chief Developer Advocate & Hacker at Large"
dob = 1979-05-27T07:32:00Z # First class dates? Why not?`)

var jsonExample = []byte(`{
"id": "0001",
"type": "donut",
"name": "Cake",
"ppu": 0.55,
"batters": {
        "batter": [
                { "type": "Regular" },
                { "type": "Chocolate" },
                { "type": "Blueberry" },
                { "type": "Devil's Food" }
            ]
    }
}`)

var hclExample = []byte(`
id = "0001"
type = "donut"
name = "Cake"
ppu = 0.55
foos {
	foo {
		key = 1
	}
	foo {
		key = 2
	}
	foo {
		key = 3
	}
	foo {
		key = 4
	}
}`)

var propertiesExample = []byte(`
p_id: 0001
p_type: donut
p_name: Cake
p_ppu: 0.55
p_batters.batter.type: Regular
`)

var remoteExample = []byte(`{
"id":"0002",
"type":"cronut",
"newkey":"remote"
}`)

func initConfigs() {
	Reset()
	SetConfigType("yaml")
	r := bytes.NewReader(yamlExample)
	unmarshalReader(r, v.config)

	SetConfigType("json")
	r = bytes.NewReader(jsonExample)
	unmarshalReader(r, v.config)

	SetConfigType("hcl")
	r = bytes.NewReader(hclExample)
	unmarshalReader(r, v.config)

	SetConfigType("properties")
	r = bytes.NewReader(propertiesExample)
	unmarshalReader(r, v.config)

	SetConfigType("toml")
	r = bytes.NewReader(tomlExample)
	unmarshalReader(r, v.config)

	SetConfigType("json")
	remote := bytes.NewReader(remoteExample)
	unmarshalReader(remote, v.kvstore)
}

func initYAML() {
	Reset()
	SetConfigType("yaml")
	r := bytes.NewReader(yamlExample)

	unmarshalReader(r, v.config)
}

func initJSON() {
	Reset()
	SetConfigType("json")
	r := bytes.NewReader(jsonExample)

	unmarshalReader(r, v.config)
}

func initProperties() {
	Reset()
	SetConfigType("properties")
	r := bytes.NewReader(propertiesExample)

	unmarshalReader(r, v.config)
}

func initTOML() {
	Reset()
	SetConfigType("toml")
	r := bytes.NewReader(tomlExample)

	unmarshalReader(r, v.config)
}

func initHcl() {
	Reset()
	SetConfigType("hcl")
	r := bytes.NewReader(hclExample)

	unmarshalReader(r, v.config)
}

// make directories for testing
func initDirs(t *testing.T) (string, string, func()) {

	var (
		testDirs = []string{`a a`, `b`, `c\c`, `D_`}
		config   = `improbable`
	)

	root, err := ioutil.TempDir("", "")

	cleanup := true
	defer func() {
		if cleanup {
			os.Chdir("..")
			os.RemoveAll(root)
		}
	}()

	assert.Nil(t, err)

	err = os.Chdir(root)
	assert.Nil(t, err)

	for _, dir := range testDirs {
		err = os.Mkdir(dir, 0750)
		assert.Nil(t, err)

		err = ioutil.WriteFile(
			path.Join(dir, config+".toml"),
			[]byte("key = \"value is "+dir+"\"\n"),
			0640)
		assert.Nil(t, err)
	}

	cleanup = false
	return root, config, func() {
		os.Chdir("..")
		os.RemoveAll(root)
	}
}

//stubs for PFlag Values
type stringValue string

func newStringValue(val string, p *string) *stringValue {
	*p = val
	return (*stringValue)(p)
}

func (s *stringValue) Set(val string) error {
	*s = stringValue(val)
	return nil
}

func (s *stringValue) Type() string {
	return "string"
}

func (s *stringValue) String() string {
	return fmt.Sprintf("%s", *s)
}

func TestBasics(t *testing.T) {
	SetConfigFile("/tmp/config.yaml")
	assert.Equal(t, "/tmp/config.yaml", v.getConfigFile())
}

func TestDefault(t *testing.T) {
	SetDefault("age", 45)
	assert.Equal(t, 45, Get("age"))

	SetDefault("clothing.jacket", "slacks")
	assert.Equal(t, "slacks", Get("clothing.jacket"))

	SetConfigType("yaml")
	err := ReadConfig(bytes.NewBuffer(yamlExample))

	assert.NoError(t, err)
	assert.Equal(t, "leather", Get("clothing.jacket"))
}

func TestUnmarshalling(t *testing.T) {
	SetConfigType("yaml")
	r := bytes.NewReader(yamlExample)

	unmarshalReader(r, v.config)
	assert.True(t, InConfig("name"))
	assert.False(t, InConfig("state"))
	assert.Equal(t, "steve", Get("name"))
	assert.Equal(t, []interface{}{"skateboarding", "snowboarding", "go"}, Get("hobbies"))
	assert.Equal(t, map[interface{}]interface{}{"jacket": "leather", "trousers": "denim", "pants": map[interface{}]interface{}{"size": "large"}}, Get("clothing"))
	assert.Equal(t, 35, Get("age"))
}

func TestUnmarshalExact(t *testing.T) {
	vip := New()
	target := &testUnmarshalExtra{}
	vip.SetConfigType("yaml")
	r := bytes.NewReader(yamlExampleWithExtras)
	vip.ReadConfig(r)
	err := vip.UnmarshalExact(target)
	if err == nil {
		t.Fatal("UnmarshalExact should error when populating a struct from a conf that contains unused fields")
	}
}

func TestOverrides(t *testing.T) {
	Set("age", 40)
	assert.Equal(t, 40, Get("age"))
}

func TestDefaultPost(t *testing.T) {
	assert.NotEqual(t, "NYC", Get("state"))
	SetDefault("state", "NYC")
	assert.Equal(t, "NYC", Get("state"))
}

func TestAliases(t *testing.T) {
	RegisterAlias("years", "age")
	assert.Equal(t, 40, Get("years"))
	Set("years", 45)
	assert.Equal(t, 45, Get("age"))
}

func TestAliasInConfigFile(t *testing.T) {
	// the config file specifies "beard".  If we make this an alias for
	// "hasbeard", we still want the old config file to work with beard.
	RegisterAlias("beard", "hasbeard")
	assert.Equal(t, true, Get("hasbeard"))
	Set("hasbeard", false)
	assert.Equal(t, false, Get("beard"))
}

func TestYML(t *testing.T) {
	initYAML()
	assert.Equal(t, "steve", Get("name"))
}

func TestJSON(t *testing.T) {
	initJSON()
	assert.Equal(t, "0001", Get("id"))
}

func TestProperties(t *testing.T) {
	initProperties()
	assert.Equal(t, "0001", Get("p_id"))
}

func TestTOML(t *testing.T) {
	initTOML()
	assert.Equal(t, "TOML Example", Get("title"))
}

func TestHCL(t *testing.T) {
	initHcl()
	assert.Equal(t, "0001", Get("id"))
	assert.Equal(t, 0.55, Get("ppu"))
	assert.Equal(t, "donut", Get("type"))
	assert.Equal(t, "Cake", Get("name"))
	Set("id", "0002")
	assert.Equal(t, "0002", Get("id"))
	assert.NotEqual(t, "cronut", Get("type"))
}

func TestRemotePrecedence(t *testing.T) {
	initJSON()

	remote := bytes.NewReader(remoteExample)
	assert.Equal(t, "0001", Get("id"))
	unmarshalReader(remote, v.kvstore)
	assert.Equal(t, "0001", Get("id"))
	assert.NotEqual(t, "cronut", Get("type"))
	assert.Equal(t, "remote", Get("newkey"))
	Set("newkey", "newvalue")
	assert.NotEqual(t, "remote", Get("newkey"))
	assert.Equal(t, "newvalue", Get("newkey"))
	Set("newkey", "remote")
}

func TestEnv(t *testing.T) {
	initJSON()

	BindEnv("id")
	BindEnv("f", "FOOD")

	os.Setenv("ID", "13")
	os.Setenv("FOOD", "apple")
	os.Setenv("NAME", "crunk")

	assert.Equal(t, "13", Get("id"))
	assert.Equal(t, "apple", Get("f"))
	assert.Equal(t, "Cake", Get("name"))

	AutomaticEnv()

	assert.Equal(t, "crunk", Get("name"))

}

func TestEnvPrefix(t *testing.T) {
	initJSON()

	SetEnvPrefix("foo") // will be uppercased automatically
	BindEnv("id")
	BindEnv("f", "FOOD") // not using prefix

	os.Setenv("FOO_ID", "13")
	os.Setenv("FOOD", "apple")
	os.Setenv("FOO_NAME", "crunk")

	assert.Equal(t, "13", Get("id"))
	assert.Equal(t, "apple", Get("f"))
	assert.Equal(t, "Cake", Get("name"))

	AutomaticEnv()

	assert.Equal(t, "crunk", Get("name"))
}

func TestAutoEnv(t *testing.T) {
	Reset()

	AutomaticEnv()
	os.Setenv("FOO_BAR", "13")
	assert.Equal(t, "13", Get("foo_bar"))
}

func TestAutoEnvWithPrefix(t *testing.T) {
	Reset()

	AutomaticEnv()
	SetEnvPrefix("Baz")
	os.Setenv("BAZ_BAR", "13")
	assert.Equal(t, "13", Get("bar"))
}

func TestSetEnvReplacer(t *testing.T) {
	Reset()

	AutomaticEnv()
	os.Setenv("REFRESH_INTERVAL", "30s")

	replacer := strings.NewReplacer("-", "_")
	SetEnvKeyReplacer(replacer)

	assert.Equal(t, "30s", Get("refresh-interval"))
}

func TestAllKeys(t *testing.T) {
	initConfigs()

	ks := sort.StringSlice{"title", "newkey", "owner", "name", "beard", "ppu", "batters", "hobbies", "clothing", "age", "hacker", "id", "type", "eyes", "p_id", "p_ppu", "p_batters.batter.type", "p_type", "p_name", "foos"}
	dob, _ := time.Parse(time.RFC3339, "1979-05-27T07:32:00Z")
	all := map[string]interface{}{"owner": map[string]interface{}{"organization": "MongoDB", "Bio": "MongoDB Chief Developer Advocate & Hacker at Large", "dob": dob}, "title": "TOML Example", "ppu": 0.55, "eyes": "brown", "clothing": map[interface{}]interface{}{"trousers": "denim", "jacket": "leather", "pants": map[interface{}]interface{}{"size": "large"}}, "id": "0001", "batters": map[string]interface{}{"batter": []interface{}{map[string]interface{}{"type": "Regular"}, map[string]interface{}{"type": "Chocolate"}, map[string]interface{}{"type": "Blueberry"}, map[string]interface{}{"type": "Devil's Food"}}}, "hacker": true, "beard": true, "hobbies": []interface{}{"skateboarding", "snowboarding", "go"}, "age": 35, "type": "donut", "newkey": "remote", "name": "Cake", "p_id": "0001", "p_ppu": "0.55", "p_name": "Cake", "p_batters.batter.type": "Regular", "p_type": "donut", "foos": []map[string]interface{}{map[string]interface{}{"foo": []map[string]interface{}{map[string]interface{}{"key": 1}, map[string]interface{}{"key": 2}, map[string]interface{}{"key": 3}, map[string]interface{}{"key": 4}}}}}

	var allkeys sort.StringSlice
	allkeys = AllKeys()
	allkeys.Sort()
	ks.Sort()

	assert.Equal(t, ks, allkeys)
	assert.Equal(t, all, AllSettings())
}

func TestCaseInSensitive(t *testing.T) {
	assert.Equal(t, true, Get("hacker"))
	Set("Title", "Checking Case")
	assert.Equal(t, "Checking Case", Get("tItle"))
}

func TestAliasesOfAliases(t *testing.T) {
	RegisterAlias("Foo", "Bar")
	RegisterAlias("Bar", "Title")
	assert.Equal(t, "Checking Case", Get("FOO"))
}

func TestRecursiveAliases(t *testing.T) {
	RegisterAlias("Baz", "Roo")
	RegisterAlias("Roo", "baz")
}

func TestUnmarshal(t *testing.T) {
	SetDefault("port", 1313)
	Set("name", "Steve")

	type config struct {
		Port int
		Name string
	}

	var C config

	err := Unmarshal(&C)
	if err != nil {
		t.Fatalf("unable to decode into struct, %v", err)
	}

	assert.Equal(t, &C, &config{Name: "Steve", Port: 1313})

	Set("port", 1234)
	err = Unmarshal(&C)
	if err != nil {
		t.Fatalf("unable to decode into struct, %v", err)
	}
	assert.Equal(t, &C, &config{Name: "Steve", Port: 1234})
}

func TestBindPFlags(t *testing.T) {
	flagSet := pflag.NewFlagSet("test", pflag.ContinueOnError)

	var testValues = map[string]*string{
		"host":     nil,
		"port":     nil,
		"endpoint": nil,
	}

	var mutatedTestValues = map[string]string{
		"host":     "localhost",
		"port":     "6060",
		"endpoint": "/public",
	}

	for name, _ := range testValues {
		testValues[name] = flagSet.String(name, "", "test")
	}

	err := BindPFlags(flagSet)
	if err != nil {
		t.Fatalf("error binding flag set, %v", err)
	}

	flagSet.VisitAll(func(flag *pflag.Flag) {
		flag.Value.Set(mutatedTestValues[flag.Name])
		flag.Changed = true
	})

	for name, expected := range mutatedTestValues {
		assert.Equal(t, Get(name), expected)
	}

}

func TestBindPFlag(t *testing.T) {
	var testString = "testing"
	var testValue = newStringValue(testString, &testString)

	flag := &pflag.Flag{
		Name:    "testflag",
		Value:   testValue,
		Changed: false,
	}

	BindPFlag("testvalue", flag)

	assert.Equal(t, testString, Get("testvalue"))

	flag.Value.Set("testing_mutate")
	flag.Changed = true //hack for pflag usage

	assert.Equal(t, "testing_mutate", Get("testvalue"))

}

func TestBoundCaseSensitivity(t *testing.T) {

	assert.Equal(t, "brown", Get("eyes"))

	BindEnv("eYEs", "TURTLE_EYES")
	os.Setenv("TURTLE_EYES", "blue")

	assert.Equal(t, "blue", Get("eyes"))

	var testString = "green"
	var testValue = newStringValue(testString, &testString)

	flag := &pflag.Flag{
		Name:    "eyeballs",
		Value:   testValue,
		Changed: true,
	}

	BindPFlag("eYEs", flag)
	assert.Equal(t, "green", Get("eyes"))

}

func TestSizeInBytes(t *testing.T) {
	input := map[string]uint{
		"":               0,
		"b":              0,
		"12 bytes":       0,
		"200000000000gb": 0,
		"12 b":           12,
		"43 MB":          43 * (1 << 20),
		"10mb":           10 * (1 << 20),
		"1gb":            1 << 30,
	}

	for str, expected := range input {
		assert.Equal(t, expected, parseSizeInBytes(str), str)
	}
}

func TestFindsNestedKeys(t *testing.T) {
	initConfigs()
	dob, _ := time.Parse(time.RFC3339, "1979-05-27T07:32:00Z")

	Set("super", map[string]interface{}{
		"deep": map[string]interface{}{
			"nested": "value",
		},
	})

	expected := map[string]interface{}{
		"super": map[string]interface{}{
			"deep": map[string]interface{}{
				"nested": "value",
			},
		},
		"super.deep": map[string]interface{}{
			"nested": "value",
		},
		"super.deep.nested":  "value",
		"owner.organization": "MongoDB",
		"batters.batter": []interface{}{
			map[string]interface{}{
				"type": "Regular",
			},
			map[string]interface{}{
				"type": "Chocolate",
			},
			map[string]interface{}{
				"type": "Blueberry",
			},
			map[string]interface{}{
				"type": "Devil's Food",
			},
		},
		"hobbies": []interface{}{
			"skateboarding", "snowboarding", "go",
		},
		"title":  "TOML Example",
		"newkey": "remote",
		"batters": map[string]interface{}{
			"batter": []interface{}{
				map[string]interface{}{
					"type": "Regular",
				},
				map[string]interface{}{
					"type": "Chocolate",
				}, map[string]interface{}{
					"type": "Blueberry",
				}, map[string]interface{}{
					"type": "Devil's Food",
				},
			},
		},
		"eyes": "brown",
		"age":  35,
		"owner": map[string]interface{}{
			"organization": "MongoDB",
			"Bio":          "MongoDB Chief Developer Advocate & Hacker at Large",
			"dob":          dob,
		},
		"owner.Bio": "MongoDB Chief Developer Advocate & Hacker at Large",
		"type":      "donut",
		"id":        "0001",
		"name":      "Cake",
		"hacker":    true,
		"ppu":       0.55,
		"clothing": map[interface{}]interface{}{
			"jacket":   "leather",
			"trousers": "denim",
			"pants": map[interface{}]interface{}{
				"size": "large",
			},
		},
		"clothing.jacket":     "leather",
		"clothing.pants.size": "large",
		"clothing.trousers":   "denim",
		"owner.dob":           dob,
		"beard":               true,
		"foos": []map[string]interface{}{
			map[string]interface{}{
				"foo": []map[string]interface{}{
					map[string]interface{}{
						"key": 1,
					},
					map[string]interface{}{
						"key": 2,
					},
					map[string]interface{}{
						"key": 3,
					},
					map[string]interface{}{
						"key": 4,
					},
				},
			},
		},
	}

	for key, expectedValue := range expected {

		assert.Equal(t, expectedValue, v.Get(key))
	}

}

func TestReadBufConfig(t *testing.T) {
	v := New()
	v.SetConfigType("yaml")
	v.ReadConfig(bytes.NewBuffer(yamlExample))
	t.Log(v.AllKeys())

	assert.True(t, v.InConfig("name"))
	assert.False(t, v.InConfig("state"))
	assert.Equal(t, "steve", v.Get("name"))
	assert.Equal(t, []interface{}{"skateboarding", "snowboarding", "go"}, v.Get("hobbies"))
	assert.Equal(t, map[interface{}]interface{}{"jacket": "leather", "trousers": "denim", "pants": map[interface{}]interface{}{"size": "large"}}, v.Get("clothing"))
	assert.Equal(t, 35, v.Get("age"))
}

func TestIsSet(t *testing.T) {
	v := New()
	v.SetConfigType("yaml")
	v.ReadConfig(bytes.NewBuffer(yamlExample))
	assert.True(t, v.IsSet("clothing.jacket"))
	assert.False(t, v.IsSet("clothing.jackets"))
	assert.False(t, v.IsSet("helloworld"))
	v.Set("helloworld", "fubar")
	assert.True(t, v.IsSet("helloworld"))
}

func TestDirsSearch(t *testing.T) {

	root, config, cleanup := initDirs(t)
	defer cleanup()

	v := New()
	v.SetConfigName(config)
	v.SetDefault(`key`, `default`)

	entries, err := ioutil.ReadDir(root)
	for _, e := range entries {
		if e.IsDir() {
			v.AddConfigPath(e.Name())
		}
	}

	err = v.ReadInConfig()
	assert.Nil(t, err)

	assert.Equal(t, `value is `+path.Base(v.configPaths[0]), v.GetString(`key`))
}

func TestWrongDirsSearchNotFound(t *testing.T) {

	_, config, cleanup := initDirs(t)
	defer cleanup()

	v := New()
	v.SetConfigName(config)
	v.SetDefault(`key`, `default`)

	v.AddConfigPath(`whattayoutalkingbout`)
	v.AddConfigPath(`thispathaintthere`)

	err := v.ReadInConfig()
	assert.Equal(t, reflect.TypeOf(UnsupportedConfigError("")), reflect.TypeOf(err))

	// Even though config did not load and the error might have
	// been ignored by the client, the default still loads
	assert.Equal(t, `default`, v.GetString(`key`))
}

func TestSub(t *testing.T) {
	v := New()
	v.SetConfigType("yaml")
	v.ReadConfig(bytes.NewBuffer(yamlExample))

	subv := v.Sub("clothing")
	assert.Equal(t, v.Get("clothing.pants.size"), subv.Get("pants.size"))

	subv = v.Sub("clothing.pants")
	assert.Equal(t, v.Get("clothing.pants.size"), subv.Get("size"))

	subv = v.Sub("clothing.pants.size")
	assert.Equal(t, subv, (*Viper)(nil))
}

var yamlMergeExampleTgt = []byte(`
hello:
    pop: 37890
    lagrenum: 765432101234567
    world:
    - us
    - uk
    - fr
    - de
`)

var yamlMergeExampleSrc = []byte(`
hello:
    pop: 45000
    lagrenum: 7654321001234567
    universe:
    - mw
    - ad
fu: bar
`)

func TestMergeConfig(t *testing.T) {
	v := New()
	v.SetConfigType("yml")
	if err := v.ReadConfig(bytes.NewBuffer(yamlMergeExampleTgt)); err != nil {
		t.Fatal(err)
	}

	if pop := v.GetInt("hello.pop"); pop != 37890 {
		t.Fatalf("pop != 37890, = %d", pop)
	}

	if pop := v.GetInt("hello.lagrenum"); pop != 765432101234567 {
		t.Fatalf("lagrenum != 765432101234567, = %d", pop)
	}

	if pop := v.GetInt64("hello.lagrenum"); pop != int64(765432101234567) {
		t.Fatalf("int64 lagrenum != 765432101234567, = %d", pop)
	}

	if world := v.GetStringSlice("hello.world"); len(world) != 4 {
		t.Fatalf("len(world) != 4, = %d", len(world))
	}

	if fu := v.GetString("fu"); fu != "" {
		t.Fatalf("fu != \"\", = %s", fu)
	}

	if err := v.MergeConfig(bytes.NewBuffer(yamlMergeExampleSrc)); err != nil {
		t.Fatal(err)
	}

	if pop := v.GetInt("hello.pop"); pop != 45000 {
		t.Fatalf("pop != 45000, = %d", pop)
	}

	if pop := v.GetInt("hello.lagrenum"); pop != 7654321001234567 {
		t.Fatalf("lagrenum != 7654321001234567, = %d", pop)
	}

	if pop := v.GetInt64("hello.lagrenum"); pop != int64(7654321001234567) {
		t.Fatalf("int64 lagrenum != 7654321001234567, = %d", pop)
	}

	if world := v.GetStringSlice("hello.world"); len(world) != 4 {
		t.Fatalf("len(world) != 4, = %d", len(world))
	}

	if universe := v.GetStringSlice("hello.universe"); len(universe) != 2 {
		t.Fatalf("len(universe) != 2, = %d", len(universe))
	}

	if fu := v.GetString("fu"); fu != "bar" {
		t.Fatalf("fu != \"bar\", = %s", fu)
	}
}

func TestMergeConfigNoMerge(t *testing.T) {
	v := New()
	v.SetConfigType("yml")
	if err := v.ReadConfig(bytes.NewBuffer(yamlMergeExampleTgt)); err != nil {
		t.Fatal(err)
	}

	if pop := v.GetInt("hello.pop"); pop != 37890 {
		t.Fatalf("pop != 37890, = %d", pop)
	}

	if world := v.GetStringSlice("hello.world"); len(world) != 4 {
		t.Fatalf("len(world) != 4, = %d", len(world))
	}

	if fu := v.GetString("fu"); fu != "" {
		t.Fatalf("fu != \"\", = %s", fu)
	}

	if err := v.ReadConfig(bytes.NewBuffer(yamlMergeExampleSrc)); err != nil {
		t.Fatal(err)
	}

	if pop := v.GetInt("hello.pop"); pop != 45000 {
		t.Fatalf("pop != 45000, = %d", pop)
	}

	if world := v.GetStringSlice("hello.world"); len(world) != 0 {
		t.Fatalf("len(world) != 0, = %d", len(world))
	}

	if universe := v.GetStringSlice("hello.universe"); len(universe) != 2 {
		t.Fatalf("len(universe) != 2, = %d", len(universe))
	}

	if fu := v.GetString("fu"); fu != "bar" {
		t.Fatalf("fu != \"bar\", = %s", fu)
	}
}

func TestUnmarshalingWithAliases(t *testing.T) {
	SetDefault("Id", 1)
	Set("name", "Steve")
	Set("lastname", "Owen")

	RegisterAlias("UserID", "Id")
	RegisterAlias("Firstname", "name")
	RegisterAlias("Surname", "lastname")

	type config struct {
		Id        int
		FirstName string
		Surname   string
	}

	var C config

	err := Unmarshal(&C)
	if err != nil {
		t.Fatalf("unable to decode into struct, %v", err)
	}

	assert.Equal(t, &C, &config{Id: 1, FirstName: "Steve", Surname: "Owen"})
}

func TestSetConfigNameClearsFileCache(t *testing.T) {
	SetConfigFile("/tmp/config.yaml")
	SetConfigName("default")
	assert.Empty(t, v.getConfigFile())
}
