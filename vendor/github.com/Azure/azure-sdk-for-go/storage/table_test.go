package storage

import (
	"crypto/rand"
	"encoding/base64"
	"fmt"
	"reflect"

	chk "gopkg.in/check.v1"
)

type TableClient struct{}

func getTableClient(c *chk.C) TableServiceClient {
	return getBasicClient(c).GetTableService()
}

type CustomEntity struct {
	Name    string `json:"name"`
	Surname string `json:"surname"`
	Number  int
	PKey    string `json:"pk" table:"-"`
	RKey    string `json:"rk" table:"-"`
}

type CustomEntityExtended struct {
	*CustomEntity
	ExtraField string
}

func (c *CustomEntity) PartitionKey() string {
	return c.PKey
}

func (c *CustomEntity) RowKey() string {
	return c.RKey
}

func (c *CustomEntity) SetPartitionKey(s string) error {
	c.PKey = s
	return nil
}

func (c *CustomEntity) SetRowKey(s string) error {
	c.RKey = s
	return nil
}

func (s *StorageBlobSuite) Test_SharedKeyLite(c *chk.C) {
	cli := getTableClient(c)

	// override the accountKey and accountName
	// but make sure to reset when returning
	oldAK := cli.client.accountKey
	oldAN := cli.client.accountName

	defer func() {
		cli.client.accountKey = oldAK
		cli.client.accountName = oldAN
	}()

	// don't worry, I've already changed mine :)
	key, err := base64.StdEncoding.DecodeString("zHDHGs7C+Di9pZSDMuarxJJz3xRBzAHBYaobxpLEc7kwTptR/hPEa9j93hIfb2Tbe9IA50MViGmjQ6nUF/OVvA==")
	if err != nil {
		c.Fail()
	}

	cli.client.accountKey = key
	cli.client.accountName = "mindgotest"

	headers := map[string]string{
		"Accept-Charset": "UTF-8",
		"Content-Type":   "application/json",
		"x-ms-date":      "Wed, 23 Sep 2015 16:40:05 GMT",
		"Content-Length": "0",
		"x-ms-version":   "2015-02-21",
		"Accept":         "application/json;odata=nometadata",
	}
	url := "https://mindgotest.table.core.windows.net/tquery()"

	ret, err := cli.client.createSharedKeyLite(url, headers)
	if err != nil {
		c.Fail()
	}

	c.Assert(ret, chk.Equals, "SharedKeyLite mindgotest:+32DTgsPUgXPo/O7RYaTs0DllA6FTXMj3uK4Qst8y/E=")
}

func (s *StorageBlobSuite) Test_CreateAndDeleteTable(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)

	err = cli.DeleteTable(tn)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) Test_InsertEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Luke", Surname: "Skywalker", Number: 1543, PKey: "pkey"}

	for i := 0; i < 12; i++ {
		ce.SetRowKey(fmt.Sprintf("%d", i))

		err = cli.InsertEntity(tn, ce)
		c.Assert(err, chk.IsNil)
	}
}

func (s *StorageBlobSuite) Test_InsertOrReplaceEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Darth", Surname: "Skywalker", Number: 60, PKey: "pkey", RKey: "5"}

	err = cli.InsertOrReplaceEntity(tn, ce)
	c.Assert(err, chk.IsNil)

	cextra := &CustomEntityExtended{&CustomEntity{PKey: "pkey", RKey: "5"}, "extra"}
	err = cli.InsertOrReplaceEntity(tn, cextra)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) Test_InsertOrMergeEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Darth", Surname: "Skywalker", Number: 60, PKey: "pkey", RKey: "5"}

	err = cli.InsertOrMergeEntity(tn, ce)
	c.Assert(err, chk.IsNil)

	cextra := &CustomEntityExtended{&CustomEntity{PKey: "pkey", RKey: "5"}, "extra"}
	err = cli.InsertOrReplaceEntity(tn, cextra)
	c.Assert(err, chk.IsNil)
}

func (s *StorageBlobSuite) Test_InsertAndGetEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Darth", Surname: "Skywalker", Number: 60, PKey: "pkey", RKey: "100"}
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	ce.SetRowKey("200")
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	entries, _, err := cli.QueryTableEntities(tn, nil, reflect.TypeOf(ce), 10, "")
	c.Assert(err, chk.IsNil)

	c.Assert(len(entries), chk.Equals, 2)

	c.Assert(ce.RowKey(), chk.Equals, entries[1].RowKey())

	c.Assert(entries[1].(*CustomEntity), chk.DeepEquals, ce)
}

func (s *StorageBlobSuite) Test_InsertAndQueryEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Darth", Surname: "Skywalker", Number: 60, PKey: "pkey", RKey: "100"}
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	ce.SetRowKey("200")
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	entries, _, err := cli.QueryTableEntities(tn, nil, reflect.TypeOf(ce), 10, "RowKey eq '200'")
	c.Assert(err, chk.IsNil)

	c.Assert(len(entries), chk.Equals, 1)

	c.Assert(ce.RowKey(), chk.Equals, entries[0].RowKey())
}

func (s *StorageBlobSuite) Test_InsertAndDeleteEntities(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	ce := &CustomEntity{Name: "Test", Surname: "Test2", Number: 0, PKey: "pkey", RKey: "r01"}
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	ce.Number = 1
	ce.SetRowKey("r02")
	c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)

	entries, _, err := cli.QueryTableEntities(tn, nil, reflect.TypeOf(ce), 10, "Number eq 1")
	c.Assert(err, chk.IsNil)

	c.Assert(len(entries), chk.Equals, 1)

	c.Assert(entries[0].(*CustomEntity), chk.DeepEquals, ce)

	c.Assert(cli.DeleteEntityWithoutCheck(tn, entries[0]), chk.IsNil)

	entries, _, err = cli.QueryTableEntities(tn, nil, reflect.TypeOf(ce), 10, "")
	c.Assert(err, chk.IsNil)

	// only 1 entry must be present
	c.Assert(len(entries), chk.Equals, 1)
}

func (s *StorageBlobSuite) Test_ContinuationToken(c *chk.C) {
	cli := getTableClient(c)

	tn := AzureTable(randTable())

	err := cli.CreateTable(tn)
	c.Assert(err, chk.IsNil)
	defer cli.DeleteTable(tn)

	var ce *CustomEntity
	var ceList [5]*CustomEntity

	for i := 0; i < 5; i++ {
		ce = &CustomEntity{Name: "Test", Surname: "Test2", Number: i, PKey: "pkey", RKey: fmt.Sprintf("r%d", i)}
		ceList[i] = ce
		c.Assert(cli.InsertOrReplaceEntity(tn, ce), chk.IsNil)
	}

	// retrieve using top = 2. Should return 2 entries, 2 entries and finally
	// 1 entry
	entries, contToken, err := cli.QueryTableEntities(tn, nil, reflect.TypeOf(ce), 2, "")
	c.Assert(err, chk.IsNil)
	c.Assert(len(entries), chk.Equals, 2)
	c.Assert(entries[0].(*CustomEntity), chk.DeepEquals, ceList[0])
	c.Assert(entries[1].(*CustomEntity), chk.DeepEquals, ceList[1])
	c.Assert(contToken, chk.NotNil)

	entries, contToken, err = cli.QueryTableEntities(tn, contToken, reflect.TypeOf(ce), 2, "")
	c.Assert(err, chk.IsNil)
	c.Assert(len(entries), chk.Equals, 2)
	c.Assert(entries[0].(*CustomEntity), chk.DeepEquals, ceList[2])
	c.Assert(entries[1].(*CustomEntity), chk.DeepEquals, ceList[3])
	c.Assert(contToken, chk.NotNil)

	entries, contToken, err = cli.QueryTableEntities(tn, contToken, reflect.TypeOf(ce), 2, "")
	c.Assert(err, chk.IsNil)
	c.Assert(len(entries), chk.Equals, 1)
	c.Assert(entries[0].(*CustomEntity), chk.DeepEquals, ceList[4])
	c.Assert(contToken, chk.IsNil)
}

func randTable() string {
	const alphanum = "abcdefghijklmnopqrstuvwxyz"
	var bytes = make([]byte, 32)
	rand.Read(bytes)
	for i, b := range bytes {
		bytes[i] = alphanum[b%byte(len(alphanum))]
	}
	return string(bytes)
}
