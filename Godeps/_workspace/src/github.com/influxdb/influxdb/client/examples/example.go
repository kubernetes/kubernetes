package examples

import (
	"fmt"

	"github.com/influxdb/influxdb/client"
)

func main() {
	TestClient()
}

func TestClient() {
	internalTest(true)
}

func TestClientWithoutCompression() {
	internalTest(false)
}

func internalTest(compression bool) {
	c, err := client.NewClient(&client.ClientConfig{})
	if err != nil {
		panic(err)
	}

	admins, err := c.GetClusterAdminList()
	if err != nil {
		panic(err)
	}

	if len(admins) == 1 {
		if err := c.CreateClusterAdmin("admin", "password"); err != nil {
			panic(err)
		}
	}

	admins, err = c.GetClusterAdminList()
	if err != nil {
		panic(err)
	}

	if len(admins) != 2 {
		panic("more than two admins returned")
	}

	dbs, err := c.GetDatabaseList()
	if err != nil {
		panic(err)
	}

	if len(dbs) == 0 {
		if err := c.CreateDatabase("foobar"); err != nil {
			panic(err)
		}
	}

	dbs, err = c.GetDatabaseList()
	if err != nil {
		panic(err)
	}

	if len(dbs) != 1 && dbs[0]["foobar"] == nil {
		panic("List of databases don't match")
	}

	users, err := c.GetDatabaseUserList("foobar")
	if err != nil {
		panic(err)
	}

	if len(users) == 0 {
		if err := c.CreateDatabaseUser("foobar", "dbuser", "pass"); err != nil {
			panic(err)
		}

		if err := c.AlterDatabasePrivilege("foobar", "dbuser", true); err != nil {
			panic(err)
		}
	}

	users, err = c.GetDatabaseUserList("foobar")
	if err != nil {
		panic(err)
	}

	if len(users) != 1 {
		panic("more than one user returned")
	}

	c, err = client.NewClient(&client.ClientConfig{
		Username: "dbuser",
		Password: "pass",
		Database: "foobar",
	})

	if !compression {
		c.DisableCompression()
	}

	if err != nil {
		panic(err)
	}

	name := "ts9"
	if !compression {
		name = "ts9_uncompressed"
	}

	series := &client.Series{
		Name:    name,
		Columns: []string{"value"},
		Points: [][]interface{}{
			{1.0},
		},
	}
	if err := c.WriteSeries([]*client.Series{series}); err != nil {
		panic(err)
	}

	result, err := c.Query("select * from " + name)
	if err != nil {
		panic(err)
	}

	if len(result) != 1 {
		panic(fmt.Errorf("expected one time series returned: %d", len(result)))
	}

	if len(result[0].Points) != 1 {
		panic(fmt.Errorf("Expected one point: %d", len(result[0].Points)))
	}

	if result[0].Points[0][2].(float64) != 1 {
		panic("Value not equal to 1")
	}

	c, err = client.NewClient(&client.ClientConfig{
		Username: "root",
		Password: "root",
	})

	if err != nil {
		panic(err)
	}

	spaces, err := c.GetShardSpaces()
	if err != nil || len(spaces) == 0 {
		panic(fmt.Errorf("Got empty spaces back: %s", err))
	}
	if spaces[0].Name != "default" {
		panic("Space name isn't default")
	}
	space := &client.ShardSpace{Name: "foo", Regex: "/^paul_is_rad/"}
	err = c.CreateShardSpace("foobar", space)
	if err != nil {
		panic(err)
	}
	spaces, _ = c.GetShardSpaces()
	if spaces[1].Name != "foo" {
		panic("Space name isn't foo")
	}
	shards, err := c.GetShards()
	if err != nil {
		panic(fmt.Errorf("Couldn't get shards back: %s", err))
	}

	c, err = client.NewClient(&client.ClientConfig{
		Username: "root",
		Password: "root",
		Database: "",
	})
	series = &client.Series{
		Name:    "paul_is_rad",
		Columns: []string{"value"},
		Points: [][]interface{}{
			{1.0},
		},
	}
	if err := c.WriteSeries([]*client.Series{series}); err != nil {
		panic(err)
	}

	spaces, _ = c.GetShardSpaces()
	count := 0
	for _, s := range shards.All {
		if s.SpaceName == "foo" {
			count++
		}
	}

	if err := c.DropShardSpace("foobar", "foo"); err != nil {
		panic(fmt.Errorf("Error: %s", err))
	}

	spaces, err = c.GetShardSpaces()
	if err != nil || len(spaces) != 1 || spaces[0].Name != "default" {
		panic(fmt.Errorf("Error: %s, %d, %s", err, len(spaces), spaces[0].Name))
	}
}
