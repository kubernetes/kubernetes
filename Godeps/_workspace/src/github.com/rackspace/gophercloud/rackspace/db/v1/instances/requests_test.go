package instances

import (
	"testing"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/openstack/db/v1/flavors"
	os "github.com/rackspace/gophercloud/openstack/db/v1/instances"
	"github.com/rackspace/gophercloud/pagination"
	"github.com/rackspace/gophercloud/rackspace/db/v1/backups"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
	"github.com/rackspace/gophercloud/testhelper/fixture"
)

func TestInstanceList(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()

	fixture.SetupHandler(t, "/instances", "GET", "", listInstancesResp, 200)

	opts := &ListOpts{
		IncludeHA:       false,
		IncludeReplicas: false,
	}

	pages := 0
	err := List(fake.ServiceClient(), opts).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractInstances(page)
		if err != nil {
			return false, err
		}

		th.CheckDeepEquals(t, []Instance{*expectedInstance}, actual)
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetConfig(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL+"/configuration", "GET", "", getConfigResp, 200)

	config, err := GetDefaultConfig(fake.ServiceClient(), instanceID).Extract()

	expected := map[string]string{
		"basedir":                      "/usr",
		"connect_timeout":              "15",
		"datadir":                      "/var/lib/mysql",
		"default_storage_engine":       "innodb",
		"innodb_buffer_pool_instances": "1",
		"innodb_buffer_pool_size":      "175M",
		"innodb_checksum_algorithm":    "crc32",
		"innodb_data_file_path":        "ibdata1:10M:autoextend",
		"innodb_file_per_table":        "1",
		"innodb_io_capacity":           "200",
		"innodb_log_file_size":         "256M",
		"innodb_log_files_in_group":    "2",
		"innodb_open_files":            "8192",
		"innodb_thread_concurrency":    "0",
		"join_buffer_size":             "1M",
		"key_buffer_size":              "50M",
		"local-infile":                 "0",
		"log-error":                    "/var/log/mysql/mysqld.log",
		"max_allowed_packet":           "16M",
		"max_connect_errors":           "10000",
		"max_connections":              "40",
		"max_heap_table_size":          "16M",
		"myisam-recover":               "BACKUP",
		"open_files_limit":             "8192",
		"performance_schema":           "off",
		"pid_file":                     "/var/run/mysqld/mysqld.pid",
		"port":                         "3306",
		"query_cache_limit":            "1M",
		"query_cache_size":             "8M",
		"query_cache_type":             "1",
		"read_buffer_size":             "256K",
		"read_rnd_buffer_size":         "1M",
		"server_id":                    "1",
		"skip-external-locking":        "1",
		"skip_name_resolve":            "1",
		"sort_buffer_size":             "256K",
		"table_open_cache":             "4096",
		"thread_stack":                 "192K",
		"tmp_table_size":               "16M",
		"tmpdir":                       "/var/tmp",
		"user":                         "mysql",
		"wait_timeout":                 "3600",
	}

	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expected, config)
}

func TestAssociateWithConfigGroup(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL, "PUT", associateReq, associateResp, 202)

	res := AssociateWithConfigGroup(fake.ServiceClient(), instanceID, "{configGroupID}")
	th.AssertNoErr(t, res.Err)
}

func TestListBackups(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL+"/backups", "GET", "", listBackupsResp, 200)

	pages := 0

	err := ListBackups(fake.ServiceClient(), instanceID).EachPage(func(page pagination.Page) (bool, error) {
		pages++
		actual, err := backups.ExtractBackups(page)
		th.AssertNoErr(t, err)

		expected := []backups.Backup{
			backups.Backup{
				Created:     timeVal,
				Description: "Backup from Restored Instance",
				ID:          "87972694-4be2-40f5-83f8-501656e0032a",
				InstanceID:  "29af2cd9-0674-48ab-b87a-b160f00208e6",
				LocationRef: "http://localhost/path/to/backup",
				Name:        "restored_backup",
				ParentID:    "",
				Size:        0.141026,
				Status:      "COMPLETED",
				Updated:     timeVal,
				Datastore:   datastores.DatastorePartial{Version: "5.1", Type: "MySQL", VersionID: "20000000-0000-0000-0000-000000000002"},
			},
		}

		th.AssertDeepEquals(t, expected, actual)
		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestCreateReplica(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _rootURL, "POST", createReplicaReq, createReplicaResp, 200)

	opts := CreateOpts{
		Name:      "t2s1_ALT_GUEST",
		FlavorRef: "9",
		Size:      1,
		ReplicaOf: "6bdca2fc-418e-40bd-a595-62abda61862d",
	}

	replica, err := Create(fake.ServiceClient(), opts).Extract()
	th.AssertNoErr(t, err)
	th.AssertDeepEquals(t, expectedReplica, replica)
}

func TestListReplicas(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, _rootURL, "GET", "", listReplicasResp, 200)

	pages := 0
	err := List(fake.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		pages++

		actual, err := ExtractInstances(page)
		if err != nil {
			return false, err
		}

		expected := []Instance{
			Instance{
				Status: "ACTIVE",
				Name:   "t1s1_ALT_GUEST",
				Links: []gophercloud.Link{
					gophercloud.Link{Rel: "self", Href: "https://ord.databases.api.rackspacecloud.com/v1.0/1234/instances/3c691f06-bf9a-4618-b7ec-2817ce0cf254"},
					gophercloud.Link{Rel: "bookmark", Href: "https://ord.databases.api.rackspacecloud.com/instances/3c691f06-bf9a-4618-b7ec-2817ce0cf254"},
				},
				ID:        "3c691f06-bf9a-4618-b7ec-2817ce0cf254",
				IP:        []string{"10.0.0.3"},
				Volume:    os.Volume{Size: 1},
				Flavor:    flavors.Flavor{ID: "9"},
				Datastore: datastores.DatastorePartial{Version: "5.6", Type: "mysql"},
				ReplicaOf: &Instance{
					ID: "8b499b45-52d6-402d-b398-f9d8f279c69a",
				},
			},
		}

		th.CheckDeepEquals(t, expected, actual)

		return true, nil
	})

	th.AssertNoErr(t, err)
	th.AssertEquals(t, 1, pages)
}

func TestGetReplica(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL, "GET", "", getReplicaResp, 200)

	replica, err := Get(fake.ServiceClient(), instanceID).Extract()
	th.AssertNoErr(t, err)

	expectedReplica := &Instance{
		Status:  "ACTIVE",
		Updated: timeVal,
		Name:    "t1_ALT_GUEST",
		Created: timeVal,
		IP: []string{
			"10.0.0.2",
		},
		Replicas: []Instance{
			Instance{ID: "3c691f06-bf9a-4618-b7ec-2817ce0cf254"},
		},
		ID: "8b499b45-52d6-402d-b398-f9d8f279c69a",
		Volume: os.Volume{
			Used: 0.54,
			Size: 1,
		},
		Flavor: flavors.Flavor{ID: "9"},
		Datastore: datastores.DatastorePartial{
			Version: "5.6",
			Type:    "mysql",
		},
	}

	th.AssertDeepEquals(t, replica, expectedReplica)
}

func TestDetachReplica(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	fixture.SetupHandler(t, resURL, "PATCH", detachReq, "", 202)

	err := DetachReplica(fake.ServiceClient(), instanceID).ExtractErr()
	th.AssertNoErr(t, err)
}
