package instances

import (
	"fmt"
	"time"

	"github.com/rackspace/gophercloud"
	"github.com/rackspace/gophercloud/openstack/db/v1/datastores"
	"github.com/rackspace/gophercloud/openstack/db/v1/flavors"
	os "github.com/rackspace/gophercloud/openstack/db/v1/instances"
)

var (
	timestamp  = "2015-11-12T14:22:42Z"
	timeVal, _ = time.Parse(time.RFC3339, timestamp)
)

var instance = `
{
  "created": "` + timestamp + `",
  "datastore": {
    "type": "mysql",
    "version": "5.6"
  },
  "flavor": {
    "id": "1",
    "links": [
      {
        "href": "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1",
        "rel": "self"
      },
      {
        "href": "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1",
        "rel": "bookmark"
      }
    ]
  },
  "links": [
    {
      "href": "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1",
      "rel": "self"
    }
  ],
  "hostname": "e09ad9a3f73309469cf1f43d11e79549caf9acf2.rackspaceclouddb.com",
  "id": "{instanceID}",
  "name": "json_rack_instance",
  "status": "BUILD",
  "updated": "` + timestamp + `",
  "volume": {
    "size": 2
  }
}
`

var createReq = `
{
  "instance": {
    "databases": [
      {
        "character_set": "utf8",
        "collate": "utf8_general_ci",
        "name": "sampledb"
      },
      {
        "name": "nextround"
      }
    ],
    "flavorRef": "1",
    "name": "json_rack_instance",
    "users": [
      {
        "databases": [
          {
            "name": "sampledb"
          }
        ],
        "name": "demouser",
        "password": "demopassword"
      }
    ],
    "volume": {
      "size": 2
    },
    "restorePoint": {
      "backupRef": "1234567890"
    }
  }
}
`

var createReplicaReq = `
{
  "instance": {
    "volume": {
      "size": 1
    },
    "flavorRef": "9",
    "name": "t2s1_ALT_GUEST",
    "replica_of": "6bdca2fc-418e-40bd-a595-62abda61862d"
  }
}
`

var createReplicaResp = `
{
  "instance": {
    "status": "BUILD",
    "updated": "` + timestamp + `",
    "name": "t2s1_ALT_GUEST",
    "links": [
      {
        "href": "https://ord.databases.api.rackspacecloud.com/v1.0/5919009/instances/8367c312-7c40-4a66-aab1-5767478914fc",
        "rel": "self"
      },
      {
        "href": "https://ord.databases.api.rackspacecloud.com/instances/8367c312-7c40-4a66-aab1-5767478914fc",
        "rel": "bookmark"
      }
    ],
    "created": "` + timestamp + `",
    "id": "8367c312-7c40-4a66-aab1-5767478914fc",
    "volume": {
      "size": 1
    },
    "flavor": {
      "id": "9"
    },
    "datastore": {
      "version": "5.6",
      "type": "mysql"
    },
    "replica_of": {
      "id": "6bdca2fc-418e-40bd-a595-62abda61862d"
    }
  }
}
`

var listReplicasResp = `
{
  "instances": [
    {
      "status": "ACTIVE",
      "name": "t1s1_ALT_GUEST",
      "links": [
        {
          "href": "https://ord.databases.api.rackspacecloud.com/v1.0/1234/instances/3c691f06-bf9a-4618-b7ec-2817ce0cf254",
          "rel": "self"
        },
        {
          "href": "https://ord.databases.api.rackspacecloud.com/instances/3c691f06-bf9a-4618-b7ec-2817ce0cf254",
          "rel": "bookmark"
        }
      ],
      "ip": [
        "10.0.0.3"
      ],
      "id": "3c691f06-bf9a-4618-b7ec-2817ce0cf254",
      "volume": {
        "size": 1
      },
      "flavor": {
        "id": "9"
      },
      "datastore": {
        "version": "5.6",
        "type": "mysql"
      },
      "replica_of": {
        "id": "8b499b45-52d6-402d-b398-f9d8f279c69a"
      }
    }
  ]
}
`

var getReplicaResp = `
{
  "instance": {
    "status": "ACTIVE",
    "updated": "` + timestamp + `",
    "name": "t1_ALT_GUEST",
    "created": "` + timestamp + `",
    "ip": [
      "10.0.0.2"
    ],
    "replicas": [
      {
        "id": "3c691f06-bf9a-4618-b7ec-2817ce0cf254"
      }
    ],
    "id": "8b499b45-52d6-402d-b398-f9d8f279c69a",
    "volume": {
      "used": 0.54,
      "size": 1
    },
    "flavor": {
      "id": "9"
    },
    "datastore": {
      "version": "5.6",
      "type": "mysql"
    }
  }
}
`

var detachReq = `
{
  "instance": {
    "replica_of": "",
    "slave_of": ""
  }
}
`

var getConfigResp = `
{
  "instance": {
    "configuration": {
      "basedir": "/usr",
      "connect_timeout": "15",
      "datadir": "/var/lib/mysql",
      "default_storage_engine": "innodb",
      "innodb_buffer_pool_instances": "1",
      "innodb_buffer_pool_size": "175M",
      "innodb_checksum_algorithm": "crc32",
      "innodb_data_file_path": "ibdata1:10M:autoextend",
      "innodb_file_per_table": "1",
      "innodb_io_capacity": "200",
      "innodb_log_file_size": "256M",
      "innodb_log_files_in_group": "2",
      "innodb_open_files": "8192",
      "innodb_thread_concurrency": "0",
      "join_buffer_size": "1M",
      "key_buffer_size": "50M",
      "local-infile": "0",
      "log-error": "/var/log/mysql/mysqld.log",
      "max_allowed_packet": "16M",
      "max_connect_errors": "10000",
      "max_connections": "40",
      "max_heap_table_size": "16M",
      "myisam-recover": "BACKUP",
      "open_files_limit": "8192",
      "performance_schema": "off",
      "pid_file": "/var/run/mysqld/mysqld.pid",
      "port": "3306",
      "query_cache_limit": "1M",
      "query_cache_size": "8M",
      "query_cache_type": "1",
      "read_buffer_size": "256K",
      "read_rnd_buffer_size": "1M",
      "server_id": "1",
      "skip-external-locking": "1",
      "skip_name_resolve": "1",
      "sort_buffer_size": "256K",
      "table_open_cache": "4096",
      "thread_stack": "192K",
      "tmp_table_size": "16M",
      "tmpdir": "/var/tmp",
      "user": "mysql",
      "wait_timeout": "3600"
    }
  }
}
`

var associateReq = `{"instance": {"configuration": "{configGroupID}"}}`

var listBackupsResp = `
{
  "backups": [
    {
      "status": "COMPLETED",
      "updated": "` + timestamp + `",
      "description": "Backup from Restored Instance",
      "datastore": {
        "version": "5.1",
        "type": "MySQL",
        "version_id": "20000000-0000-0000-0000-000000000002"
      },
      "id": "87972694-4be2-40f5-83f8-501656e0032a",
      "size": 0.141026,
      "name": "restored_backup",
      "created": "` + timestamp + `",
      "instance_id": "29af2cd9-0674-48ab-b87a-b160f00208e6",
      "parent_id": null,
      "locationRef": "http://localhost/path/to/backup"
    }
  ]
}
`

var (
	createResp        = fmt.Sprintf(`{"instance":%s}`, instance)
	getResp           = fmt.Sprintf(`{"instance":%s}`, instance)
	associateResp     = fmt.Sprintf(`{"instance":%s}`, instance)
	listInstancesResp = fmt.Sprintf(`{"instances":[%s]}`, instance)
)

var instanceID = "{instanceID}"

var expectedInstance = &Instance{
	Created:   timeVal,
	Updated:   timeVal,
	Datastore: datastores.DatastorePartial{Type: "mysql", Version: "5.6"},
	Flavor: flavors.Flavor{
		ID: "1",
		Links: []gophercloud.Link{
			gophercloud.Link{Href: "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1", Rel: "self"},
			gophercloud.Link{Href: "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1", Rel: "bookmark"},
		},
	},
	Hostname: "e09ad9a3f73309469cf1f43d11e79549caf9acf2.rackspaceclouddb.com",
	ID:       instanceID,
	Links: []gophercloud.Link{
		gophercloud.Link{Href: "https://ord.databases.api.rackspacecloud.com/v1.0/1234/flavors/1", Rel: "self"},
	},
	Name:   "json_rack_instance",
	Status: "BUILD",
	Volume: os.Volume{Size: 2},
}

var expectedReplica = &Instance{
	Status:  "BUILD",
	Updated: timeVal,
	Name:    "t2s1_ALT_GUEST",
	Links: []gophercloud.Link{
		gophercloud.Link{Rel: "self", Href: "https://ord.databases.api.rackspacecloud.com/v1.0/5919009/instances/8367c312-7c40-4a66-aab1-5767478914fc"},
		gophercloud.Link{Rel: "bookmark", Href: "https://ord.databases.api.rackspacecloud.com/instances/8367c312-7c40-4a66-aab1-5767478914fc"},
	},
	Created:   timeVal,
	ID:        "8367c312-7c40-4a66-aab1-5767478914fc",
	Volume:    os.Volume{Size: 1},
	Flavor:    flavors.Flavor{ID: "9"},
	Datastore: datastores.DatastorePartial{Version: "5.6", Type: "mysql"},
	ReplicaOf: &Instance{
		ID: "6bdca2fc-418e-40bd-a595-62abda61862d",
	},
}
