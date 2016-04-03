package stacks

import (
	"testing"

	os "github.com/rackspace/gophercloud/openstack/orchestration/v1/stacks"
	"github.com/rackspace/gophercloud/pagination"
	th "github.com/rackspace/gophercloud/testhelper"
	fake "github.com/rackspace/gophercloud/testhelper/client"
)

func TestCreateStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreateSuccessfully(t, CreateOutput)

	createOpts := os.CreateOpts{
		Name:    "stackcreated",
		Timeout: 60,
		Template: `{
      "outputs": {
        "db_host": {
          "value": {
            "get_attr": [
            "db",
            "hostname"
            ]
          }
        }
      },
      "heat_template_version": "2014-10-16",
      "description": "HEAT template for creating a Cloud Database.\n",
      "parameters": {
        "db_name": {
          "default": "wordpress",
          "type": "string",
          "description": "the name for the database",
          "constraints": [
          {
            "length": {
              "max": 64,
              "min": 1
            },
            "description": "must be between 1 and 64 characters"
          },
          {
            "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
            "description": "must begin with a letter and contain only alphanumeric characters."
          }
          ]
        },
        "db_instance_name": {
          "default": "Cloud_DB",
          "type": "string",
          "description": "the database instance name"
        },
        "db_username": {
          "default": "admin",
          "hidden": true,
          "type": "string",
          "description": "database admin account username",
          "constraints": [
          {
            "length": {
              "max": 16,
              "min": 1
                },
              "description": "must be between 1 and 16 characters"
            },
            {
              "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
              "description": "must begin with a letter and contain only alphanumeric characters."
            }
          ]
          },
          "db_volume_size": {
            "default": 30,
            "type": "number",
            "description": "database volume size (in GB)",
            "constraints": [
            {
              "range": {
                "max": 1024,
                "min": 1
              },
              "description": "must be between 1 and 1024 GB"
            }
            ]
          },
          "db_flavor": {
            "default": "1GB Instance",
            "type": "string",
            "description": "database instance size",
            "constraints": [
            {
              "description": "must be a valid cloud database flavor",
              "allowed_values": [
              "1GB Instance",
              "2GB Instance",
              "4GB Instance",
              "8GB Instance",
              "16GB Instance"
              ]
            }
            ]
          },
        "db_password": {
          "default": "admin",
          "hidden": true,
          "type": "string",
          "description": "database admin account password",
          "constraints": [
          {
            "length": {
              "max": 41,
              "min": 1
            },
            "description": "must be between 1 and 14 characters"
          },
          {
            "allowed_pattern": "[a-zA-Z0-9]*",
            "description": "must contain only alphanumeric characters."
          }
          ]
        }
      },
      "resources": {
        "db": {
          "type": "OS::Trove::Instance",
          "properties": {
            "flavor": {
              "get_param": "db_flavor"
            },
            "size": {
              "get_param": "db_volume_size"
            },
            "users": [
            {
              "password": {
                "get_param": "db_password"
              },
              "name": {
                "get_param": "db_username"
              },
              "databases": [
              {
                "get_param": "db_name"
              }
              ]
            }
            ],
            "name": {
              "get_param": "db_instance_name"
            },
            "databases": [
            {
              "name": {
                "get_param": "db_name"
              }
            }
            ]
          }
        }
      }
    }`,
		DisableRollback: os.Disable,
	}
	actual, err := Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestCreateStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreateSuccessfully(t, CreateOutput)

	createOpts := os.CreateOpts{
		Name:            "stackcreated",
		Timeout:         60,
		TemplateOpts:    new(os.Template),
		DisableRollback: os.Disable,
	}
	createOpts.TemplateOpts.Bin = []byte(`{
		    "outputs": {
		        "db_host": {
		            "value": {
		                "get_attr": [
		                    "db",
		                    "hostname"
		                ]
		            }
		        }
		    },
		    "heat_template_version": "2014-10-16",
		    "description": "HEAT template for creating a Cloud Database.\n",
		    "parameters": {
		        "db_name": {
		            "default": "wordpress",
		            "type": "string",
		            "description": "the name for the database",
		            "constraints": [
		                {
		                    "length": {
		                        "max": 64,
		                        "min": 1
		                    },
		                    "description": "must be between 1 and 64 characters"
		                },
		                {
		                    "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
		                    "description": "must begin with a letter and contain only alphanumeric characters."
		                }
		            ]
		        },
		        "db_instance_name": {
		            "default": "Cloud_DB",
		            "type": "string",
		            "description": "the database instance name"
		        },
		        "db_username": {
		            "default": "admin",
		            "hidden": true,
		            "type": "string",
		            "description": "database admin account username",
		            "constraints": [
		                {
		                    "length": {
		                        "max": 16,
		                        "min": 1
		                    },
		                    "description": "must be between 1 and 16 characters"
		                },
		                {
		                    "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
		                    "description": "must begin with a letter and contain only alphanumeric characters."
		                }
		            ]
		        },
		        "db_volume_size": {
		            "default": 30,
		            "type": "number",
		            "description": "database volume size (in GB)",
		            "constraints": [
		                {
		                    "range": {
		                        "max": 1024,
		                        "min": 1
		                    },
		                    "description": "must be between 1 and 1024 GB"
		                }
		            ]
		        },
		        "db_flavor": {
		            "default": "1GB Instance",
		            "type": "string",
		            "description": "database instance size",
		            "constraints": [
		                {
		                    "description": "must be a valid cloud database flavor",
		                    "allowed_values": [
		                        "1GB Instance",
		                        "2GB Instance",
		                        "4GB Instance",
		                        "8GB Instance",
		                        "16GB Instance"
		                    ]
		                }
		            ]
		        },
		        "db_password": {
		            "default": "admin",
		            "hidden": true,
		            "type": "string",
		            "description": "database admin account password",
		            "constraints": [
		                {
		                    "length": {
		                        "max": 41,
		                        "min": 1
		                    },
		                    "description": "must be between 1 and 14 characters"
		                },
		                {
		                    "allowed_pattern": "[a-zA-Z0-9]*",
		                    "description": "must contain only alphanumeric characters."
		                }
		            ]
		        }
		    },
		    "resources": {
		        "db": {
		            "type": "OS::Trove::Instance",
		            "properties": {
		                "flavor": {
		                    "get_param": "db_flavor"
		                },
		                "size": {
		                    "get_param": "db_volume_size"
		                },
		                "users": [
		                    {
		                        "password": {
		                            "get_param": "db_password"
		                        },
		                        "name": {
		                            "get_param": "db_username"
		                        },
		                        "databases": [
		                            {
		                                "get_param": "db_name"
		                            }
		                        ]
		                    }
		                ],
		                "name": {
		                    "get_param": "db_instance_name"
		                },
		                "databases": [
		                    {
		                        "name": {
		                            "get_param": "db_name"
		                        }
		                    }
		                ]
		            }
		        }
		    }
		}`)
	actual, err := Create(fake.ServiceClient(), createOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAdoptStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreateSuccessfully(t, CreateOutput)

	adoptOpts := os.AdoptOpts{
		AdoptStackData: `{\"environment\":{\"parameters\":{}},    \"status\":\"COMPLETE\",\"name\": \"trovestack\",\n  \"template\": {\n    \"outputs\": {\n      \"db_host\": {\n        \"value\": {\n          \"get_attr\": [\n            \"db\",\n            \"hostname\"\n          ]\n        }\n      }\n    },\n    \"heat_template_version\": \"2014-10-16\",\n    \"description\": \"HEAT template for creating a Cloud Database.\\n\",\n    \"parameters\": {\n      \"db_instance_name\": {\n        \"default\": \"Cloud_DB\",\n        \"type\": \"string\",\n        \"description\": \"the database instance name\"\n      },\n      \"db_flavor\": {\n        \"default\": \"1GB Instance\",\n        \"type\": \"string\",\n        \"description\": \"database instance size\",\n        \"constraints\": [\n          {\n            \"description\": \"must be a valid cloud database flavor\",\n            \"allowed_values\": [\n              \"1GB Instance\",\n              \"2GB Instance\",\n              \"4GB Instance\",\n              \"8GB Instance\",\n              \"16GB Instance\"\n            ]\n          }\n        ]\n      },\n      \"db_password\": {\n        \"default\": \"admin\",\n        \"hidden\": true,\n        \"type\": \"string\",\n        \"description\": \"database admin account password\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 41,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 14 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z0-9]*\",\n            \"description\": \"must contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_name\": {\n        \"default\": \"wordpress\",\n        \"type\": \"string\",\n        \"description\": \"the name for the database\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 64,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 64 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z][a-zA-Z0-9]*\",\n            \"description\": \"must begin with a letter and contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_username\": {\n        \"default\": \"admin\",\n        \"hidden\": true,\n        \"type\": \"string\",\n        \"description\": \"database admin account username\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 16,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 16 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z][a-zA-Z0-9]*\",\n            \"description\": \"must begin with a letter and contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_volume_size\": {\n        \"default\": 30,\n        \"type\": \"number\",\n        \"description\": \"database volume size (in GB)\",\n        \"constraints\": [\n          {\n            \"range\": {\n              \"max\": 1024,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 1024 GB\"\n          }\n        ]\n      }\n    },\n    \"resources\": {\n      \"db\": {\n        \"type\": \"OS::Trove::Instance\",\n        \"properties\": {\n          \"flavor\": {\n            \"get_param\": \"db_flavor\"\n          },\n          \"databases\": [\n            {\n              \"name\": {\n                \"get_param\": \"db_name\"\n              }\n            }\n          ],\n          \"users\": [\n            {\n              \"password\": {\n                \"get_param\": \"db_password\"\n              },\n              \"name\": {\n                \"get_param\": \"db_username\"\n              },\n              \"databases\": [\n                {\n                  \"get_param\": \"db_name\"\n                }\n              ]\n            }\n          ],\n          \"name\": {\n            \"get_param\": \"db_instance_name\"\n          },\n          \"size\": {\n            \"get_param\": \"db_volume_size\"\n          }\n        }\n      }\n    }\n  },\n  \"action\": \"CREATE\",\n  \"id\": \"exxxxd-7xx5-4xxb-bxx2-cxxxxxx5\",\n  \"resources\": {\n    \"db\": {\n      \"status\": \"COMPLETE\",\n      \"name\": \"db\",\n      \"resource_data\": {},\n      \"resource_id\": \"exxxx2-9xx0-4xxxb-bxx2-dxxxxxx4\",\n      \"action\": \"CREATE\",\n      \"type\": \"OS::Trove::Instance\",\n      \"metadata\": {}\n    }\n  }\n},`,
		Name:           "stackadopted",
		Timeout:        60,
		Template: `{
      "outputs": {
        "db_host": {
          "value": {
            "get_attr": [
            "db",
            "hostname"
            ]
          }
        }
      },
      "heat_template_version": "2014-10-16",
      "description": "HEAT template for creating a Cloud Database.\n",
      "parameters": {
        "db_name": {
          "default": "wordpress",
          "type": "string",
          "description": "the name for the database",
          "constraints": [
          {
            "length": {
              "max": 64,
              "min": 1
            },
            "description": "must be between 1 and 64 characters"
          },
          {
            "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
            "description": "must begin with a letter and contain only alphanumeric characters."
          }
          ]
        },
        "db_instance_name": {
          "default": "Cloud_DB",
          "type": "string",
          "description": "the database instance name"
        },
        "db_username": {
          "default": "admin",
          "hidden": true,
          "type": "string",
          "description": "database admin account username",
          "constraints": [
          {
            "length": {
              "max": 16,
              "min": 1
            },
            "description": "must be between 1 and 16 characters"
          },
          {
            "allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
            "description": "must begin with a letter and contain only alphanumeric characters."
          }
          ]
        },
        "db_volume_size": {
          "default": 30,
          "type": "number",
          "description": "database volume size (in GB)",
          "constraints": [
          {
            "range": {
              "max": 1024,
              "min": 1
            },
            "description": "must be between 1 and 1024 GB"
          }
          ]
        },
        "db_flavor": {
          "default": "1GB Instance",
          "type": "string",
          "description": "database instance size",
          "constraints": [
          {
            "description": "must be a valid cloud database flavor",
            "allowed_values": [
            "1GB Instance",
            "2GB Instance",
            "4GB Instance",
            "8GB Instance",
            "16GB Instance"
            ]
          }
          ]
        },
        "db_password": {
          "default": "admin",
          "hidden": true,
          "type": "string",
          "description": "database admin account password",
          "constraints": [
          {
            "length": {
              "max": 41,
              "min": 1
            },
            "description": "must be between 1 and 14 characters"
          },
          {
            "allowed_pattern": "[a-zA-Z0-9]*",
            "description": "must contain only alphanumeric characters."
          }
          ]
        }
      },
      "resources": {
        "db": {
          "type": "OS::Trove::Instance",
          "properties": {
            "flavor": {
              "get_param": "db_flavor"
            },
            "size": {
              "get_param": "db_volume_size"
            },
            "users": [
            {
              "password": {
                "get_param": "db_password"
              },
              "name": {
                "get_param": "db_username"
              },
              "databases": [
              {
                "get_param": "db_name"
              }
              ]
            }
            ],
            "name": {
              "get_param": "db_instance_name"
            },
            "databases": [
            {
              "name": {
                "get_param": "db_name"
              }
            }
            ]
          }
        }
      }
    }`,
		DisableRollback: os.Disable,
	}
	actual, err := Adopt(fake.ServiceClient(), adoptOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAdoptStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleCreateSuccessfully(t, CreateOutput)
	template := new(os.Template)
	template.Bin = []byte(`{
  "outputs": {
	"db_host": {
	  "value": {
		"get_attr": [
		"db",
		"hostname"
		]
	  }
	}
  },
  "heat_template_version": "2014-10-16",
  "description": "HEAT template for creating a Cloud Database.\n",
  "parameters": {
	"db_name": {
	  "default": "wordpress",
	  "type": "string",
	  "description": "the name for the database",
	  "constraints": [
	  {
		"length": {
		  "max": 64,
		  "min": 1
		},
		"description": "must be between 1 and 64 characters"
	  },
	  {
		"allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
		"description": "must begin with a letter and contain only alphanumeric characters."
	  }
	  ]
	},
	"db_instance_name": {
	  "default": "Cloud_DB",
	  "type": "string",
	  "description": "the database instance name"
	},
	"db_username": {
	  "default": "admin",
	  "hidden": true,
	  "type": "string",
	  "description": "database admin account username",
	  "constraints": [
	  {
		"length": {
		  "max": 16,
		  "min": 1
		},
		"description": "must be between 1 and 16 characters"
	  },
	  {
		"allowed_pattern": "[a-zA-Z][a-zA-Z0-9]*",
		"description": "must begin with a letter and contain only alphanumeric characters."
	  }
	  ]
	},
	"db_volume_size": {
	  "default": 30,
	  "type": "number",
	  "description": "database volume size (in GB)",
	  "constraints": [
	  {
		"range": {
		  "max": 1024,
		  "min": 1
		},
		"description": "must be between 1 and 1024 GB"
	  }
	  ]
	},
	"db_flavor": {
	  "default": "1GB Instance",
	  "type": "string",
	  "description": "database instance size",
	  "constraints": [
	  {
		"description": "must be a valid cloud database flavor",
		"allowed_values": [
		"1GB Instance",
		"2GB Instance",
		"4GB Instance",
		"8GB Instance",
		"16GB Instance"
		]
	  }
	  ]
	},
	"db_password": {
	  "default": "admin",
	  "hidden": true,
	  "type": "string",
	  "description": "database admin account password",
	  "constraints": [
	  {
		"length": {
		  "max": 41,
		  "min": 1
		},
		"description": "must be between 1 and 14 characters"
	  },
	  {
		"allowed_pattern": "[a-zA-Z0-9]*",
		"description": "must contain only alphanumeric characters."
	  }
	  ]
	}
  },
  "resources": {
	"db": {
	  "type": "OS::Trove::Instance",
	  "properties": {
		"flavor": {
		  "get_param": "db_flavor"
		},
		"size": {
		  "get_param": "db_volume_size"
		},
		"users": [
		{
		  "password": {
			"get_param": "db_password"
		  },
		  "name": {
			"get_param": "db_username"
		  },
		  "databases": [
		  {
			"get_param": "db_name"
		  }
		  ]
		}
		],
		"name": {
		  "get_param": "db_instance_name"
		},
		"databases": [
		{
		  "name": {
			"get_param": "db_name"
		  }
		}
		]
	  }
	}
  }
}`)

	adoptOpts := os.AdoptOpts{
		AdoptStackData:  `{\"environment\":{\"parameters\":{}},    \"status\":\"COMPLETE\",\"name\": \"trovestack\",\n  \"template\": {\n    \"outputs\": {\n      \"db_host\": {\n        \"value\": {\n          \"get_attr\": [\n            \"db\",\n            \"hostname\"\n          ]\n        }\n      }\n    },\n    \"heat_template_version\": \"2014-10-16\",\n    \"description\": \"HEAT template for creating a Cloud Database.\\n\",\n    \"parameters\": {\n      \"db_instance_name\": {\n        \"default\": \"Cloud_DB\",\n        \"type\": \"string\",\n        \"description\": \"the database instance name\"\n      },\n      \"db_flavor\": {\n        \"default\": \"1GB Instance\",\n        \"type\": \"string\",\n        \"description\": \"database instance size\",\n        \"constraints\": [\n          {\n            \"description\": \"must be a valid cloud database flavor\",\n            \"allowed_values\": [\n              \"1GB Instance\",\n              \"2GB Instance\",\n              \"4GB Instance\",\n              \"8GB Instance\",\n              \"16GB Instance\"\n            ]\n          }\n        ]\n      },\n      \"db_password\": {\n        \"default\": \"admin\",\n        \"hidden\": true,\n        \"type\": \"string\",\n        \"description\": \"database admin account password\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 41,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 14 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z0-9]*\",\n            \"description\": \"must contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_name\": {\n        \"default\": \"wordpress\",\n        \"type\": \"string\",\n        \"description\": \"the name for the database\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 64,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 64 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z][a-zA-Z0-9]*\",\n            \"description\": \"must begin with a letter and contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_username\": {\n        \"default\": \"admin\",\n        \"hidden\": true,\n        \"type\": \"string\",\n        \"description\": \"database admin account username\",\n        \"constraints\": [\n          {\n            \"length\": {\n              \"max\": 16,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 16 characters\"\n          },\n          {\n            \"allowed_pattern\": \"[a-zA-Z][a-zA-Z0-9]*\",\n            \"description\": \"must begin with a letter and contain only alphanumeric characters.\"\n          }\n        ]\n      },\n      \"db_volume_size\": {\n        \"default\": 30,\n        \"type\": \"number\",\n        \"description\": \"database volume size (in GB)\",\n        \"constraints\": [\n          {\n            \"range\": {\n              \"max\": 1024,\n              \"min\": 1\n            },\n            \"description\": \"must be between 1 and 1024 GB\"\n          }\n        ]\n      }\n    },\n    \"resources\": {\n      \"db\": {\n        \"type\": \"OS::Trove::Instance\",\n        \"properties\": {\n          \"flavor\": {\n            \"get_param\": \"db_flavor\"\n          },\n          \"databases\": [\n            {\n              \"name\": {\n                \"get_param\": \"db_name\"\n              }\n            }\n          ],\n          \"users\": [\n            {\n              \"password\": {\n                \"get_param\": \"db_password\"\n              },\n              \"name\": {\n                \"get_param\": \"db_username\"\n              },\n              \"databases\": [\n                {\n                  \"get_param\": \"db_name\"\n                }\n              ]\n            }\n          ],\n          \"name\": {\n            \"get_param\": \"db_instance_name\"\n          },\n          \"size\": {\n            \"get_param\": \"db_volume_size\"\n          }\n        }\n      }\n    }\n  },\n  \"action\": \"CREATE\",\n  \"id\": \"exxxxd-7xx5-4xxb-bxx2-cxxxxxx5\",\n  \"resources\": {\n    \"db\": {\n      \"status\": \"COMPLETE\",\n      \"name\": \"db\",\n      \"resource_data\": {},\n      \"resource_id\": \"exxxx2-9xx0-4xxxb-bxx2-dxxxxxx4\",\n      \"action\": \"CREATE\",\n      \"type\": \"OS::Trove::Instance\",\n      \"metadata\": {}\n    }\n  }\n},`,
		Name:            "stackadopted",
		Timeout:         60,
		TemplateOpts:    template,
		DisableRollback: os.Disable,
	}
	actual, err := Adopt(fake.ServiceClient(), adoptOpts).Extract()
	th.AssertNoErr(t, err)

	expected := CreateExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestListStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleListSuccessfully(t, os.FullListOutput)

	count := 0
	err := List(fake.ServiceClient(), nil).EachPage(func(page pagination.Page) (bool, error) {
		count++
		actual, err := os.ExtractStacks(page)
		th.AssertNoErr(t, err)

		th.CheckDeepEquals(t, os.ListExpected, actual)

		return true, nil
	})
	th.AssertNoErr(t, err)
	th.CheckEquals(t, count, 1)
}

func TestUpdateStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleUpdateSuccessfully(t)

	updateOpts := os.UpdateOpts{
		Template: `
    {
      "heat_template_version": "2013-05-23",
      "description": "Simple template to test heat commands",
      "parameters": {
        "flavor": {
          "default": "m1.tiny",
          "type": "string"
        }
      },
      "resources": {
        "hello_world": {
          "type":"OS::Nova::Server",
          "properties": {
            "key_name": "heat_key",
            "flavor": {
              "get_param": "flavor"
            },
            "image": "ad091b52-742f-469e-8f3c-fd81cadf0743",
            "user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
          }
        }
      }
    }`,
	}
	err := Update(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada", updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestUpdateStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleUpdateSuccessfully(t)

	updateOpts := os.UpdateOpts{
		TemplateOpts: new(os.Template),
	}
	updateOpts.TemplateOpts.Bin = []byte(`
		{
			"stack_name": "postman_stack",
			"template": {
				"heat_template_version": "2013-05-23",
				"description": "Simple template to test heat commands",
				"parameters": {
					"flavor": {
						"default": "m1.tiny",
						"type": "string"
					}
				},
				"resources": {
					"hello_world": {
						"type": "OS::Nova::Server",
						"properties": {
							"key_name": "heat_key",
							"flavor": {
								"get_param": "flavor"
							},
							"image": "ad091b52-742f-469e-8f3c-fd81cadf0743",
							"user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
						}
					}
				}
			}
		}`)
	err := Update(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada", updateOpts).ExtractErr()
	th.AssertNoErr(t, err)
}

func TestDeleteStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleDeleteSuccessfully(t)

	err := Delete(fake.ServiceClient(), "gophercloud-test-stack-2", "db6977b2-27aa-4775-9ae7-6213212d4ada").ExtractErr()
	th.AssertNoErr(t, err)
}

func TestPreviewStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandlePreviewSuccessfully(t, os.GetOutput)

	previewOpts := os.PreviewOpts{
		Name:    "stackcreated",
		Timeout: 60,
		Template: `
    {
      "stack_name": "postman_stack",
      "template": {
        "heat_template_version": "2013-05-23",
        "description": "Simple template to test heat commands",
        "parameters": {
          "flavor": {
            "default": "m1.tiny",
            "type": "string"
          }
        },
        "resources": {
          "hello_world": {
            "type":"OS::Nova::Server",
            "properties": {
              "key_name": "heat_key",
              "flavor": {
                "get_param": "flavor"
              },
              "image": "ad091b52-742f-469e-8f3c-fd81cadf0743",
              "user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
            }
          }
        }
      }
    }`,
		DisableRollback: os.Disable,
	}
	actual, err := Preview(fake.ServiceClient(), previewOpts).Extract()
	th.AssertNoErr(t, err)

	expected := os.PreviewExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestPreviewStackNewTemplateFormat(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandlePreviewSuccessfully(t, os.GetOutput)

	previewOpts := os.PreviewOpts{
		Name:            "stackcreated",
		Timeout:         60,
		TemplateOpts:    new(os.Template),
		DisableRollback: os.Disable,
	}
	previewOpts.TemplateOpts.Bin = []byte(`
		{
		    "stack_name": "postman_stack",
		    "template": {
		        "heat_template_version": "2013-05-23",
		        "description": "Simple template to test heat commands",
		        "parameters": {
		            "flavor": {
		                "default": "m1.tiny",
		                "type": "string"
		            }
		        },
		        "resources": {
		            "hello_world": {
		                "type": "OS::Nova::Server",
		                "properties": {
		                    "key_name": "heat_key",
		                    "flavor": {
		                        "get_param": "flavor"
		                    },
		                    "image": "ad091b52-742f-469e-8f3c-fd81cadf0743",
		                    "user_data": "#!/bin/bash -xv\necho \"hello world\" &gt; /root/hello-world.txt\n"
		                }
		            }
		        }
		    }
		}`)
	actual, err := Preview(fake.ServiceClient(), previewOpts).Extract()
	th.AssertNoErr(t, err)

	expected := os.PreviewExpected
	th.AssertDeepEquals(t, expected, actual)
}

func TestAbandonStack(t *testing.T) {
	th.SetupHTTP()
	defer th.TeardownHTTP()
	os.HandleAbandonSuccessfully(t, os.AbandonOutput)

	actual, err := Abandon(fake.ServiceClient(), "postman_stack", "16ef0584-4458-41eb-87c8-0dc8d5f66c8").Extract()
	th.AssertNoErr(t, err)

	expected := os.AbandonExpected
	th.AssertDeepEquals(t, expected, actual)
}
