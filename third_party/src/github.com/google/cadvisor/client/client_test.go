// Copyright 2014 Google Inc. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

package cadvisor

import (
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"reflect"
	"testing"

	"github.com/google/cadvisor/info"
)

func testGetJsonData(
	strRep string,
	emptyData interface{},
	f func() (interface{}, error),
) error {
	err := json.Unmarshal([]byte(strRep), emptyData)
	if err != nil {
		return fmt.Errorf("invalid json input: %v", err)
	}
	reply, err := f()
	if err != nil {
		return fmt.Errorf("unable to retrieve data: %v", err)
	}
	if !reflect.DeepEqual(reply, emptyData) {
		return fmt.Errorf("retrieved wrong data: %+v != %+v", reply, emptyData)
	}
	return nil
}

func cadvisorTestClient(path, reply string) (*Client, *httptest.Server, error) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == path {
			fmt.Fprint(w, reply)
		} else if r.URL.Path == "/api/v1.0/machine" {
			fmt.Fprint(w, `{"num_cores":8,"memory_capacity":31625871360}`)
		} else {
			w.WriteHeader(http.StatusNotFound)
			fmt.Fprintf(w, "Page not found.")
		}
	}))
	client, err := NewClient(ts.URL)
	if err != nil {
		ts.Close()
		return nil, nil, err
	}
	return client, ts, err
}

func TestGetMachineinfo(t *testing.T) {
	respStr := `{"num_cores":8,"memory_capacity":31625871360}`
	client, server, err := cadvisorTestClient("/api/v1.0/machine", respStr)
	if err != nil {
		t.Fatalf("unable to get a client %v", err)
	}
	defer server.Close()
	err = testGetJsonData(respStr, &info.MachineInfo{}, func() (interface{}, error) {
		return client.MachineInfo()
	})
	if err != nil {
		t.Fatal(err)
	}
}

func TestGetContainerInfo(t *testing.T) {
	respStr := `
{
  "name": "%v",
  "spec": {
    "cpu": {
      "limit": 18446744073709551000,
      "max_limit": 0,
      "mask": {
        "data": [
          18446744073709551000
        ]
      }
    },
    "memory": {
      "limit": 18446744073709551000,
      "swap_limit": 18446744073709551000
    }
  },
  "stats": [
    {
      "timestamp": "2014-06-13T01:03:26.434981825Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:27.538394608Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:28.640302072Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:29.74247308Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:30.844494537Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:31.946757066Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:33.050214062Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:34.15222186Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:35.25417391Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:36.355902169Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:37.457585928Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:38.559417379Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:39.662978029Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:40.764671232Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    },
    {
      "timestamp": "2014-06-13T01:03:41.866456459Z",
      "cpu": {
        "usage": {
          "total": 56896502,
          "per_cpu": [
            20479682,
            13579420,
            6025040,
            2255123,
            3635661,
            2489368,
            5158288,
            3273920
          ],
          "user": 10000000,
          "system": 30000000
        },
        "load": 0
      },
      "memory": {
        "usage": 495616,
        "container_data": {
          "pgfault": 2279
        },
        "hierarchical_data": {
          "pgfault": 2279
        }
      }
    }
  ],
  "stats_summary": {
    "max_memory_usage": 495616,
    "samples": [
      {
        "timestamp": "2014-06-13T01:03:27.538394608Z",
        "duration": 1103412783,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:28.640302072Z",
        "duration": 1101907464,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:29.74247308Z",
        "duration": 1102171008,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:30.844494537Z",
        "duration": 1102021457,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:31.946757066Z",
        "duration": 1102262529,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:33.050214062Z",
        "duration": 1103456996,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:34.15222186Z",
        "duration": 1102007798,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:35.25417391Z",
        "duration": 1101952050,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:36.355902169Z",
        "duration": 1101728259,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:37.457585928Z",
        "duration": 1101683759,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:38.559417379Z",
        "duration": 1101831451,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:39.662978029Z",
        "duration": 1103560650,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:40.764671232Z",
        "duration": 1101693203,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      },
      {
        "timestamp": "2014-06-13T01:03:41.866456459Z",
        "duration": 1101785227,
        "cpu": {
          "usage": 0
        },
        "memory": {
          "usage": 495616
        }
      }
    ],
    "memory_usage_percentiles": [
      {
        "percentage": 50,
        "value": 495616
      },
      {
        "percentage": 80,
        "value": 495616
      },
      {
        "percentage": 90,
        "value": 495616
      },
      {
        "percentage": 95,
        "value": 495616
      },
      {
        "percentage": 99,
        "value": 495616
      }
    ],
    "cpu_usage_percentiles": [
      {
        "percentage": 50,
        "value": 0
      },
      {
        "percentage": 80,
        "value": 0
      },
      {
        "percentage": 90,
        "value": 0
      },
      {
        "percentage": 95,
        "value": 0
      },
      {
        "percentage": 99,
        "value": 0
      }
    ]
  }
}
`
	containerName := "/some/container"
	respStr = fmt.Sprintf(respStr, containerName)
	client, server, err := cadvisorTestClient(fmt.Sprintf("/api/v1.0/containers%v", containerName), respStr)
	if err != nil {
		t.Fatalf("unable to get a client %v", err)
	}
	defer server.Close()
	err = testGetJsonData(respStr, &info.ContainerInfo{}, func() (interface{}, error) {
		return client.ContainerInfo(containerName)
	})
	if err != nil {
		t.Fatal(err)
	}
}
