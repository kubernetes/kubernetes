# Networking API

### List networks

`GET /networks`

List networks

**Example request**:

        GET /networks HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
          {
            "name": "none",
            "id": "8e4e55c6863ef4241c548c1c6fc77289045e9e5d5b5e4875401a675326981898",
            "type": "null",
            "endpoints": []
          },
          {
            "name": "host",
            "id": "062b6d9ea7913fde549e2d186ff0402770658f8c4e769958e1b943ff4e675011",
            "type": "host",
            "endpoints": []
          },
          {
            "name": "bridge",
            "id": "a87dd9a9d58f030962df1c15fb3fa142fbd9261339de458bc89be1895cef2c70",
            "type": "bridge",
            "endpoints": []
          }
        ]

Query Parameters:

-   **name** – Filter results with the given name
-   **partial-id** – Filter results using the partial network ID

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Create a Network

`POST /networks`

**Example request**

        POST /networks HTTP/1.1
        Content-Type: application/json

        {
          "name": "foo",
          "network_type": "",
          "options": {}
        }

**Example Response**

        HTTP/1.1 200 OK
        "32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653",

Status Codes:

-   **200** – no error
-   **400** – bad request
-   **500** – server error

### Get a network

`GET /networks/<network_id>`

Get a network

**Example request**:

        GET /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
          "name": "foo",
          "id": "32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653",
          "type": "bridge",
          "endpoints": []
        }

Status Codes:

-   **200** – no error
-   **404** – not found
-   **500** – server error

### List a networks endpoints

`GET /networks/<network_id>/endpoints`

**Example request**

        GET /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653/endpoints HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
            {
                "id": "7e0c116b882ee489a8a5345a2638c0129099aa47f4ba114edde34e75c1e4ae0d",
                "name": "/lonely_pasteur",
                "network": "foo"
            }
        ]

Query Parameters:

-   **name** – Filter results with the given name
-   **partial-id** – Filter results using the partial network ID

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Create an endpoint on a network

`POST /networks/<network_id>/endpoints`

**Example request**

        POST /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653/endpoints HTTP/1.1
        Content-Type: application/json

        {
          "name": "baz",
          "exposed_ports": [
            {
              "proto": 6,
              "port": 8080
            }
          ],
          "port_mapping": null
        }

**Example Response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        "b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a"

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Get an endpoint

`GET /networks/<network_id>/endpoints/<endpoint_id>`

**Example request**

        GET /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653/endpoints/b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
            "id": "b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a",
            "name": "baz",
            "network": "foo"
        }

Status Codes:

-   **200** – no error
-   **404** - not found
-   **500** – server error

### Join an endpoint to a container

`POST /networks/<network_id>/endpoints/<endpoint_id>/containers`

**Example request**

        POST /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653//endpoints/b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a/containers HTTP/1.1
        Content-Type: application/json

        {
            "container_id": "e76f406417031bd24c17aeb9bb2f5968b628b9fb6067da264b234544754bf857",
            "host_name": null,
            "domain_name": null,
            "hosts_path": null,
            "resolv_conf_path": null,
            "dns": null,
            "extra_hosts": null,
            "parent_updates": null,
            "use_default_sandbox": true
        }

**Example response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        "/var/run/docker/netns/e76f40641703"


Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **404** - not found
-   **500** – server error

### Detach an endpoint from a container

`DELETE /networks/<network_id>/endpoints/<endpoint_id>/containers/<container_id>`

**Example request**

        DELETE /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653/endpoints/b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a/containers/e76f406417031bd24c17aeb9bb2f5968b628b9fb6067da264b234544754bf857 HTTP/1.1
        Content-Type: application/json

**Example response**

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **404** - not found
-   **500** – server error


### Delete an endpoint

`DELETE /networks/<network_id>/endpoints/<endpoint_id>`

**Example request**

        DELETE /networks/32fbf63200e2897f5de72cb2a4b653e4b1a523b15116e96e3d73f7849e583653/endpoints/b18b795af8bad85cdd691ff24ffa2b08c02219d51992309dd120322689d2ab5a HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **404** - not found
-   **500** – server error

### Delete a network

`DELETE /networks/<network_id>`

Delete a network

**Example request**:

        DELETE /networks/0984d158bd8ae108e4d6bc8fcabedf51da9a174b32cc777026d4a29045654951 HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **404** – not found
-   **500** – server error

# Services API

### Publish a Service

`POST /services`

Publish a service

**Example Request**

        POST /services HTTP/1.1
        Content-Type: application/json

        {
          "name": "bar",
          "network_name": "foo",
          "exposed_ports": null,
          "port_mapping": null
        }

**Example Response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        "0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff"

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Get a Service

`GET /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff`

Get a service

**Example Request**:

        GET /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff HTTP/1.1

**Example Response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        {
          "name": "bar",
          "id": "0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff",
          "network": "foo"
        }

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **404** - not found
-   **500** – server error

### Attach a backend to a service

`POST /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff/backend`

Attach a backend to a service

**Example Request**:

        POST /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff/backend HTTP/1.1
        Content-Type: application/json

        {
          "container_id": "98c5241f9475e9efc17e7198e931fb48166010b80f96d48df204e251378ca547",
          "host_name": "",
          "domain_name": "",
          "hosts_path": "",
          "resolv_conf_path": "",
          "dns": null,
          "extra_hosts": null,
          "parent_updates": null,
          "use_default_sandbox": false
        }

**Example Response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        "/var/run/docker/netns/98c5241f9475"

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Get Backends for a Service

Get all backends for a given service

**Example Request**

        GET /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff/backend HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
          {
            "id": "98c5241f9475e9efc17e7198e931fb48166010b80f96d48df204e251378ca547"
          }
        ]

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### List Services

`GET /services`

List services

**Example request**:

        GET /services HTTP/1.1

**Example response**:

        HTTP/1.1 200 OK
        Content-Type: application/json

        [
          {
            "name": "/stupefied_stallman",
            "id": "c826b26bf736fb4a77db33f83562e59f9a770724e259ab9c3d50d948f8233ae4",
            "network": "bridge"
          },
          {
            "name": "bar",
            "id": "0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff",
            "network": "foo"
          }
        ]

Query Parameters:

-   **name** – Filter results with the given name
-   **partial-id** – Filter results using the partial network ID
-   **network** - Filter results by the given network

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Detach a Backend from a Service

`DELETE /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff/backend/98c5241f9475e9efc17e7198e931fb48166010b80f96d48df204e251378ca547`

Detach a backend from a service

**Example Request**

        DELETE /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff/backend/98c5241f9475e9efc17e7198e931fb48166010b80f96d48df204e251378ca547 HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error

### Un-Publish a Service

`DELETE /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff`

Unpublish a service

**Example Request**

        DELETE /services/0aee0899e6c5e903cf3ef2bdc28a1c9aaf639c8c8c331fa4ae26344d9e32c1ff HTTP/1.1

**Example Response**

        HTTP/1.1 200 OK

Status Codes:

-   **200** – no error
-   **400** – bad parameter
-   **500** – server error
