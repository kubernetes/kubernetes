echo "creating database"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE foo"

echo "creating retention policy"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY bar ON foo DURATION 300d REPLICATION 3 DEFAULT"

echo "inserting data"
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-01T00:00:00.000Z","fields": {"value": 1.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-01T08:00:00.000Z","fields": {"value": 1.2}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-01T16:00:00.000Z","fields": {"value": 1.3}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-01T16:59:00.000Z","fields": {"value": 1.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-02T00:00:00.000Z","fields": {"value": 2.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-02T08:00:00.000Z","fields": {"value": 2.2}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-02T16:00:00.000Z","fields": {"value": 2.3}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-02T16:59:00.000Z","fields": {"value": 2.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-03T00:00:00.000Z","fields": {"value": 3.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-03T08:00:00.000Z","fields": {"value": 3.2}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-03T16:00:00.000Z","fields": {"value": 3.3}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-03T16:59:00.000Z","fields": {"value": 3.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-04T00:00:00.000Z","fields": {"value": 4.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-04T08:00:00.000Z","fields": {"value": 4.2}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-04T16:00:00.000Z","fields": {"value": 4.3}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-05-04T16:59:00.000Z","fields": {"value": 4.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "names", "time": "2015-05-01T00:00:00.000Z","fields": {"first": "suzie", "last": "smith"}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "names", "time": "2015-05-01T08:00:00.000Z","fields": {"first": "frank", "last": "smith"}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "names", "time": "2015-05-01T16:00:00.000Z","fields": {"first": "jonny", "last": "jones"}}]}' -H "Content-Type: application/json" http://localhost:8086/write


curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "sensor", "time": "2015-05-01T00:00:00.000Z","fields": {"on": true, "off": false}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "sensor", "time": "2015-05-01T08:00:00.000Z","fields": {"on": false, "off": true}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "sensor", "time": "2015-05-01T16:00:00.000Z","fields": {"on": true, "off": false}}]}' -H "Content-Type: application/json" http://localhost:8086/write
