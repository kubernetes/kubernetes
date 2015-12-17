echo "creating database"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE foo"

echo "creating retention policy"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY bar ON foo DURATION 300d REPLICATION 3 DEFAULT"

echo "inserting data"
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-02-26T22:01:11.703Z","fields": {"value": 8.9}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-02-27T22:01:11.703Z","fields": {"value": 1.3}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "time": "2015-02-28T22:01:11.703Z","fields": {"value": 50.4}}]}' -H "Content-Type: application/json" http://localhost:8086/write


curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "mem", "tags": {"host": "server01"},"time": "2015-02-26T22:01:11.703Z","fields": {"value": 16432}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "mem", "tags": {"host": "server01"},"time": "2015-02-27T22:01:11.703Z","fields": {"value": 23453}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "mem", "tags": {"host": "server02"},"time": "2015-02-28T22:01:11.703Z","fields": {"value": 90234}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "temp", "tags": {"host": "server01","region":"uswest"}, "time": "2015-02-26T22:01:11.703Z","fields": {"value": 98.6}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "temp", "tags": {"host": "server01","region":"useast"}, "time": "2015-02-27T22:01:11.703Z","fields": {"value": 101.1}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "temp", "tags": {"host": "server02","region":"useast"}, "time": "2015-02-28T22:01:11.703Z","fields": {"value": 105.4}}]}' -H "Content-Type: application/json" http://localhost:8086/write

curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "network", "tags": {"host": "server01","region":"uswest"},"time": "2015-02-26T22:01:11.703Z","fields": {"rx": 2342,"tx": 9804}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "network", "tags": {"host": "server01","region":"useast"},"time": "2015-02-27T22:01:11.703Z","fields": {"rx": 4324,"tx": 7930}}]}' -H "Content-Type: application/json" http://localhost:8086/write
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "network", "tags": {"host": "server02","region":"useast"},"time": "2015-02-28T22:01:11.703Z","fields": {"rx": 2342,"tx": 8234}}]}' -H "Content-Type: application/json" http://localhost:8086/write
