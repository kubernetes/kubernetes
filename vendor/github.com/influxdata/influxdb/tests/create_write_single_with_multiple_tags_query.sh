echo "creating database"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE foo"

echo "creating retention policy"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY bar ON foo DURATION 1h REPLICATION 3 DEFAULT"

echo "inserting data"
curl -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "tags": {"region":"uswest","host": "server01"},"time": "2015-01-26T22:01:11.703Z","fields": {"value": 100}}]}' -H "Content-Type: application/json" http://localhost:8086/write

echo "querying data"
curl -G http://localhost:8086/query --data-urlencode "db=foo" --data-urlencode "q=SELECT sum(value) FROM \"foo\".\"bar\".cpu GROUP BY time(1h)"
