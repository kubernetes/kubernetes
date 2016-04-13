#!/bin/bash
set -e

echo "> creating database"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE foo"

echo ""
echo "> creating retention policy"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY bar ON foo DURATION 1h REPLICATION 1 DEFAULT"

echo ""
echo "> inserting data"
curl -v -X POST "http://localhost:8086/write?db=foo&rp=bar" -d 'cpu,host=server01 value=1.0'
curl -v -d '{"database" : "foo", "retentionPolicy" : "bar", "points": [{"measurement": "cpu", "tags": {"host": "server02"},"fields": {"value": 100}}]}' -H "Content-Type: application/json" http://localhost:8086/write

echo ""
echo "> querying data"
curl -G http://localhost:8086/query --data-urlencode "db=foo" --data-urlencode "pretty=true" --data-urlencode "q=SELECT * FROM \"foo\".\"bar\".cpu"

