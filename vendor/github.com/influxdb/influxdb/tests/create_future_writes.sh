now=$(date '+%FT%T.000Z')
tomorrow=$(date -v +1d '+%FT%T.000Z')

echo "creating database"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE DATABASE foo"

echo "creating retention policy"
curl -G http://localhost:8086/query --data-urlencode "q=CREATE RETENTION POLICY bar ON foo DURATION INF REPLICATION 1 DEFAULT"

echo "inserting data"
curl -d "{\"database\" : \"foo\", \"retentionPolicy\" : \"bar\", \"points\": [{\"measurement\": \"cpu\", \"tags\": {\"region\":\"uswest\",\"host\": \"server01\"},\"time\": \"$now\",\"fields\": {\"value\": 100}}]}" -H "Content-Type: application/json" http://localhost:8086/write

echo "inserting data"
curl -d "{\"database\" : \"foo\", \"retentionPolicy\" : \"bar\", \"points\": [{\"measurement\": \"cpu\", \"tags\": {\"region\":\"uswest\",\"host\": \"server01\"},\"time\": \"$tomorrow\",\"fields\": {\"value\": 200}}]}" -H "Content-Type: application/json" http://localhost:8086/write

sleep 1

echo "querying data"
curl -G http://localhost:8086/query --data-urlencode "db=foo" --data-urlencode "q=SELECT count(value) FROM \"foo\".\"bar\".cpu"

echo "querying data"
curl -G http://localhost:8086/query --data-urlencode "db=foo" --data-urlencode "q=SELECT count(value) FROM \"foo\".\"bar\".cpu where time < now() + 10d"
