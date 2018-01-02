

## Generating Sample JSON

Use [http://www.json-generator.com/](http://www.json-generator.com/)

```json
[
  '{{repeat(1000,1000)}}',
  {
    "database": "foo", 
    "retentionPolicy": "bar",
    "points": [
      {
        "name": "cpu", 
        "tags": {"host": "server01"},
        "time": "{{date(new Date(2015, 15, 1), new Date(), 'YYYY-MM-ddThh:mm:ss Z')}}",
        "fields": {
          "value": '{{integer(1, 1000)}}'
        }
      }
    ]
  }
]
```

You can curl the data with the following command:

```bash
cat sample.json | curl -d @- -H "Content-Type: application/json" http://localhost:8086/write
```
