package test

import (
	"encoding/json"
	"github.com/json-iterator/go"
	"io/ioutil"
	"os"
	"testing"
)

//func Test_large_file(t *testing.T) {
//	file, err := os.Open("/tmp/large-file.json")
//	if err != nil {
//		t.Fatal(err)
//	}
//	iter := Parse(file, 4096)
//	count := 0
//	for iter.ReadArray() {
//		iter.Skip()
//		count++
//	}
//	if count != 11351 {
//		t.Fatal(count)
//	}
//}

func init() {
	ioutil.WriteFile("/tmp/large-file.json", []byte(`[{
  "person": {
    "id": "d50887ca-a6ce-4e59-b89f-14f0b5d03b03",
    "name": {
      "fullName": "Leonid Bugaev",
      "givenName": "Leonid",
      "familyName": "Bugaev"
    },
    "email": "leonsbox@gmail.com",
    "gender": "male",
    "location": "Saint Petersburg, Saint Petersburg, RU",
    "geo": {
      "city": "Saint Petersburg",
      "state": "Saint Petersburg",
      "country": "Russia",
      "lat": 59.9342802,
      "lng": 30.3350986
    },
    "bio": "Senior engineer at Granify.com",
    "site": "http://flickfaver.com",
    "avatar": "https://d1ts43dypk8bqh.cloudfront.net/v1/avatars/d50887ca-a6ce-4e59-b89f-14f0b5d03b03",
    "employment": {
      "name": "www.latera.ru",
      "title": "Software Engineer",
      "domain": "gmail.com"
    },
    "facebook": {
      "handle": "leonid.bugaev"
    },
    "github": {
      "handle": "buger",
      "id": 14009,
      "avatar": "https://avatars.githubusercontent.com/u/14009?v=3",
      "company": "Granify",
      "blog": "http://leonsbox.com",
      "followers": 95,
      "following": 10
    },
    "twitter": {
      "handle": "flickfaver",
      "id": 77004410,
      "bio": null,
      "followers": 2,
      "following": 1,
      "statuses": 5,
      "favorites": 0,
      "location": "",
      "site": "http://flickfaver.com",
      "avatar": null
    },
    "linkedin": {
      "handle": "in/leonidbugaev"
    },
    "googleplus": {
      "handle": null
    },
    "angellist": {
      "handle": "leonid-bugaev",
      "id": 61541,
      "bio": "Senior engineer at Granify.com",
      "blog": "http://buger.github.com",
      "site": "http://buger.github.com",
      "followers": 41,
      "avatar": "https://d1qb2nb5cznatu.cloudfront.net/users/61541-medium_jpg?1405474390"
    },
    "klout": {
      "handle": null,
      "score": null
    },
    "foursquare": {
      "handle": null
    },
    "aboutme": {
      "handle": "leonid.bugaev",
      "bio": null,
      "avatar": null
    },
    "gravatar": {
      "handle": "buger",
      "urls": [
      ],
      "avatar": "http://1.gravatar.com/avatar/f7c8edd577d13b8930d5522f28123510",
      "avatars": [
        {
          "url": "http://1.gravatar.com/avatar/f7c8edd577d13b8930d5522f28123510",
          "type": "thumbnail"
        }
      ]
    },
    "fuzzy": false
  },
  "company": "hello"
}]`), 0666)
}

/*
200000	      8886 ns/op	    4336 B/op	       6 allocs/op
50000	     34244 ns/op	    6744 B/op	      14 allocs/op
*/
func Benchmark_jsoniter_large_file(b *testing.B) {
	b.ReportAllocs()
	for n := 0; n < b.N; n++ {
		file, _ := os.Open("/tmp/large-file.json")
		iter := jsoniter.Parse(jsoniter.ConfigDefault, file, 4096)
		count := 0
		iter.ReadArrayCB(func(iter *jsoniter.Iterator) bool {
			// Skip() is strict by default, use --tags jsoniter-sloppy to skip without validation
			iter.Skip()
			count++
			return true
		})
		file.Close()
		if iter.Error != nil {
			b.Error(iter.Error)
		}
	}
}

func Benchmark_json_large_file(b *testing.B) {
	b.ReportAllocs()
	for n := 0; n < b.N; n++ {
		file, _ := os.Open("/tmp/large-file.json")
		bytes, _ := ioutil.ReadAll(file)
		file.Close()
		result := []struct{}{}
		err := json.Unmarshal(bytes, &result)
		if err != nil {
			b.Error(err)
		}
	}
}
