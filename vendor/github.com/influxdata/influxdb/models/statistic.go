package models

type Statistic struct {
	Name   string                 `json:"name"`
	Tags   map[string]string      `json:"tags"`
	Values map[string]interface{} `json:"values"`
}

func NewStatistic(name string) Statistic {
	return Statistic{
		Name:   name,
		Tags:   make(map[string]string),
		Values: make(map[string]interface{}),
	}
}

// StatisticTags is a map that can be merged with others without causing
// mutations to either map.
type StatisticTags map[string]string

// Merge creates a new map containing the merged contents of tags and t.
// If both tags and the receiver map contain the same key, the value in tags
// is used in the resulting map.
//
// Merge always returns a usable map.
func (t StatisticTags) Merge(tags map[string]string) map[string]string {
	// Add everything in tags to the result.
	out := make(map[string]string, len(tags))
	for k, v := range tags {
		out[k] = v
	}

	// Only add values from t that don't appear in tags.
	for k, v := range t {
		if _, ok := tags[k]; !ok {
			out[k] = v
		}
	}
	return out
}
