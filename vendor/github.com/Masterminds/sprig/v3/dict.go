package sprig

import (
	"github.com/imdario/mergo"
	"github.com/mitchellh/copystructure"
)

func get(d map[string]interface{}, key string) interface{} {
	if val, ok := d[key]; ok {
		return val
	}
	return ""
}

func set(d map[string]interface{}, key string, value interface{}) map[string]interface{} {
	d[key] = value
	return d
}

func unset(d map[string]interface{}, key string) map[string]interface{} {
	delete(d, key)
	return d
}

func hasKey(d map[string]interface{}, key string) bool {
	_, ok := d[key]
	return ok
}

func pluck(key string, d ...map[string]interface{}) []interface{} {
	res := []interface{}{}
	for _, dict := range d {
		if val, ok := dict[key]; ok {
			res = append(res, val)
		}
	}
	return res
}

func keys(dicts ...map[string]interface{}) []string {
	k := []string{}
	for _, dict := range dicts {
		for key := range dict {
			k = append(k, key)
		}
	}
	return k
}

func pick(dict map[string]interface{}, keys ...string) map[string]interface{} {
	res := map[string]interface{}{}
	for _, k := range keys {
		if v, ok := dict[k]; ok {
			res[k] = v
		}
	}
	return res
}

func omit(dict map[string]interface{}, keys ...string) map[string]interface{} {
	res := map[string]interface{}{}

	omit := make(map[string]bool, len(keys))
	for _, k := range keys {
		omit[k] = true
	}

	for k, v := range dict {
		if _, ok := omit[k]; !ok {
			res[k] = v
		}
	}
	return res
}

func dict(v ...interface{}) map[string]interface{} {
	dict := map[string]interface{}{}
	lenv := len(v)
	for i := 0; i < lenv; i += 2 {
		key := strval(v[i])
		if i+1 >= lenv {
			dict[key] = ""
			continue
		}
		dict[key] = v[i+1]
	}
	return dict
}

func merge(dst map[string]interface{}, srcs ...map[string]interface{}) interface{} {
	for _, src := range srcs {
		if err := mergo.Merge(&dst, src); err != nil {
			// Swallow errors inside of a template.
			return ""
		}
	}
	return dst
}

func mustMerge(dst map[string]interface{}, srcs ...map[string]interface{}) (interface{}, error) {
	for _, src := range srcs {
		if err := mergo.Merge(&dst, src); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

func mergeOverwrite(dst map[string]interface{}, srcs ...map[string]interface{}) interface{} {
	for _, src := range srcs {
		if err := mergo.MergeWithOverwrite(&dst, src); err != nil {
			// Swallow errors inside of a template.
			return ""
		}
	}
	return dst
}

func mustMergeOverwrite(dst map[string]interface{}, srcs ...map[string]interface{}) (interface{}, error) {
	for _, src := range srcs {
		if err := mergo.MergeWithOverwrite(&dst, src); err != nil {
			return nil, err
		}
	}
	return dst, nil
}

func values(dict map[string]interface{}) []interface{} {
	values := []interface{}{}
	for _, value := range dict {
		values = append(values, value)
	}

	return values
}

func deepCopy(i interface{}) interface{} {
	c, err := mustDeepCopy(i)
	if err != nil {
		panic("deepCopy error: " + err.Error())
	}

	return c
}

func mustDeepCopy(i interface{}) (interface{}, error) {
	return copystructure.Copy(i)
}
