package gmeasure

import (
	"crypto/md5"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"github.com/onsi/gomega/internal/gutil"
)

const CACHE_EXT = ".gmeasure-cache"

/*
ExperimentCache provides a director-and-file based cache of experiments
*/
type ExperimentCache struct {
	Path string
}

/*
NewExperimentCache creates and initializes a new cache.  Path must point to a directory (if path does not exist, NewExperimentCache will create a directory at path).

Cached Experiments are stored as separate files in the cache directory - the filename is a hash of the Experiment name.  Each file contains two JSON-encoded objects - a CachedExperimentHeader that includes the experiment's name and cache version number, and then the Experiment itself.
*/
func NewExperimentCache(path string) (ExperimentCache, error) {
	stat, err := os.Stat(path)
	if os.IsNotExist(err) {
		err := os.MkdirAll(path, 0777)
		if err != nil {
			return ExperimentCache{}, err
		}
	} else if !stat.IsDir() {
		return ExperimentCache{}, fmt.Errorf("%s is not a directory", path)
	}

	return ExperimentCache{
		Path: path,
	}, nil
}

/*
CachedExperimentHeader captures the name of the Cached Experiment and its Version
*/
type CachedExperimentHeader struct {
	Name    string
	Version int
}

func (cache ExperimentCache) hashOf(name string) string {
	return fmt.Sprintf("%x", md5.Sum([]byte(name)))
}

func (cache ExperimentCache) readHeader(filename string) (CachedExperimentHeader, error) {
	out := CachedExperimentHeader{}
	f, err := os.Open(filepath.Join(cache.Path, filename))
	if err != nil {
		return out, err
	}
	defer f.Close()
	err = json.NewDecoder(f).Decode(&out)
	return out, err
}

/*
List returns a list of all Cached Experiments found in the cache.
*/
func (cache ExperimentCache) List() ([]CachedExperimentHeader, error) {
	var out []CachedExperimentHeader
	names, err := gutil.ReadDir(cache.Path)
	if err != nil {
		return out, err
	}
	for _, name := range names {
		if filepath.Ext(name) != CACHE_EXT {
			continue
		}
		header, err := cache.readHeader(name)
		if err != nil {
			return out, err
		}
		out = append(out, header)
	}
	return out, nil
}

/*
Clear empties out the cache - this will delete any and all detected cache files in the cache directory.  Use with caution!
*/
func (cache ExperimentCache) Clear() error {
	names, err := gutil.ReadDir(cache.Path)
	if err != nil {
		return err
	}
	for _, name := range names {
		if filepath.Ext(name) != CACHE_EXT {
			continue
		}
		err := os.Remove(filepath.Join(cache.Path, name))
		if err != nil {
			return err
		}
	}
	return nil
}

/*
Load fetches an experiment from the cache.  Lookup occurs by name.  Load requires that the version number in the cache is equal to or greater than the passed-in version.

If an experiment with corresponding name and version >= the passed-in version is found, it is unmarshaled and returned.

If no experiment is found, or the cached version is smaller than the passed-in version, Load will return nil.

When paired with Ginkgo you can cache experiments and prevent potentially expensive recomputation with this pattern:

		const EXPERIMENT_VERSION = 1 //bump this to bust the cache and recompute _all_ experiments

	    Describe("some experiments", func() {
	    	var cache gmeasure.ExperimentCache
	    	var experiment *gmeasure.Experiment

	    	BeforeEach(func() {
	    		cache = gmeasure.NewExperimentCache("./gmeasure-cache")
	    		name := CurrentSpecReport().LeafNodeText
	    		experiment = cache.Load(name, EXPERIMENT_VERSION)
	    		if experiment != nil {
	    			AddReportEntry(experiment)
	    			Skip("cached")
	    		}
	    		experiment = gmeasure.NewExperiment(name)
				AddReportEntry(experiment)
	    	})

	    	It("foo runtime", func() {
	    		experiment.SampleDuration("runtime", func() {
	    			//do stuff
	    		}, gmeasure.SamplingConfig{N:100})
	    	})

	    	It("bar runtime", func() {
	    		experiment.SampleDuration("runtime", func() {
	    			//do stuff
	    		}, gmeasure.SamplingConfig{N:100})
	    	})

	    	AfterEach(func() {
	    		if !CurrentSpecReport().State.Is(types.SpecStateSkipped) {
		    		cache.Save(experiment.Name, EXPERIMENT_VERSION, experiment)
		    	}
	    	})
	    })
*/
func (cache ExperimentCache) Load(name string, version int) *Experiment {
	path := filepath.Join(cache.Path, cache.hashOf(name)+CACHE_EXT)
	f, err := os.Open(path)
	if err != nil {
		return nil
	}
	defer f.Close()
	dec := json.NewDecoder(f)
	header := CachedExperimentHeader{}
	dec.Decode(&header)
	if header.Version < version {
		return nil
	}
	out := NewExperiment("")
	err = dec.Decode(out)
	if err != nil {
		return nil
	}
	return out
}

/*
Save stores the passed-in experiment to the cache with the passed-in name and version.
*/
func (cache ExperimentCache) Save(name string, version int, experiment *Experiment) error {
	path := filepath.Join(cache.Path, cache.hashOf(name)+CACHE_EXT)
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	enc := json.NewEncoder(f)
	err = enc.Encode(CachedExperimentHeader{
		Name:    name,
		Version: version,
	})
	if err != nil {
		return err
	}
	return enc.Encode(experiment)
}

/*
Delete removes the experiment with the passed-in name from the cache
*/
func (cache ExperimentCache) Delete(name string) error {
	path := filepath.Join(cache.Path, cache.hashOf(name)+CACHE_EXT)
	return os.Remove(path)
}
