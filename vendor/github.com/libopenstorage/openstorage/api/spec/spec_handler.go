package spec

import (
	"fmt"
	"regexp"
	"strconv"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/units"
)

// SpecHandler provides conversion function from what gets passed in over the
// plugin API to an api.VolumeSpec object.
type SpecHandler interface {
	// SpecFromString parses options from the name.
	// If the scheduler was unable to pass in the volume spec via the API,
	// the spec can be passed in via the name in the format:
	// "key=value;key=value;name=volname"
	// If the spec was parsed, it returns:
	//  	(true, parsed_spec, parsed_name)
	// If the input string didn't contain the string, it returns:
	// 	(false, DefaultSpec(), inputString)
	SpecFromString(inputString string) (bool, *api.VolumeSpec, string)

	// SpecFromOpts parses in docker options passed in the the docker run
	// command of the form --opt name=value
	// If the options are validated then it returns:
	// 	(resultant_VolumeSpec, nil)
	// If the options have invalid values then it returns:
	//	(nil, error)

	SpecFromOpts(opts map[string]string) (*api.VolumeSpec, error)
	// Returns a default VolumeSpec if no docker options or string encoding
	// was provided.
	DefaultSpec() *api.VolumeSpec
}

var (
	nameRegex       = regexp.MustCompile(api.Name + "=([0-9A-Za-z]+),?")
	sizeRegex       = regexp.MustCompile(api.SpecSize + "=([0-9A-Za-z]+),?")
	scaleRegex      = regexp.MustCompile(api.SpecScale + "=([0-9A-Za-z]+),?")
	fsRegex         = regexp.MustCompile(api.SpecFilesystem + "=([0-9A-Za-z]+),?")
	bsRegex         = regexp.MustCompile(api.SpecBlockSize + "=([0-9]+),?")
	haRegex         = regexp.MustCompile(api.SpecHaLevel + "=([0-9]+),?")
	cosRegex        = regexp.MustCompile(api.SpecPriority + "=([A-Za-z]+),?")
	sharedRegex     = regexp.MustCompile(api.SpecShared + "=([A-Za-z]+),?")
	passphraseRegex = regexp.MustCompile(api.SpecPassphrase + "=([0-9A-Za-z_@./#&+-]+),?")
)

type specHandler struct {
}

func NewSpecHandler() SpecHandler {
	return &specHandler{}
}

func (d *specHandler) cosLevel(cos string) (uint32, error) {
	switch cos {
	case "high", "3":
		return uint32(api.CosType_HIGH), nil
	case "medium", "2":
		return uint32(api.CosType_MEDIUM), nil
	case "low", "1", "":
		return uint32(api.CosType_LOW), nil
	}
	return uint32(api.CosType_LOW),
		fmt.Errorf("Cos must be one of %q | %q | %q", "high", "medium", "low")
}

func (d *specHandler) getVal(r *regexp.Regexp, str string) (bool, string) {
	found := r.FindString(str)
	if found == "" {
		return false, ""
	}

	submatches := r.FindStringSubmatch(str)
	if len(submatches) < 2 {
		return false, ""
	}

	val := submatches[1]

	return true, val
}

func (d *specHandler) DefaultSpec() *api.VolumeSpec {
	return &api.VolumeSpec{
		VolumeLabels: make(map[string]string),
		Format:       api.FSType_FS_TYPE_EXT4,
		HaLevel:      1,
	}
}

func (d *specHandler) SpecFromOpts(
	opts map[string]string,
) (*api.VolumeSpec, error) {
	spec := d.DefaultSpec()

	for k, v := range opts {
		switch k {
		case api.SpecEphemeral:
			spec.Ephemeral, _ = strconv.ParseBool(v)
		case api.SpecSize:
			if size, err := units.Parse(v); err != nil {
				return nil, err
			} else {
				spec.Size = uint64(size)
			}
		case api.SpecFilesystem:
			if value, err := api.FSTypeSimpleValueOf(v); err != nil {
				return nil, err
			} else {
				spec.Format = value
			}
		case api.SpecBlockSize:
			if blockSize, err := units.Parse(v); err != nil {
				return nil, err
			} else {
				spec.BlockSize = blockSize
			}
		case api.SpecHaLevel:
			haLevel, _ := strconv.ParseInt(v, 10, 64)
			spec.HaLevel = haLevel
		case api.SpecPriority:
			cos, _ := api.CosTypeSimpleValueOf(v)
			spec.Cos = cos
		case api.SpecDedupe:
			spec.Dedupe, _ = strconv.ParseBool(v)
		case api.SpecSnapshotInterval:
			snapshotInterval, _ := strconv.ParseUint(v, 10, 32)
			spec.SnapshotInterval = uint32(snapshotInterval)
		case api.SpecAggregationLevel:
			aggregationLevel, _ := strconv.ParseUint(v, 10, 32)
			spec.AggregationLevel = uint32(aggregationLevel)
		case api.SpecShared:
			if shared, err := strconv.ParseBool(v); err != nil {
				return nil, err
			} else {
				spec.Shared = shared
			}
		case api.SpecPassphrase:
			spec.Encrypted = true
			spec.Passphrase = v
		default:
			spec.VolumeLabels[k] = v
		}
	}
	return spec, nil
}

func (d *specHandler) SpecFromString(
	str string,
) (bool, *api.VolumeSpec, string) {
	// If we can't parse the name, the rest of the spec is invalid.
	ok, name := d.getVal(nameRegex, str)
	if !ok {
		return false, d.DefaultSpec(), str
	}

	opts := make(map[string]string)

	if ok, sz := d.getVal(sizeRegex, str); ok {
		opts[api.SpecSize] = sz
	}
	if ok, scale := d.getVal(scaleRegex, str); ok {
		opts[api.SpecScale] = scale
	}
	if ok, fs := d.getVal(fsRegex, str); ok {
		opts[api.SpecFilesystem] = fs
	}
	if ok, bs := d.getVal(bsRegex, str); ok {
		opts[api.SpecBlockSize] = bs
	}
	if ok, ha := d.getVal(haRegex, str); ok {
		opts[api.SpecHaLevel] = ha
	}
	if ok, priority := d.getVal(cosRegex, str); ok {
		opts[api.SpecPriority] = priority
	}
	if ok, shared := d.getVal(sharedRegex, str); ok {
		opts[api.SpecShared] = shared
	}
	if ok, passphrase := d.getVal(passphraseRegex, str); ok {
		opts[api.SpecPassphrase] = passphrase
	}

	spec, err := d.SpecFromOpts(opts)
	if err != nil {
		return false, d.DefaultSpec(), name
	}
	return true, spec, name
}
