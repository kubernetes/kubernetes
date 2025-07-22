package spec

import (
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/libopenstorage/openstorage/api"
	"github.com/libopenstorage/openstorage/pkg/parser"
	"github.com/libopenstorage/openstorage/pkg/units"
)

// SpecHandler provides conversion function from what gets passed in over the
// plugin API to an api.VolumeSpec object.
type SpecHandler interface {
	// SpecFromString parses options from the name.
	// If the scheduler was unable to pass in the volume spec via the API,
	// the spec can be passed in via the name in the format:
	// "key=value;key=value;name=volname"
	// source is populated if key parent=<volume_id> is specified.
	// If the spec was parsed, it returns:
	//  	(true, parsed_spec, locator, source, parsed_name)
	// If the input string didn't contain the string, it returns:
	// 	(false, DefaultSpec(), nil, nil, inputString)
	SpecFromString(inputString string) (
		bool,
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		string,
	)

	// SpecFromOpts parses in docker options passed in the the docker run
	// command of the form --opt name=value
	// source is populated if --opt parent=<volume_id> is specified.
	// If the options are validated then it returns:
	// 	(resultant_VolumeSpec, source, locator, nil)
	// If the options have invalid values then it returns:
	//	(nil, nil, nil, error)

	SpecFromOpts(opts map[string]string) (
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		error,
	)

	// UpdateSpecFromOpts parses in volume options passed through the opts map and updates given spec, locator & source
	// If the options are validated then it returns:
	// 	(resultant_VolumeSpec, source, locator, nil)
	// If the options have invalid values then it returns:
	//	(nil, nil, nil, error)
	UpdateSpecFromOpts(opts map[string]string, spec *api.VolumeSpec, locator *api.VolumeLocator, source *api.Source) (
		*api.VolumeSpec,
		*api.VolumeLocator,
		*api.Source,
		error,
	)

	// Returns a default VolumeSpec if no docker options or string encoding
	// was provided.
	DefaultSpec() *api.VolumeSpec
}

var (
	nameRegex       = regexp.MustCompile(api.Name + "=([0-9A-Za-z_-]+),?")
	nodesRegex      = regexp.MustCompile(api.SpecNodes + "=([0-9A-Za-z_-]+),?")
	sizeRegex       = regexp.MustCompile(api.SpecSize + "=([0-9A-Za-z]+),?")
	scaleRegex      = regexp.MustCompile(api.SpecScale + "=([0-9]+),?")
	fsRegex         = regexp.MustCompile(api.SpecFilesystem + "=([0-9A-Za-z]+),?")
	bsRegex         = regexp.MustCompile(api.SpecBlockSize + "=([0-9]+),?")
	haRegex         = regexp.MustCompile(api.SpecHaLevel + "=([0-9]+),?")
	cosRegex        = regexp.MustCompile(api.SpecPriority + "=([A-Za-z]+),?")
	sharedRegex     = regexp.MustCompile(api.SpecShared + "=([A-Za-z]+),?")
	passphraseRegex = regexp.MustCompile(api.SpecPassphrase + "=([0-9A-Za-z_@./#&+-]+),?")
	stickyRegex     = regexp.MustCompile(api.SpecSticky + "=([A-Za-z]+),?")
	secureRegex     = regexp.MustCompile(api.SpecSecure + "=([A-Za-z]+),?")
	zonesRegex      = regexp.MustCompile(api.SpecZones + "=([A-Za-z]+),?")
	racksRegex      = regexp.MustCompile(api.SpecRacks + "=([A-Za-z]+),?")
	aggrRegex       = regexp.MustCompile(api.SpecAggregationLevel + "=([0-9]+|" +
		api.SpecAutoAggregationValue + "),?")
	compressedRegex   = regexp.MustCompile(api.SpecCompressed + "=([A-Za-z]+),?")
	snapScheduleRegex = regexp.MustCompile(api.SpecSnapshotSchedule +
		`=([A-Za-z0-9:;@=#]+),?`)
)

type specHandler struct {
}

// NewSpecHandler returns a new SpecHandler interface
func NewSpecHandler() SpecHandler {
	return &specHandler{}
}

func (d *specHandler) cosLevel(cos string) (api.CosType, error) {
	cos = strings.ToLower(cos)
	switch cos {
	case "high", "3":
		return api.CosType_HIGH, nil
	case "medium", "2":
		return api.CosType_MEDIUM, nil
	case "low", "1", "":
		return api.CosType_LOW, nil
	}
	return api.CosType_NONE,
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

func (d *specHandler) UpdateSpecFromOpts(opts map[string]string, spec *api.VolumeSpec, locator *api.VolumeLocator,
	source *api.Source) (*api.VolumeSpec, *api.VolumeLocator, *api.Source, error) {
	nodeList := make([]string, 0)

	if spec == nil {
		spec = d.DefaultSpec()
	}

	if source == nil {
		source = &api.Source{}
	}

	if locator == nil {
		locator = &api.VolumeLocator{
			VolumeLabels: make(map[string]string),
		}
	}

	for k, v := range opts {
		switch k {
		case api.SpecNodes:
			inputNodes := strings.Split(v, ",")
			for _, node := range inputNodes {
				if len(node) != 0 {
					nodeList = append(nodeList, node)
				}
			}
			spec.ReplicaSet = &api.ReplicaSet{Nodes: nodeList}
		case api.SpecParent:
			source.Parent = v
		case api.SpecEphemeral:
			spec.Ephemeral, _ = strconv.ParseBool(v)
		case api.SpecSize:
			if size, err := units.Parse(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Size = uint64(size)
			}
		case api.SpecScale:
			if scale, err := strconv.ParseUint(v, 10, 64); err == nil {
				spec.Scale = uint32(scale)
			}

		case api.SpecFilesystem:
			if value, err := api.FSTypeSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Format = value
			}
		case api.SpecBlockSize:
			if blockSize, err := units.Parse(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.BlockSize = blockSize
			}
		case api.SpecHaLevel:
			haLevel, _ := strconv.ParseInt(v, 10, 64)
			spec.HaLevel = haLevel
		case api.SpecPriority:
			cos, err := d.cosLevel(v)
			if err != nil {
				return nil, nil, nil, err
			}
			spec.Cos = cos
		case api.SpecPriorityAlias:
			cos, err := d.cosLevel(v)
			if err != nil {
				return nil, nil, nil, err
			}
			spec.Cos = cos
		case api.SpecDedupe:
			spec.Dedupe, _ = strconv.ParseBool(v)
		case api.SpecSnapshotInterval:
			snapshotInterval, _ := strconv.ParseUint(v, 10, 32)
			spec.SnapshotInterval = uint32(snapshotInterval)
		case api.SpecSnapshotSchedule:
			spec.SnapshotSchedule = v
		case api.SpecAggregationLevel:
			if v == api.SpecAutoAggregationValue {
				spec.AggregationLevel = api.AutoAggregation
			} else {
				aggregationLevel, _ := strconv.ParseUint(v, 10, 32)
				spec.AggregationLevel = uint32(aggregationLevel)
			}
		case api.SpecShared:
			if shared, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Shared = shared
			}
		case api.SpecSticky:
			if sticky, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Sticky = sticky
			}
		case api.SpecSecure:
			if secure, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Encrypted = secure
			}
		case api.SpecPassphrase:
			spec.Encrypted = true
			spec.Passphrase = v
		case api.SpecGroup:
			spec.Group = &api.Group{Id: v}
		case api.SpecGroupEnforce:
			if groupEnforced, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.GroupEnforced = groupEnforced
			}
		case api.SpecZones, api.SpecRacks:
			locator.VolumeLabels[k] = v
		case api.SpecCompressed:
			if compressed, err := strconv.ParseBool(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.Compressed = compressed
			}
		case api.SpecLabels:
			if labels, err := parser.LabelsFromString(v); err != nil {
				return nil, nil, nil, err
			} else {
				for k, v := range labels {
					locator.VolumeLabels[k] = v
				}
			}
		case api.SpecIoProfile:
			if ioProfile, err := api.IoProfileSimpleValueOf(v); err != nil {
				return nil, nil, nil, err
			} else {
				spec.IoProfile = ioProfile
			}
		default:
			spec.VolumeLabels[k] = v
		}
	}
	return spec, locator, source, nil
}

func (d *specHandler) SpecFromOpts(
	opts map[string]string,
) (*api.VolumeSpec, *api.VolumeLocator, *api.Source, error) {
	source := &api.Source{}
	locator := &api.VolumeLocator{
		VolumeLabels: make(map[string]string),
	}

	spec := d.DefaultSpec()
	return d.UpdateSpecFromOpts(opts, spec, locator, source)
}

func (d *specHandler) SpecFromString(
	str string,
) (bool, *api.VolumeSpec, *api.VolumeLocator, *api.Source, string) {
	// If we can't parse the name, the rest of the spec is invalid.
	ok, name := d.getVal(nameRegex, str)
	if !ok {
		return false, d.DefaultSpec(), nil, nil, str
	}

	opts := make(map[string]string)

	if ok, sz := d.getVal(sizeRegex, str); ok {
		opts[api.SpecSize] = sz
	}
	if ok, nodes := d.getVal(nodesRegex, str); ok {
		opts[api.SpecNodes] = nodes
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
	if ok, sticky := d.getVal(stickyRegex, str); ok {
		opts[api.SpecSticky] = sticky
	}
	if ok, secure := d.getVal(secureRegex, str); ok {
		opts[api.SpecSecure] = secure
	}
	if ok, passphrase := d.getVal(passphraseRegex, str); ok {
		opts[api.SpecPassphrase] = passphrase
	}
	if ok, zones := d.getVal(zonesRegex, str); ok {
		opts[api.SpecZones] = zones
	}
	if ok, racks := d.getVal(racksRegex, str); ok {
		opts[api.SpecRacks] = racks
	}
	if ok, aggregationLvl := d.getVal(aggrRegex, str); ok {
		opts[api.SpecAggregationLevel] = aggregationLvl
	}
	if ok, compressed := d.getVal(compressedRegex, str); ok {
		opts[api.SpecCompressed] = compressed
	}
	if ok, sched := d.getVal(snapScheduleRegex, str); ok {
		opts[api.SpecSnapshotSchedule] = strings.Replace(sched, "#", ",", -1)
	}

	spec, locator, source, err := d.SpecFromOpts(opts)
	if err != nil {
		return false, d.DefaultSpec(), nil, nil, name
	}
	return true, spec, locator, source, name
}
